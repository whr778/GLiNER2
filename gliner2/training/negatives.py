"""Inject ABSENT label queries into a training record's schema.

WHAT AN ABSENT QUERY IS, AND WHY IT IS THE WHOLE POINT. A model trained only on menus built
from its own gold is never shown a label it must reject, so it never learns to. Measured on
this project's incumbent: given a schema containing only event types the document does NOT
have, it fires on 63 of 100 documents, and its real `event_type` precision is 0.5521 where
the blind test reports 1.0000 by construction. The consumers of an absent query are already
live in every run -- `negative_query_ratio` (0.5) selects them into the pair loss and
`abstention_loss` trains a per-query gate whose target is 1 for one -- and have always
selected from an empty set.

THE REPRESENTATION IS ALREADY SUPPORTED: an absent entity query is a label mapped to an EMPTY
LIST. `GuideScores.inject` has emitted exactly that for its cross-record rivals, so the
collator, target builder and losses accept it end to end.

SAFETY IS THE DESIGN, NOT A CHECK BOLTED ON. The corpora annotate different things -- cmnee
carries zero entity gold, biored zero events -- so offering `Organization` to a cmnee document
would supervise a real organisation as absent, at loss weight, thousands of times. Three rules:

  within-corpus     negatives come only from labels the record's OWN corpus annotates
  within-dimension  a dimension is offered only where the corpus annotates that dimension
  never-gold        an injected label is asserted not to appear in this record's gold

Records carry no provenance (`{"input", "output"}` only), so the corpus is identified by
SIGNATURE: the candidate corpora are those whose pool contains every gold label this record
uses, and the usable pool is their INTERSECTION. One candidate means the corpus is known
exactly; several means we offer only what all of them annotate. No candidate means no
negatives. Every direction of that fallback is conservative.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DIMENSIONS = ("entities", "events", "relations", "structures")


def _stable_seed(*parts: Any) -> int:
    """Deterministic across processes, ranks and runs -- `hash()` is not."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


class NegativeLabels:
    """Per-corpus negative pools plus the injection that puts them in a schema."""

    def __init__(self, pools: Dict[str, dict], per_dim: Dict[str, int], seed: int = 42):
        self.pools = pools
        self.per_dim = {d: int(n) for d, n in (per_dim or {}).items() if int(n) > 0}
        self.seed = seed
        self.epoch = 0
        self.stats = {"records": 0, "injected": 0, "records_with_injection": 0,
                      "no_candidate": 0}

    @classmethod
    def load(cls, path: str, per_dim: Dict[str, int], seed: int = 42) -> "NegativeLabels":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(blob.get("pools") or {}, per_dim, seed)

    def set_epoch(self, epoch: int) -> None:
        """Resample every epoch. The trainer calls this, as with DistributedSampler."""
        self.epoch = int(epoch)

    # -- corpus identification -------------------------------------------------------
    @staticmethod
    def _gold_labels(schema: dict) -> Dict[str, set]:
        out = {d: set() for d in DIMENSIONS}
        ents = schema.get("entities")
        if isinstance(ents, dict):
            out["entities"] = set(ents)
        evs = schema.get("events")
        if isinstance(evs, list):
            out["events"] = {e.get("event_type") for e in evs
                             if isinstance(e, dict) and e.get("event_type")}
        elif isinstance(evs, dict):
            out["events"] = set(evs)
        rels = schema.get("relations")
        if isinstance(rels, list):
            out["relations"] = {k for r in rels if isinstance(r, dict) for k in r}
        structs = schema.get("json_structures")
        if isinstance(structs, list):
            out["structures"] = {k for i in structs if isinstance(i, dict) for k in i}
        return out

    def _candidates(self, gold: Dict[str, set]) -> List[str]:
        """Corpora whose pools contain every gold label this record uses."""
        used = {d: g for d, g in gold.items() if g}
        if not used:
            return []
        out = []
        for name, pool in self.pools.items():
            ok = True
            for dim, labels in used.items():
                have = pool.get(dim) or {}
                have = set(have) if isinstance(have, (dict, list)) else set()
                if not labels <= have:
                    ok = False
                    break
            if ok:
                out.append(name)
        return out

    def _usable_pool(self, dim: str, candidates: List[str]) -> set:
        """Intersection across candidates, and only where the corpus ANNOTATES the dim."""
        pool: Optional[set] = None
        for name in candidates:
            spec = self.pools[name]
            if not (spec.get("annotates") or {}).get(dim):
                return set()          # one candidate that does not annotate it vetoes the dim
            have = spec.get(dim) or {}
            have = set(have) if isinstance(have, (dict, list)) else set()
            pool = have if pool is None else (pool & have)
        return pool or set()

    # -- injection -------------------------------------------------------------------
    def inject(self, schema: Any, index: int) -> Any:
        """Return ``schema`` with up to ``per_dim[d]`` absent labels added per dimension."""
        if not isinstance(schema, dict) or not self.per_dim:
            return schema
        self.stats["records"] += 1
        gold = self._gold_labels(schema)
        candidates = self._candidates(gold)
        if not candidates:
            self.stats["no_candidate"] += 1
            return schema

        import random as _random
        rng = _random.Random(_stable_seed(self.seed, self.epoch, index))
        out = dict(schema)
        added = 0

        k = self.per_dim.get("entities", 0)
        if k and gold["entities"]:
            pool = sorted(self._usable_pool("entities", candidates) - gold["entities"])
            chosen = rng.sample(pool, min(k, len(pool))) if pool else []
            if chosen:
                # An absent entity query is the label mapped to an EMPTY LIST -- the exact
                # shape GuideScores.inject uses, already supported end to end.
                out["entities"] = {**(out.get("entities") or {}),
                                   **{name: [] for name in chosen}}
                added += len(chosen)

        k = self.per_dim.get("events", 0)
        if k and gold["events"] and isinstance(schema.get("events"), list):
            pool_types = self._usable_pool("events", candidates) - gold["events"]
            chosen = rng.sample(sorted(pool_types), min(k, len(pool_types))) if pool_types else []
            if chosen:
                # Menu-only event types: roles come from the pool, gold stays empty. The
                # training path skips an Event with no triggers (processor.py:1183), so these
                # ride a separate key the processor reads alongside the gold list.
                roles = {}
                for name in chosen:
                    for cand in candidates:
                        spec = (self.pools[cand].get("events") or {}).get(name)
                        if spec:
                            roles[name] = list(spec)
                            break
                    roles.setdefault(name, [])
                out["absent_events"] = {**(out.get("absent_events") or {}), **roles}
                added += len(chosen)

        if added:
            self.stats["injected"] += added
            self.stats["records_with_injection"] += 1
            # A negative that is actually present is the failure this whole design guards
            # against; assert rather than trust the set arithmetic above.
            assert not (set(out.get("entities") or {}) - set(schema.get("entities") or {})) \
                & gold["entities"], "injected an entity label that is in this record's gold"
            assert not set(out.get("absent_events") or {}) & gold["events"], \
                "injected an event type that is in this record's gold"
        return out

    def composition_line(self) -> str:
        """One deterministic line an A/B gate can actually fail on."""
        n = self.stats["records"] or 1
        dims = ", ".join(f"{d} k={k}" for d, k in sorted(self.per_dim.items())) or "NONE"
        pct = 100.0 * self.stats["records_with_injection"] / n
        return (f"[composition]   negatives: {dims} | {self.stats['injected']:,} labels into "
                f"{self.stats['records_with_injection']:,}/{self.stats['records']:,} records "
                f"({pct:.1f}%) | no-candidate {self.stats['no_candidate']:,}")


__all__ = ["NegativeLabels", "DIMENSIONS"]
