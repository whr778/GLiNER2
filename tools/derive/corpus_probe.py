"""Measure a pre-split corpus (<base>.{train,val,test}.jsonl) for config derivation.

Composes existing measurements instead of re-implementing them: per-head gold and label
inventories from scripts/dataset_metrics.py, split hygiene from its `uniqueness`, gold
capacity from tools/train/size_gold_capacity.py, and document lengths from the BASE
MODEL's own tokenizer (lengths in another tokenizer's subwords would size the wrong window).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tools/train")]
import dataset_metrics as dm  # noqa: E402
import size_gold_capacity as sgc  # noqa: E402

SPLITS = ("train", "val", "test")


def split_paths(base: str) -> Dict[str, Path]:
    paths = {s: Path(f"{base}.{s}.jsonl") for s in SPLITS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"[corpus] missing split files: {missing}")
    return paths


def heads(m: Dict[str, Any]) -> Dict[str, int]:
    """Gold counts per head for one split."""
    return {
        "records": m["records"],
        "entity_mentions": sum(m["entities"].values()),
        "relations": sum(m["relations"].values()),
        "event_instances": m["ev_instances"],
        "trigger_gold": m["trigger_gold"],
        "argument_gold": m["argument_gold"],
        "classification_tasks": len(m["classifications"]),
        "structure_records": sum(m["structure_records"].values()),
    }


def labels(m: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Label inventory per category, with use counts (train split)."""
    return {
        "entities": dict(m["entities"]),
        "relations": dict(m["relations"]),
        "event_types": dict(m["event_triggers"]),
        "roles": {r: sum(c.values()) if isinstance(c, dict) else c
                  for r, c in _roles(m["event_roles"]).items()},
        "classifications": {t: dict(c) for t, c in m["classifications"].items()},
    }


def _roles(event_roles) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for _etype, roles in event_roles.items():
        for role, n in roles.items():
            out[role] = out.get(role, 0) + n
    return out


def lengths(path: Path, tokenizer) -> Dict[str, int]:
    """Subword length percentiles of the inputs, in the base model's tokenizer."""
    n = sorted(len(tokenizer(json.loads(l)["input"], add_special_tokens=False)["input_ids"])
               for l in open(path, encoding="utf-8") if l.strip())
    pick = lambda q: n[min(len(n) - 1, int(q * len(n)))]
    return {"docs": len(n), "p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": n[-1]}


def probe_corpus(base: str, tokenizer, event_records: bool) -> Dict[str, Any]:
    paths = split_paths(base)
    scans = {s: dm.scan_file(p) for s, p in paths.items()}
    counts, _ = sgc.measure([paths["train"]], count_events=not event_records)
    cap, pct_over_32, biggest = sgc.recommend(counts)   # cap None => the default 32 suffices
    return {
        "base": base,
        "heads": {s: heads(m) for s, m in scans.items()},
        "uniqueness": dm.uniqueness({"corpus": scans}),
        "labels": labels(scans["train"]),
        "labels_by_split": {s: labels(m) for s, m in scans.items()},
        "lengths": {s: lengths(p, tokenizer) for s, p in paths.items()},
        "gold_capacity": {"recommended_cap": cap or 32, "pct_groups_over_32": round(pct_over_32, 3),
                          "largest_group": biggest, "sampled_docs": sgc.SAMPLE,
                          "events_counted": not event_records},
    }
