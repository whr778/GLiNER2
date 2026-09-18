"""Derive an ENTITY corpus from event ARGUMENT roles (Option 4, the argument half).

`events_to_entities.py` does the TRIGGER half -- `event_type` becomes an entity label over
the trigger surface, which is how `maven_ner` was built. This is the ARGUMENT half: a role's
filler becomes a typed span, so a corpus with NO entity gold at all manufactures entity
supervision. cmnee has 9,281 records, 62,573 argument mentions and zero entity gold.

WHY IT EXISTS. Roughly a third of `event_argument` loss is spans that are NEVER PROPOSED --
measured directly on 50 cmnee documents: 96% of `Subject`, 71% of `Location` and 67% of
`Date` gold surfaces are not put forward by the entity head at all. Options 1 and 3 both
constrain BINDING and cannot recover a span that was never proposed; this adds extraction
supervision instead.

ROLE IS NOT TYPE. 19.5% of cmnee documents tag one surface with more than one role -- a
submarine is `Equipment`, `Materials` AND `Subject` in the same document. A thing's TYPE does
not change within a document; its ROLE does. And 9 of 11 role names collide case-insensitively
with a base entity label of DIFFERENT meaning (`Subject` tags subject MATTER in the base --
"Visual Arts" -- and the ACTOR in cmnee).

`--mode namespaced` (the DEFAULT) answers both by refusing to claim these are types: every role
becomes `Event<Role>`, a label of its own. Measured against the base's 2,092-label entity
vocabulary, none of the eleven collide. Two consequences follow, and they are the point:

* **The same-document conflict DISSOLVES.** `EventEquipment` and `EventSubject` on one span is
  ordinary multi-label role annotation, not contradictory typing. Nothing is dropped, so the
  supply is the whole 62,573 rather than the 15,188 a type-claiming derivation survives with --
  including `Subject`, the largest role at 20,743 and the one the entity head fails hardest on
  (96% of its gold surfaces are never proposed at all).
* **It buys PROPOSAL, not typing.** Option 4 exists to move the never-proposed third; a
  namespaced label still trains the entity head to put the span forward. It does NOT transfer
  to a real NER taxonomy, and it stays tautological for option 2 -- use `--mode canonical` or
  `hybrid` when transfer is what is wanted.

`--mode canonical` is the narrow, type-claiming derivation: only the four adjudicated-compatible
roles (`Date`, `Location`, `Quantity`, `Area`->`Location`), 24.3% of the supply, emitted as
canonical labels so the supervision transfers. `--mode hybrid` takes those four canonical and
namespaces the other seven.

Labels are emitted DIRECTLY in their final spelling -- `Area` is written as `Location`, never
written as `Area` and remapped. The `labels_file` indirection exists to protect SOURCE corpora
from being rewritten; a derived corpus is new data we author, so nothing is bent by naming it
correctly at birth.

THE OUTPUT IS PARTIAL and must be declared so. It labels Date/Location/Quantity and leaves
every other entity in the document unlabelled, so it is a source of entity POSITIVES and never
of entity NEGATIVES. Register it under `data.partial_annotation` before
`build_negative_pools.py` ever sees it, or it auto-qualifies as an entity annotator and
silently poisons the pool for every model trained afterwards.

    uv run python tools/data/roles_to_entities.py \
        --train data/cmnee.train.jsonl --val data/cmnee.val.jsonl \
        --test data/cmnee.test.jsonl --out-base data/cmnee_roles_ner
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from _split import dumps_record  # noqa: E402

# Every role cmnee uses. Order is the corpus's own frequency order, for readability.
ALL_ROLES = ("Subject", "Equipment", "Date", "Object", "Materials", "Location",
             "Militaryforce", "Content", "Result", "Quantity", "Area")

# The four adjudicated compatible with the base vocabulary by READING THE SURFACES each side
# tags, per the standing rule never to merge on string similarity. `Area` merges into
# `Location`: 67 of its surfaces already conflict with Location and they mean the same thing.
CANONICAL = {"Date": "Date", "Location": "Location", "Area": "Location", "Quantity": "Quantity"}

# Why each of the other seven cannot be emitted as a TYPE. Kept in code so the decision is
# visible where it is enforced, not only in the working paper.
INCOMPATIBLE = {
    "Subject": "base `Subject` tags subject MATTER (Visual Arts); cmnee tags the ACTOR",
    "Object": "base `Object` tags artefacts (water-jar); cmnee tags the thing acted upon",
    "Equipment": "751 same-document conflicts with Materials; different domain from the base",
    "Materials": "base tags media (Oil on canvas); cmnee tags submarines and missiles",
    "Militaryforce": "333 same-document conflicts with Equipment",
    "Content": "base tags message text; cmnee tags activities",
    "Result": "base tags numerics; cmnee tags casualty clauses",
}


def label_map(mode: str) -> dict:
    """role -> emitted label, or role absent when the mode drops it.

    `namespaced` claims no types, so every role survives; `canonical` claims types and so
    keeps only the four that can carry one; `hybrid` claims types where it can and namespaces
    the rest. Verified against the base's 2,092-label entity vocabulary: none of the eleven
    `Event<Role>` names collide with it, while 9 of 11 PLAIN role names do.
    """
    if mode == "namespaced":
        return {r: f"Event{r}" for r in ALL_ROLES}
    if mode == "canonical":
        return dict(CANONICAL)
    if mode == "hybrid":
        return {**{r: f"Event{r}" for r in ALL_ROLES}, **CANONICAL}
    raise SystemExit(f"unknown mode {mode!r}")


def derive(rec: dict, lmap: dict) -> tuple[dict, "Counter"]:
    """One event record -> one entity record, plus what happened to it.

    Returns ``({}, stats)`` when nothing survives: a record with no derived span must be
    DROPPED, never emitted with an empty entity map. An empty map asserts "no entities in
    this document", which is false here -- the roles this mode drops ARE entities -- and
    that is the within-dimension rule this corpus lives under.
    """
    stats = Counter()
    text = rec.get("input") or ""
    type_labels = set(CANONICAL.values())
    # Collect first, decide after: a surface may carry several roles, and whether that is a
    # CONFLICT depends on what the label claims. `EventEquipment` + `EventSubject` on one span
    # is ordinary multi-label ROLE annotation. `Location` + `Date` on one span is contradictory
    # TYPE annotation, because a thing has one type.
    surface_labels: dict[str, set] = {}
    for ev in rec.get("output", {}).get("events") or []:
        for arg in ev.get("arguments") or []:
            role, surface = arg.get("role"), arg.get("entity")
            if not isinstance(role, str) or not isinstance(surface, str) or not surface:
                continue
            stats[f"role:{role}"] += 1
            label = lmap.get(role)
            if label is None:
                stats["dropped_by_mode"] += 1
                continue
            if surface not in text:
                # Gate A measured 99.98% alignment; the failures are annotation artifacts
                # (a trailing digit belonging to the next token). Drop rather than emit a
                # span the tokenizer cannot locate as a subsequence.
                stats["unaligned"] += 1
                continue
            surface_labels.setdefault(surface, set()).add(label)

    by_label: dict[str, list[str]] = {}
    for surface, labels in surface_labels.items():
        claimed_types = labels & type_labels
        if len(claimed_types) > 1:
            # Fail closed on the TYPE claims only; any role labels on this surface stand.
            stats["type_conflict_dropped"] += 1
            labels = labels - claimed_types
            if not labels:
                continue
        if len(labels) > 1:
            stats["multi_label_surface"] += 1
        for label in sorted(labels):
            by_label.setdefault(label, [])
            if surface not in by_label[label]:
                by_label[label].append(surface)
                stats["kept_mention"] += 1

    if not by_label:
        stats["record_dropped"] += 1
        return {}, stats
    stats["record_kept"] += 1
    return {"input": text, "output": {"entities": by_label}}, stats


def convert(path: str, lmap: dict) -> tuple[list[dict], Counter]:
    out, total = [], Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec, stats = derive(json.loads(line), lmap)
            total.update(stats)
            if rec:
                out.append(rec)
    return out, total


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(dumps_record(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--test")
    ap.add_argument("--out-base", required=True, help="e.g. data/cmnee_roles_ner")
    ap.add_argument("--mode", choices=("namespaced", "canonical", "hybrid"),
                    default="namespaced",
                    help="namespaced (default): every role becomes Event<Role>, claims no "
                         "type, keeps the whole supply. canonical: only the four roles that "
                         "can carry a type, 24%% of the supply, transfers to the real "
                         "taxonomy. hybrid: those four canonical, the rest namespaced.")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    lmap = label_map(args.mode)
    print(f"[roles2ner] mode={args.mode}: {len(lmap)} of {len(ALL_ROLES)} roles emitted")
    base = Path(args.out_base)
    stem = base.with_suffix("") if base.suffix == ".jsonl" else base
    grand = Counter()
    for split, path in (("train", args.train), ("val", args.val), ("test", args.test)):
        if not path:
            print(f"[roles2ner] {split}: not supplied")
            continue
        recs, stats = convert(path, lmap)
        grand.update(stats)
        print(f"[roles2ner] {split}: {stats['record_kept']:,} kept, "
              f"{stats['record_dropped']:,} dropped (no derivable span), "
              f"{stats['kept_mention']:,} mentions")
        if not args.dry_run:
            _write(Path(f"{stem}.{split}.jsonl"), recs)

    kept, dropped_mode = grand["kept_mention"], grand["dropped_by_mode"]
    seen = sum(v for k, v in grand.items() if k.startswith("role:"))
    print(f"\n[roles2ner] argument mentions seen {seen:,}")
    # NOT a loss rate: `kept` counts UNIQUE (label, surface) pairs per record, so a surface
    # mentioned five times in one document contributes once. What is actually lost is
    # `dropped_by_mode` + `unaligned` + `type_conflict_dropped`; everything else is dedup.
    lost = dropped_mode + grand["unaligned"] + grand["type_conflict_dropped"]
    print(f"[roles2ner]   kept       {kept:,} unique (label, surface) pairs")
    print(f"[roles2ner]   LOST       {lost:,} ({lost / seen:.1%} of mentions)")
    print(f"[roles2ner]   dropped by mode {dropped_mode:,}")
    print(f"[roles2ner]   unaligned       {grand['unaligned']:,} (surface not verbatim)")
    print(f"[roles2ner]   type conflicts  {grand['type_conflict_dropped']:,} "
          f"(one surface, two TYPE claims)")
    print(f"[roles2ner]   multi-label surfaces kept {grand['multi_label_surface']:,} "
          f"(legitimate when the label claims a ROLE)")
    print("\n[roles2ner] per-role mentions seen:")
    for key, n in sorted(((k, v) for k, v in grand.items() if k.startswith("role:")),
                         key=lambda kv: -kv[1]):
        role = key.split(":", 1)[1]
        verdict = (f"-> {lmap[role]}" if role in lmap
                   else f"dropped ({INCOMPATIBLE.get(role, 'not in this mode')})")
        print(f"    {role:16s} {n:7,d}  {verdict}")
    print(f"\n[roles2ner] PARTIAL CORPUS. Register it under data.partial_annotation before "
          f"build_negative_pools.py sees it, or it becomes a source of entity NEGATIVES "
          f"against documents where only these roles were labelled.")


if __name__ == "__main__":
    main()
