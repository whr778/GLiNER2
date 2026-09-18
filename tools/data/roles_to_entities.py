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

ROLE IS NOT TYPE, and only four roles survive that. 19.5% of cmnee documents tag one surface
with more than one role (a submarine is `Equipment`, `Materials` AND `Subject` in the same
document) -- a thing's TYPE does not change within a document, its ROLE does, so deriving type
from role manufactures contradictory supervision. Seven of eleven roles are rejected: either
incompatible with a base label of the same spelling (`Subject` tags subject MATTER in the base
-- "Visual Arts" -- and the ACTOR in cmnee), or heavily self-conflicting. See
OPTION_4_ROLE_TO_ENTITY.md section 5 for the surface-by-surface adjudication.

THE COST OF THAT HONESTY: 15,188 of 62,573 mentions, 24.3%.

Labels are emitted CANONICAL and directly -- `Area` is written as `Location`, not written as
`Area` and remapped. The `labels_file` indirection exists to protect SOURCE corpora from being
rewritten; a derived corpus is new data we author, so nothing is bent by naming it correctly
at birth.

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

# The four roles adjudicated DERIVE, with the canonical label each becomes. `Area` merges
# into `Location`: 67 of its surfaces already conflict with Location, and they mean the same
# thing (阿富汗, 土耳其中部城市瑟瓦斯).
ROLE_MAP = {
    "Date": "Date",
    "Location": "Location",
    "Area": "Location",
    "Quantity": "Quantity",
}

# Kept explicit rather than implied by absence, so a reader sees the decision and its reason.
REJECTED = {
    "Subject": "base `Subject` tags subject MATTER (Visual Arts); cmnee tags the ACTOR",
    "Object": "base `Object` tags artefacts (water-jar); cmnee tags the thing acted upon",
    "Equipment": "751 same-document conflicts with Materials; different domain from the base",
    "Materials": "base tags media (Oil on canvas); cmnee tags submarines and missiles",
    "Militaryforce": "333 same-document conflicts with Equipment",
    "Content": "base tags message text; cmnee tags activities",
    "Result": "base tags numerics; cmnee tags casualty clauses",
}


def derive(rec: dict) -> tuple[dict, Counter]:
    """One event record -> one entity record, plus what happened to it.

    Returns ``({}, stats)`` when nothing survives: a record with no derived span must be
    DROPPED, never emitted with an empty entity map. An empty map asserts "no entities in
    this document", which is false here -- the rejected roles are entities, just not ones
    whose type we can trust -- and that is the within-dimension rule this corpus exists
    under.
    """
    stats = Counter()
    text = rec.get("input") or ""
    by_label: dict[str, list[str]] = {}
    # A surface may appear under several roles in one document. Collect first, then decide,
    # so a conflict is detected rather than resolved by whichever role was read last.
    surface_labels: dict[str, set] = {}
    for ev in rec.get("output", {}).get("events") or []:
        for arg in ev.get("arguments") or []:
            role, surface = arg.get("role"), arg.get("entity")
            if not isinstance(role, str) or not isinstance(surface, str) or not surface:
                continue
            stats[f"role:{role}"] += 1
            label = ROLE_MAP.get(role)
            if label is None:
                stats["rejected_mention"] += 1
                continue
            if surface not in text:
                # Gate A measured 99.98% alignment; the failures are annotation artifacts
                # (a trailing digit belonging to the next token). Drop rather than emit a
                # span the tokenizer cannot locate.
                stats["unaligned"] += 1
                continue
            surface_labels.setdefault(surface, set()).add(label)

    for surface, labels in surface_labels.items():
        if len(labels) > 1:
            # Fail closed. Four such surfaces survive the Area->Location merge across the
            # whole corpus; keeping either label would be inventing supervision.
            stats["conflict_dropped"] += 1
            continue
        label = next(iter(labels))
        by_label.setdefault(label, [])
        if surface not in by_label[label]:
            by_label[label].append(surface)
            stats["kept_mention"] += 1

    if not by_label:
        stats["record_dropped"] += 1
        return {}, stats
    stats["record_kept"] += 1
    return {"input": text, "output": {"entities": by_label}}, stats


def convert(path: str) -> tuple[list[dict], Counter]:
    out, total = [], Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec, stats = derive(json.loads(line))
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
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    base = Path(args.out_base)
    stem = base.with_suffix("") if base.suffix == ".jsonl" else base
    grand = Counter()
    for split, path in (("train", args.train), ("val", args.val), ("test", args.test)):
        if not path:
            print(f"[roles2ner] {split}: not supplied")
            continue
        recs, stats = convert(path)
        grand.update(stats)
        print(f"[roles2ner] {split}: {stats['record_kept']:,} kept, "
              f"{stats['record_dropped']:,} dropped (no derivable span), "
              f"{stats['kept_mention']:,} mentions")
        if not args.dry_run:
            _write(Path(f"{stem}.{split}.jsonl"), recs)

    kept, rejected = grand["kept_mention"], grand["rejected_mention"]
    seen = kept + rejected + grand["unaligned"] + grand["conflict_dropped"]
    print(f"\n[roles2ner] mentions seen {seen:,}")
    print(f"[roles2ner]   kept       {kept:,} ({kept / seen:.1%})")
    print(f"[roles2ner]   rejected   {rejected:,} (role not in the DERIVE set)")
    print(f"[roles2ner]   unaligned  {grand['unaligned']:,} (surface not verbatim in text)")
    print(f"[roles2ner]   conflicts  {grand['conflict_dropped']:,} (one surface, two labels)")
    print("\n[roles2ner] per-role mentions seen:")
    for key, n in sorted(((k, v) for k, v in grand.items() if k.startswith("role:")),
                         key=lambda kv: -kv[1]):
        role = key.split(":", 1)[1]
        verdict = f"-> {ROLE_MAP[role]}" if role in ROLE_MAP else f"REJECTED ({REJECTED.get(role, 'unlisted')})"
        print(f"    {role:16s} {n:7,d}  {verdict}")
    print(f"\n[roles2ner] PARTIAL CORPUS. Register it under data.partial_annotation before "
          f"build_negative_pools.py sees it, or it becomes a source of entity NEGATIVES "
          f"against documents where only these roles were labelled.")


if __name__ == "__main__":
    main()
