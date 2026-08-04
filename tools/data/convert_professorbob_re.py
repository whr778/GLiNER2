"""Convert ProfessorBob/relation_extraction to GLiNER2 relation-extraction JSONL.

Source rows are single-triplet relation examples over a passage::

    {"passage": "Quebec ( ... French: Quebec ...) is one of the ... provinces ...",
     "triplets": ["Quebec", "language used", "French"],
     "label": "language used",
     "synonyms": ["language spoken", "official language", ...]}

The same passage recurs across many rows, each carrying a different single
triplet. This converter **groups triplets by passage** so each output record is
one passage with all its (validated) relations -- richer supervision and no
duplicated text -- using GLiNER2's relation format (head = subject, tail =
object)::

    {"input": "<passage>",
     "output": {"relations": [
         {"language used": {"head": "Quebec", "tail": "French"}},
         {"shares border with": {"head": "Quebec", "tail": "New Brunswick"}}
     ]}}

A triplet is kept only when both surfaces occur verbatim in the (whitespace-
normalized) passage -- a span the model cannot point at is not trainable.
Source `synonyms` describe each relation label but have no slot in the record
schema, so they are not emitted (they are a training-time label-augmentation
concern, not per-record data). Source has a single `train` split; SplitWriter
carves train/val/test.

The label vocabulary is large (Wikidata-style properties). Use ``--min-count``
to drop the long singleton tail; default 1 keeps everything.

Usage::

    uv run python tools/data/convert_professorbob_re.py --out data/professorbob_re.jsonl
    uv run python tools/data/convert_professorbob_re.py --out data/professorbob_re.jsonl --min-count 5
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args

WS_RE = re.compile(r"\s+")


def parse_triplet(row: dict) -> tuple[str, str, str] | None:
    """Return (subject, relation, object) or None if the row is unusable."""
    trip = row.get("triplets")
    if not isinstance(trip, (list, tuple)) or len(trip) != 3:
        return None
    subj, rel, obj = (str(x).strip() for x in trip)
    if not (subj and rel and obj):
        return None
    return subj, rel, obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train/.val/.test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum output records (grouped passages) to emit (-1 = all).")
    parser.add_argument("--repo", default="ProfessorBob/relation_extraction",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Minimum count for a relation label to be kept (default 1).")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    if args.min_count > 1:
        print("Counting relation labels...")
        counts: Counter[str] = Counter()
        for row in ds:
            parsed = parse_triplet(row)
            if parsed:
                counts[parsed[1]] += 1
        kept = {l for l, n in counts.items() if n >= args.min_count}
        print(f"Keeping {len(kept)}/{len(counts)} labels (min_count={args.min_count})")
    else:
        kept = None

    # Group validated triplets by normalized passage, de-duplicating triplets.
    groups: "OrderedDict[str, list[tuple[str, str, str]]]" = OrderedDict()
    seen: set[tuple[str, str, str, str]] = set()
    skipped_parse = skipped_span = skipped_rare = 0

    for row in ds:
        parsed = parse_triplet(row)
        if parsed is None:
            skipped_parse += 1
            continue
        subj, rel, obj = parsed
        if kept is not None and rel not in kept:
            skipped_rare += 1
            continue
        passage = row.get("passage")
        if not isinstance(passage, str) or not passage.strip():
            skipped_parse += 1
            continue
        clean = WS_RE.sub(" ", passage).strip()
        if subj not in clean or obj not in clean:
            skipped_span += 1
            continue
        key = (clean, subj, rel, obj)
        if key in seen:
            continue
        seen.add(key)
        groups.setdefault(clean, []).append((subj, rel, obj))

    emitted = 0
    labels: set[str] = set()
    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for clean, trips in groups.items():
            relations = [{rel: {"head": subj, "tail": obj}} for subj, rel, obj in trips]
            writer.write({"input": clean, "output": {"relations": relations}})
            emitted += 1
            labels.update(rel for _, rel, _ in trips)
            if 0 <= args.max_records <= emitted:
                break
            if emitted % 5000 == 0:
                print(f"  emitted={emitted}  passages_grouped={len(groups)}  labels={len(labels)}")

    print(f"Done. passages={emitted} relations_kept={len(seen)} "
          f"skipped_parse={skipped_parse} skipped_span={skipped_span} "
          f"skipped_rare={skipped_rare} labels={len(labels)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
