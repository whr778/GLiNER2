"""Convert numind/NuNER to GLiNER2 JSONL.

The NuNER `full` split stores each row as:
    input:  "<text>"
    output: "['<surface> <> <type> <> <description>', ...]"  (Python list literal)

The `entity` split omits the `<description>` field. This script streams the
dataset from HuggingFace and emits one GLiNER2 JSONL record per input row,
keeping only spans that appear verbatim in the source text.

Usage::

    uv run python tools/data/convert_nuner.py --split full --out nuner.jsonl
    uv run python tools/data/convert_nuner.py --split full --out nuner.jsonl --max-records 10000
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


def parse_items(output_field: str) -> list[list[str]]:
    """Parse the NuNER `output` string into a list of [surface, type, desc?] lists."""
    raw = ast.literal_eval(output_field)
    parsed = []
    for item in raw:
        parts = item.split(" <> ", 2)
        if len(parts) >= 2:
            parsed.append(parts)
    return parsed


def convert_row(row: dict) -> dict | None:
    """Convert one NuNER row to a GLiNER2 record; return None if it has no entities."""
    text = row.get("input")
    output_field = row.get("output")
    # A handful of NuNER rows have input=None (e.g. row 135648 in the full split).
    if not isinstance(text, str) or not text or not isinstance(output_field, str):
        return None
    entities: dict[str, list[str]] = defaultdict(list)
    descriptions: dict[str, str] = {}

    try:
        items = parse_items(output_field)
    except (SyntaxError, ValueError):
        return None

    for parts in items:
        surface, etype = parts[0], parts[1]
        if surface not in text:
            continue
        if surface not in entities[etype]:
            entities[etype].append(surface)
        if len(parts) == 3 and etype not in descriptions:
            descriptions[etype] = parts[2]

    if not entities:
        return None

    output = {"entities": dict(entities)}
    if descriptions:
        output["entity_descriptions"] = descriptions
    return {"input": text, "output": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="full", choices=["full", "entity"],
                        help="NuNER split to convert (default: full)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train.jsonl, "
                             ".val.jsonl, .test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to read (-1 = all)")
    parser.add_argument("--repo", default="numind/NuNER",
                        help="HuggingFace dataset repo (default: numind/NuNER)")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Streaming {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    emitted = 0
    skipped_empty = 0
    skipped_parse = 0
    all_types: set[str] = set()

    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for idx, row in enumerate(ds):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row)
            if record is None:
                if "output" not in row:
                    skipped_parse += 1
                else:
                    skipped_empty += 1
                continue
            writer.write(record)
            emitted += 1
            all_types.update(record["output"]["entities"].keys())

            if emitted % 5000 == 0:
                print(f"  emitted={emitted}  skipped_empty={skipped_empty}  types={len(all_types)}")

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} skipped_parse={skipped_parse} "
          f"distinct_types={len(all_types)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
