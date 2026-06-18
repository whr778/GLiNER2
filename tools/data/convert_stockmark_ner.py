"""Convert stockmark/ner-wikipedia-dataset (Japanese NER) to GLiNER2 JSONL.

Each row is ``{text, entities: [{name, span, type}], curid}``. Entities are
already span-based, so this just groups each mention surface (``name``) by its
``type`` into ``output.entities``, keeping only surfaces present verbatim in the
text. Types are the dataset's Japanese labels (e.g. 人名, 法人名) kept as-is.

License: CC-BY-SA-3.0.

Usage::

    uv run python tools/data/convert_stockmark_ner.py --out data/stockmark_jpn.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


def convert_row(row: dict) -> dict | None:
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    entities: dict[str, list[str]] = {}
    for ent in row.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        name = ent.get("name")
        etype = ent.get("type")
        if not isinstance(name, str) or not isinstance(etype, str):
            continue
        name, etype = name.strip(), etype.strip()
        if not name or not etype or name not in text:
            continue
        bucket = entities.setdefault(etype, [])
        if name not in bucket:
            bucket.append(name)
    if not entities:
        return None
    return {"input": text, "output": {"entities": entities}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.{train,val,test}.jsonl).")
    parser.add_argument("--repo", default="stockmark/ner-wikipedia-dataset",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--max-records", type=int, default=-1, help="Max records to emit.")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    emitted = skipped_empty = total_entities = 0
    types: set[str] = set()
    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for row in ds:
            record = convert_row(row)
            if record is None:
                skipped_empty += 1
                continue
            writer.write(record)
            emitted += 1
            ents = record["output"]["entities"]
            total_entities += sum(len(v) for v in ents.values())
            types.update(ents.keys())
            if 0 <= args.max_records <= emitted:
                break

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_entities={total_entities} types={sorted(types)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
