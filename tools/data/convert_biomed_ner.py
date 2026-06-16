"""Convert knowledgator/biomed_NER to GLiNER2 JSONL.

Source rows are flat ``{text, entities}`` with character offsets::

    {"text": "Weed seed inactivation in soil mesocosms ...",
     "entities": [{"start": 0, "end": 4, "class": "ORGANISM"},
                  {"start": 26, "end": 30, "class": "CHEMICALS"},
                  ...]}

Offsets are end-exclusive (``text[start:end]``), so surfaces are sliced
directly from ``text`` and the verbatim filter that NuNER/Pile-NER need
isn't required here.

Light cleanup:

- Strip whitespace from class names (the source has a few trailing-space
  duplicates like ``"ORGANISMS "`` vs ``"ORGANISMS"``).
- Skip the ``"Unlabelled"`` class — no training signal.

Usage::

    uv run python tools/data/convert_biomed_ner.py \\
        --out data/biomed_ner.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SKIP_CLASSES = {"Unlabelled"}


def convert_row(row: dict) -> dict | None:
    """Convert one biomed_NER row to a GLiNER2 record; None if no usable entities."""
    text = row.get("text")
    spans = row.get("entities") or []
    if not isinstance(text, str) or not text.strip() or not spans:
        return None

    entities: dict[str, list[str]] = {}
    for span in spans:
        if not isinstance(span, dict):
            continue
        cls = span.get("class")
        if not isinstance(cls, str):
            continue
        cls = cls.strip()
        if not cls or cls in SKIP_CLASSES:
            continue
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end > len(text) or end <= start:
            continue
        surface = text[start:end].strip()
        if not surface:
            continue
        bucket = entities.setdefault(cls, [])
        if surface not in bucket:
            bucket.append(surface)

    if not entities:
        return None
    return {"input": text, "output": {"entities": entities}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL file path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to read (-1 = all).")
    parser.add_argument("--repo", default="knowledgator/biomed_NER",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train",
                        help="Dataset split to read.")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Streaming {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    emitted = 0
    skipped_empty = 0
    total_entities = 0
    all_types: set[str] = set()

    with args.out.open("w") as f:
        for idx, row in enumerate(ds):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row)
            if record is None:
                skipped_empty += 1
                continue
            f.write(json.dumps(record) + "\n")
            emitted += 1
            total_entities += sum(len(v) for v in record["output"]["entities"].values())
            all_types.update(record["output"]["entities"].keys())

            if emitted % 1000 == 0:
                print(f"  emitted={emitted}  skipped_empty={skipped_empty}  "
                      f"types={len(all_types)}")

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_entities={total_entities} distinct_types={len(all_types)} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
