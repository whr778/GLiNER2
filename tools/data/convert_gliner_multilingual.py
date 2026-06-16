"""Convert knowledgator/gliner-multilingual-synthetic to GLiNER2 JSONL.

The source dataset stores each row as a tokenized multilingual passage plus a
list of NER spans with inclusive token indices and JSON-quoted labels::

    tokenized_text: ["Der", "Film", "wurde", "in", "Los", "Angeles", "und",
                     "Santa", "Clarita", "gedreht", "."]
    ner:            [["7", "8", "\\"location\\""],
                     ["4", "5", "\\"location\\""]]

Unlike ``knowledgator/GLINER-multi-task-synthetic-data``, there is no prompt
prefix — the tokens are the body text itself. This converter:

1. Joins the tokens with spaces to form the ``input``.
2. For each NER span, extracts the surface by token-range slice.
3. Unwraps the JSON quoting on labels.
4. Drops surfaces that don't appear verbatim in the joined text.

Usage::

    uv run python tools/data/convert_gliner_multilingual.py \\
        --out data/gliner_multilingual.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


def unwrap_label(raw: str) -> str | None:
    """Strip the JSON quoting around a label, e.g. '\"location\"' -> 'location'."""
    if not isinstance(raw, str):
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        decoded = raw.strip('"').strip()
    if not isinstance(decoded, str) or not decoded.strip():
        return None
    return decoded.strip()


def convert_row(row: dict) -> dict | None:
    """Convert one multilingual row to a GLiNER2 record; None if unusable."""
    tokens = row.get("tokenized_text") or []
    ner = row.get("ner") or []
    if not tokens or not ner:
        return None

    text = " ".join(tokens)
    entities: dict[str, list[str]] = {}

    for span in ner:
        if not isinstance(span, (list, tuple)) or len(span) != 3:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
        except (TypeError, ValueError):
            continue
        label = unwrap_label(span[2])
        if label is None:
            continue
        if start < 0 or end >= len(tokens) or end < start:
            continue

        surface = " ".join(tokens[start:end + 1])
        if surface not in text:
            continue
        bucket = entities.setdefault(label, [])
        if surface not in bucket:
            bucket.append(surface)

    if not entities:
        return None
    return {"input": text, "output": {"entities": entities}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train.jsonl, "
                             ".val.jsonl, .test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to read (-1 = all).")
    parser.add_argument("--repo", default="knowledgator/gliner-multilingual-synthetic",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train",
                        help="Dataset split to read.")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Streaming {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    emitted = 0
    skipped_empty = 0
    total_entities = 0
    all_types: set[str] = set()

    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for idx, row in enumerate(ds):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row)
            if record is None:
                skipped_empty += 1
                continue
            writer.write(record)
            emitted += 1
            total_entities += sum(len(v) for v in record["output"]["entities"].values())
            all_types.update(record["output"]["entities"].keys())

            if emitted % 2000 == 0:
                print(f"  emitted={emitted}  skipped_empty={skipped_empty}  "
                      f"types={len(all_types)}")

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_entities={total_entities} distinct_types={len(all_types)} "
          f"{writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
