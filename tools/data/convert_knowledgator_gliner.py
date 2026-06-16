"""Convert knowledgator/GLINER-multi-task-synthetic-data to GLiNER2 JSONL.

The source dataset stores each row as::

    tokenized_text: ["Identify", "the", "following", ..., "Text", ":", "\\n",
                     "Gurgurnica", "(", ",", ")", "is", ...]
    ner:            [["19", "19", "\\"Village\\""],
                     ["30", "30", "\\"Municipality\\""], ...]

Each row carries an "Identify ... Text:" prompt prefix followed by the actual
body, plus a list of NER spans with inclusive token indices and JSON-quoted
labels. To avoid memorising the prompt template, this converter:

1. Locates the body — the tokens after the first ``("Text", ":", "\\n")`` trio.
2. Re-bases each NER span's token indices into the body.
3. Joins body tokens into a plain string and uses it as ``input``.
4. Strips the JSON quoting from labels.

Usage::

    uv run python tools/data/convert_knowledgator_gliner.py \\
        --out data/knowledgator_gliner.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


def find_body_start(tokens: list[str]) -> int | None:
    """Return the index right after the first ``"Text", ":", "\\n"`` trio."""
    for i in range(len(tokens) - 2):
        if tokens[i] == "Text" and tokens[i + 1] == ":" and tokens[i + 2] == "\n":
            return i + 3
    return None


def unwrap_label(raw: str) -> str | None:
    """Strip the JSON quoting around a label, e.g. '\"Village\"' -> 'Village'."""
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
    """Convert one knowledgator row to a GLiNER2 record; None if unusable."""
    tokens = row.get("tokenized_text") or []
    ner = row.get("ner") or []
    if not tokens or not ner:
        return None

    body_start = find_body_start(tokens)
    if body_start is None:
        return None
    body_tokens = tokens[body_start:]
    if not body_tokens:
        return None
    body_text = " ".join(body_tokens)

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

        rel_start = start - body_start
        rel_end = end - body_start
        if rel_start < 0 or rel_end >= len(body_tokens) or rel_end < rel_start:
            continue

        surface = " ".join(body_tokens[rel_start:rel_end + 1])
        if surface not in body_text:
            continue
        bucket = entities.setdefault(label, [])
        if surface not in bucket:
            bucket.append(surface)

    if not entities:
        return None
    return {"input": body_text, "output": {"entities": entities}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train.jsonl, "
                             ".val.jsonl, .test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to read (-1 = all).")
    parser.add_argument("--repo", default="knowledgator/GLINER-multi-task-synthetic-data",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train",
                        help="Dataset split to read.")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Streaming {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    emitted = 0
    skipped_no_body = 0
    skipped_empty = 0
    total_entities = 0
    all_types: set[str] = set()

    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for idx, row in enumerate(ds):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row)
            if record is None:
                if not find_body_start(row.get("tokenized_text") or []):
                    skipped_no_body += 1
                else:
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
          f"skipped_no_body={skipped_no_body} total_entities={total_entities} "
          f"distinct_types={len(all_types)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
