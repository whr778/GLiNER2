"""Convert knowledgator/gliclass-v3-logic-dataset to GLiNER2 classification JSONL.

Unlike the NER converters, this dataset trains GLiNER2's *classification*
head. The source rows are::

    {"text": "Where are there lots of seats placed in rows surrounding a court?",
     "true_labels": ["auditorium"],
     "all_labels": ["show", "auditorium", "movies", "soccer stadium", "hockey game"]}

Output JSONL records use GLiNER2's classification format::

    {"input": <text>,
     "output": {"classifications": [
         {"task": "logic",
          "labels": <all_labels>,
          "true_label": <true_labels>}
     ]}}

``Classification.__post_init__`` auto-sets ``multi_label=True`` when more
than one true label is provided.

About ~27% of source rows have ``true_labels`` that are NOT a subset of
``all_labels`` (NLI-style entries where ``true_labels`` is a relation type
like "neutral" and ``all_labels`` is a single candidate hypothesis). These
would fail ``Classification.validate()``, so we drop them.

Usage::

    uv run python tools/data/convert_gliclass_logic.py \\
        --out data/gliclass_logic.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


def convert_row(row: dict, task_name: str) -> dict | None:
    """Convert one gliclass row to a GLiNER2 classification record; None if unusable."""
    text = row.get("text")
    true_labels = row.get("true_labels") or []
    all_labels = row.get("all_labels") or []

    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(all_labels, list) or not all_labels:
        return None
    if not isinstance(true_labels, list) or not true_labels:
        return None

    labels = [str(x) for x in all_labels if isinstance(x, (str, int, float)) and str(x).strip()]
    trues = [str(x) for x in true_labels if isinstance(x, (str, int, float)) and str(x).strip()]
    if not labels or not trues:
        return None
    if not set(trues).issubset(set(labels)):
        return None

    return {
        "input": text,
        "output": {
            "classifications": [
                {"task": task_name, "labels": labels, "true_label": trues}
            ]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train.jsonl, "
                             ".val.jsonl, .test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to read (-1 = all).")
    parser.add_argument("--repo", default="knowledgator/gliclass-v3-logic-dataset",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train",
                        help="Dataset split to read.")
    parser.add_argument("--task-name", default="logic",
                        help="Classification task name written into each record.")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Streaming {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    emitted = 0
    skipped_not_subset = 0
    skipped_empty = 0
    multi_label_count = 0
    label_count_sum = 0

    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for idx, row in enumerate(ds):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row, args.task_name)
            if record is None:
                trues = row.get("true_labels") or []
                alls = row.get("all_labels") or []
                if trues and alls and not set(map(str, trues)).issubset(set(map(str, alls))):
                    skipped_not_subset += 1
                else:
                    skipped_empty += 1
                continue
            writer.write(record)
            emitted += 1
            cls = record["output"]["classifications"][0]
            label_count_sum += len(cls["labels"])
            if len(cls["true_label"]) > 1:
                multi_label_count += 1

            if emitted % 2000 == 0:
                print(f"  emitted={emitted}  skipped_not_subset={skipped_not_subset}  "
                      f"multi_label={multi_label_count}")

    avg_labels = label_count_sum / emitted if emitted else 0.0
    print(f"Done. emitted={emitted} skipped_not_subset={skipped_not_subset} "
          f"skipped_empty={skipped_empty} multi_label={multi_label_count} "
          f"avg_labels_per_row={avg_labels:.1f} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
