"""Convert knowledgator/Scientific-text-classification to GLiNER2 classification JSONL.

Source rows are flat ``{text, label}`` — a scientific abstract plus a single
domain label. The label vocabulary is *global*, so GLiNER2's per-record
``labels`` field is the same across every output record.

The raw dataset is heavily skewed: 10 broad-domain labels (mathematics,
quantum physics, astrophysics, computer science, statistics, etc.) each
have ~5,000 examples, and 28,631 fine-grained MeSH-style labels each have
exactly one example. Records with singleton labels can't train the
classification head and would inflate the per-record ``labels`` field to
~28k entries, so we filter by minimum count (``--min-count 2`` by default,
which yields exactly the 10 broad-domain labels covering ~50k rows).

Output JSONL records use GLiNER2's classification format::

    {"input": "<abstract>",
     "output": {"classifications": [
         {"task": "scientific_domain",
          "labels": ["mathematics", "quantum physics", ...],
          "true_label": "mathematics"}
     ]}}

Usage::

    uv run python tools/data/convert_scientific_text.py \\
        --out data/scientific_text.jsonl

    # Keep more of the long tail (smaller per-record `labels` quality drop)
    uv run python tools/data/convert_scientific_text.py \\
        --out data/scientific_text.jsonl --min-count 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL file path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to emit (-1 = all).")
    parser.add_argument("--repo", default="knowledgator/Scientific-text-classification",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train",
                        help="Dataset split to read.")
    parser.add_argument("--task-name", default="scientific_domain",
                        help="Classification task name written into each record.")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Minimum number of training examples a label must "
                             "have to be kept (default: 2, which drops all "
                             "singleton labels).")
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    print("Counting label frequencies...")
    counts = Counter(r["label"] for r in ds if isinstance(r.get("label"), str))
    kept_labels = sorted(label for label, n in counts.items() if n >= args.min_count)
    if not kept_labels:
        raise SystemExit(f"No labels meet --min-count={args.min_count}")
    label_set = set(kept_labels)
    print(f"Vocab: {len(kept_labels)} labels (min_count={args.min_count}) covering "
          f"{sum(counts[l] for l in kept_labels):,}/{len(ds):,} rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    skipped_rare = 0
    skipped_empty = 0

    with args.out.open("w") as f:
        for row in ds:
            text = row.get("text")
            label = row.get("label")
            if not isinstance(text, str) or not text.strip() or not isinstance(label, str):
                skipped_empty += 1
                continue
            if label not in label_set:
                skipped_rare += 1
                continue
            record = {
                "input": text,
                "output": {
                    "classifications": [
                        {"task": args.task_name,
                         "labels": kept_labels,
                         "true_label": [label]}
                    ]
                },
            }
            f.write(json.dumps(record) + "\n")
            emitted += 1
            if 0 <= args.max_records <= emitted:
                break
            if emitted % 5000 == 0:
                print(f"  emitted={emitted}  skipped_rare={skipped_rare}")

    print(f"Done. emitted={emitted} skipped_rare={skipped_rare} "
          f"skipped_empty={skipped_empty} labels={len(kept_labels)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
