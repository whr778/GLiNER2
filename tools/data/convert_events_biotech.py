"""Convert knowledgator/events_classification_biotech to GLiNER2 classification JSONL.

Despite the name, this dataset is **multi-label text classification**, not
structured event extraction. There are no trigger spans or argument roles —
each article is tagged with 1-5 of 29 event-type categories (e.g.
``funding round``, ``m&a``, ``alliance & partnership``, ``executive statement``).

The source files are CSVs (``train.csv``, ``test.csv``) with columns::

    Title, Content, Target Organization, Label 1, Label 2, Label 3, Label 4, Label 5

The repo also ships a legacy ``events_classification_biotech.py`` loader
script that newer ``datasets`` rejects, so this converter downloads the CSV
directly via ``huggingface_hub`` and parses it with pandas.

Output JSONL records use GLiNER2's classification format with the full
29-label vocabulary repeated per record::

    {"input": "<title>\\n<content>",
     "output": {"classifications": [
         {"task": "biotech_event",
          "labels": ["alliance & partnership", "article publication", ...],
          "true_label": ["funding round", "m&a"]}
     ]}}

``Classification.__post_init__`` auto-sets ``multi_label=True`` when more
than one true label is provided.

Usage::

    uv run python tools/data/convert_events_biotech.py \\
        --out data/events_biotech.jsonl

    # Convert the test split instead
    uv run python tools/data/convert_events_biotech.py \\
        --out data/events_biotech_test.jsonl --file test.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LABEL_COLUMNS = ("Label 1", "Label 2", "Label 3", "Label 4", "Label 5")


def extract_labels(row, by) -> list[str]:
    """Collect non-NaN, non-empty label strings from the Label 1..5 columns."""
    labels = []
    for col in LABEL_COLUMNS:
        v = row[col]
        if v is None or by(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            labels.append(s)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL file path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to emit (-1 = all).")
    parser.add_argument("--repo", default="knowledgator/events_classification_biotech",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--file", default="train.csv",
                        help="CSV file inside the repo to convert "
                             "(default: train.csv; also available: test.csv).")
    parser.add_argument("--task-name", default="biotech_event",
                        help="Classification task name written into each record.")
    args = parser.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    print(f"Downloading {args.repo}/{args.file}...")
    src_path = Path(hf_hub_download(args.repo, args.file, repo_type="dataset"))
    df = pd.read_csv(src_path)
    print(f"Loaded {len(df):,} rows, columns={list(df.columns)}")

    # Build global vocabulary from the chosen split's labels.
    print("Building label vocabulary...")
    vocab: set[str] = set()
    for _, row in df.iterrows():
        vocab.update(extract_labels(row, pd.isna))
    kept_labels = sorted(vocab)
    print(f"Vocab: {len(kept_labels)} labels")
    label_set = set(kept_labels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    skipped_empty = 0
    multi_label_count = 0

    with args.out.open("w") as f:
        for _, row in df.iterrows():
            title = str(row.get("Title") or "").strip()
            content = str(row.get("Content") or "").strip()
            text = (title + "\n" + content).strip() if title else content
            if not text:
                skipped_empty += 1
                continue

            true_labels = [l for l in extract_labels(row, pd.isna) if l in label_set]
            if not true_labels:
                skipped_empty += 1
                continue

            record = {
                "input": text,
                "output": {
                    "classifications": [
                        {"task": args.task_name,
                         "labels": kept_labels,
                         "true_label": true_labels}
                    ]
                },
            }
            f.write(json.dumps(record) + "\n")
            emitted += 1
            if len(true_labels) > 1:
                multi_label_count += 1
            if 0 <= args.max_records <= emitted:
                break
            if emitted % 500 == 0:
                print(f"  emitted={emitted}  multi_label={multi_label_count}")

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"multi_label={multi_label_count} labels={len(kept_labels)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
