"""Convert knowledgator/sentence_rex to GLiNER2 relation-extraction JSONL.

Source rows are sentence-level relation classification with the two arguments
marked inline by ``<e1>...</e1>`` and ``<e2>...</e2>`` tags::

    {"sentences": "<e1>Pope Pius XII</e1> re-opened the cause on 7 December 1954, "
                  "and Pope John Paul II proclaimed him <e2>Venerable</e2> on 6 July 1985.",
     "labels": "canonization status"}

This converter strips the tags to recover the clean text, extracts the two
argument surfaces, and emits a single relation per record using GLiNER2's
relation format (head = e1, tail = e2)::

    {"input": "<clean text>",
     "output": {"relations": [
         {"<label>": {"head": "<e1 surface>", "tail": "<e2 surface>"}}
     ]}}

The label vocabulary is large (~850 relation types, mostly Wikidata
properties). Use ``--min-count`` to drop the long singleton tail if you
want a cleaner training signal; default 1 keeps everything.

Usage::

    uv run python tools/data/convert_sentence_rex.py \\
        --out data/sentence_rex.jsonl

    # Drop relation labels with fewer than 5 examples
    uv run python tools/data/convert_sentence_rex.py \\
        --out data/sentence_rex.jsonl --min-count 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


E1_RE = re.compile(r"<e1>\s*(.*?)\s*</e1>", re.DOTALL)
E2_RE = re.compile(r"<e2>\s*(.*?)\s*</e2>", re.DOTALL)
TAG_RE = re.compile(r"</?e[12]>")


def parse_row(row: dict) -> tuple[str, str, str, str] | None:
    """Return (clean_text, e1_surface, e2_surface, label) or None if unparseable."""
    sentence = row.get("sentences")
    label = row.get("labels")
    if not isinstance(sentence, str) or not isinstance(label, str):
        return None
    label = label.strip()
    if not label:
        return None

    m1 = E1_RE.search(sentence)
    m2 = E2_RE.search(sentence)
    if not m1 or not m2:
        return None
    e1 = m1.group(1).strip()
    e2 = m2.group(1).strip()
    if not e1 or not e2:
        return None

    clean = TAG_RE.sub("", sentence)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        return None
    if e1 not in clean or e2 not in clean:
        return None
    return clean, e1, e2, label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train.jsonl, "
                             ".val.jsonl, .test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to emit (-1 = all).")
    parser.add_argument("--repo", default="knowledgator/sentence_rex",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train",
                        help="Dataset split to read.")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Minimum count for a relation label to be kept "
                             "(default 1 = keep everything).")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    label_counts: Counter[str] = Counter()
    if args.min_count > 1:
        print("Counting relation labels...")
        for row in ds:
            lbl = row.get("labels")
            if isinstance(lbl, str) and lbl.strip():
                label_counts[lbl.strip()] += 1
        kept = {l for l, n in label_counts.items() if n >= args.min_count}
        print(f"Keeping {len(kept)}/{len(label_counts)} labels (min_count={args.min_count})")
    else:
        kept = None

    emitted = 0
    skipped_parse = 0
    skipped_rare = 0
    all_labels: set[str] = set()

    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for row in ds:
            parsed = parse_row(row)
            if parsed is None:
                skipped_parse += 1
                continue
            clean, e1, e2, label = parsed
            if kept is not None and label not in kept:
                skipped_rare += 1
                continue
            record = {
                "input": clean,
                "output": {"relations": [{label: {"head": e1, "tail": e2}}]},
            }
            writer.write(record)
            emitted += 1
            all_labels.add(label)
            if 0 <= args.max_records <= emitted:
                break
            if emitted % 5000 == 0:
                print(f"  emitted={emitted}  skipped_parse={skipped_parse}  "
                      f"labels={len(all_labels)}")

    print(f"Done. emitted={emitted} skipped_parse={skipped_parse} "
          f"skipped_rare={skipped_rare} labels={len(all_labels)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
