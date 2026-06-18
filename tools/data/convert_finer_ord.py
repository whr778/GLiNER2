"""Convert gtfintechlab/finer-ord (financial NER) to GLiNER2 JSONL.

The source is token-per-row: each row is ``{gold_token, gold_label, doc_idx,
sent_idx}``. This regroups tokens into sentences by ``(doc_idx, sent_idx)`` (in
reading order), then folds the documented label scheme into ``output.entities``::

    {0: O, 1: PER_B, 2: PER_I, 3: LOC_B, 4: LOC_I, 5: ORG_B, 6: ORG_I}

i.e. ``*_B`` begins an entity of that type and ``*_I`` continues it. Surfaces are
the space-joined token spans (verbatim in the space-joined text).

License: CC-BY-NC-4.0 (non-commercial).

Usage::

    uv run python tools/data/convert_finer_ord.py --out data/finer_ord.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args

# label id -> (entity_type, is_begin); 0/other -> O
LABELS = {1: ("PER", True), 2: ("PER", False),
          3: ("LOC", True), 4: ("LOC", False),
          5: ("ORG", True), 6: ("ORG", False)}


def sentence_to_record(tokens: list, labels: list) -> dict | None:
    """Fold a sentence's (token, label-id) sequence into a GLiNER2 record."""
    text = " ".join(tokens)
    entities: dict[str, list[str]] = {}
    spans: list[tuple[str, int, int]] = []
    cur_type: str | None = None
    start = 0
    for i, lid in enumerate(labels):
        info = LABELS.get(lid)
        if info is None:
            if cur_type is not None:
                spans.append((cur_type, start, i))
                cur_type = None
            continue
        typ, is_begin = info
        if is_begin or typ != cur_type:
            if cur_type is not None:
                spans.append((cur_type, start, i))
            cur_type, start = typ, i
    if cur_type is not None:
        spans.append((cur_type, start, len(tokens)))

    for typ, s, e in spans:
        surface = " ".join(tokens[s:e]).strip()
        if not surface:
            continue
        bucket = entities.setdefault(typ, [])
        if surface not in bucket:
            bucket.append(surface)
    if not entities:
        return None
    return {"input": text, "output": {"entities": entities}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.{train,val,test}.jsonl).")
    parser.add_argument("--repo", default="gtfintechlab/finer-ord", help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--max-records", type=int, default=-1, help="Max sentences to emit.")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    # Group tokens into sentences by (doc, sent) in reading order.
    sentences: "OrderedDict[tuple, dict]" = OrderedDict()
    for row in ds:
        tok = row.get("gold_token")
        if not isinstance(tok, str) or not tok.strip():
            continue
        key = (row.get("doc_idx"), row.get("sent_idx"))
        s = sentences.setdefault(key, {"tokens": [], "labels": []})
        s["tokens"].append(tok)
        s["labels"].append(row.get("gold_label"))

    emitted = skipped_empty = total_entities = 0
    types: set[str] = set()
    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for s in sentences.values():
            record = sentence_to_record(s["tokens"], s["labels"])
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

    print(f"Done. sentences={len(sentences)} emitted={emitted} skipped_empty={skipped_empty} "
          f"total_entities={total_entities} types={sorted(types)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
