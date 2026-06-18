"""Convert a HuggingFace token-classification NER dataset (tokens + BIO tags) to GLiNER2 JSONL.

Handles the three tag encodings these datasets use:

* BIO strings (``["B-LAW", "I-LAW", "O", ...]``)            -- e.g. kaznerd
* ClassLabel ints whose names live in the feature           -- e.g. bc4chemd
* bare ints needing an explicit label file (``--label-file``) -- e.g. tner/bc5cdr

The text is the space-joined tokens; each maximal ``B-/I-<type>`` run becomes an
entity surface (the space-joined token span, so it appears verbatim) grouped by
``<type>`` under ``output.entities``.

Usage::

    uv run python tools/data/convert_hf_token_ner.py \\
        --repo yeshpanovrustem/kaznerd --out data/kaznerd.jsonl

    uv run python tools/data/convert_hf_token_ner.py \\
        --repo chintagunta85/bc4chemd --revision refs/convert/parquet \\
        --out data/bc4chemd.jsonl

    uv run python tools/data/convert_hf_token_ner.py \\
        --repo tner/bc5cdr --revision refs/convert/parquet --tags-col tags \\
        --label-file dataset/label.json --out data/bc5cdr.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


def bio_to_entities(tokens: list, tags: list) -> tuple[str, dict]:
    """Join tokens into text and fold BIO ``<tag>`` runs into {type: [surface]}."""
    text = " ".join(tokens)
    entities: dict[str, list[str]] = {}
    spans: list[tuple[str, int, int]] = []
    cur_type: str | None = None
    start = 0
    for idx, tag in enumerate(tags):
        if not tag or tag == "O":
            if cur_type is not None:
                spans.append((cur_type, start, idx))
                cur_type = None
            continue
        prefix, _, typ = tag.partition("-") if "-" in tag else ("B", "", tag)
        typ = typ or prefix  # tag without a prefix is its own type
        if prefix == "B" or typ != cur_type:
            if cur_type is not None:
                spans.append((cur_type, start, idx))
            cur_type, start = typ, idx
    if cur_type is not None:
        spans.append((cur_type, start, len(tokens)))

    for typ, s, e in spans:
        surface = " ".join(tokens[s:e]).strip()
        if not surface:
            continue
        bucket = entities.setdefault(typ, [])
        if surface not in bucket:
            bucket.append(surface)
    return text, entities


def _id_to_name(features, tags_col, label_file_names):
    """Return an id->name list for int tags, or None if tags are already strings."""
    if label_file_names is not None:
        return label_file_names
    feat = features.get(tags_col)
    inner = getattr(feat, "feature", None)
    return getattr(inner, "names", None) or getattr(feat, "names", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="HuggingFace dataset repo.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.{train,val,test}.jsonl).")
    parser.add_argument("--revision", default=None,
                        help="Dataset revision (e.g. refs/convert/parquet for script datasets).")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--tokens-col", default="tokens", help="Token list column.")
    parser.add_argument("--tags-col", default="ner_tags", help="BIO tag column.")
    parser.add_argument("--label-file", default=None,
                        help="Repo file with a {label: id} map for bare-int tags "
                             "(e.g. tner's dataset/label.json).")
    parser.add_argument("--max-records", type=int, default=-1, help="Max records to emit.")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    names = None
    if args.label_file:
        from huggingface_hub import hf_hub_download
        lp = hf_hub_download(args.repo, args.label_file, repo_type="dataset")
        label2id = json.loads(Path(lp).read_text())
        names = [None] * (max(label2id.values()) + 1)
        for label, i in label2id.items():
            names[i] = label

    print(f"Loading {args.repo} revision={args.revision} split={args.split}...")
    ds = load_dataset(args.repo, revision=args.revision, split=args.split)
    id2name = _id_to_name(ds.features, args.tags_col, names)

    emitted = skipped_empty = total_entities = mismatched = 0
    types: set[str] = set()
    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for row in ds:
            tokens = row.get(args.tokens_col) or []
            tags = row.get(args.tags_col) or []
            m = min(len(tokens), len(tags))
            if m == 0:
                skipped_empty += 1
                continue
            if len(tokens) != len(tags):
                mismatched += 1
            tokens, tags = tokens[:m], tags[:m]  # align on the common prefix
            if id2name is not None:
                tags = [id2name[t] if isinstance(t, int) and 0 <= t < len(id2name) else "O" for t in tags]
            text, entities = bio_to_entities([str(t) for t in tokens], tags)
            if not entities:
                skipped_empty += 1
                continue
            writer.write({"input": text, "output": {"entities": entities}})
            emitted += 1
            total_entities += sum(len(v) for v in entities.values())
            types.update(entities.keys())
            if 0 <= args.max_records <= emitted:
                break

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"len_mismatch_aligned={mismatched} total_entities={total_entities} "
          f"types={sorted(types)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
