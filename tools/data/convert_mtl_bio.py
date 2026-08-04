"""Convert Crichton et al. (2017) MTL-Bioinformatics-2016 CoNLL BIO corpora to GLiNER2 NER JSONL.

The cambridgeltl/MTL-Bioinformatics-2016 repo holds biomedical NER corpora in
CoNLL BIO format (token ``\\t`` tag, blank line between sentences), each with
**canonical** ``train.tsv`` / ``devel.tsv`` / ``test.tsv`` splits (CC BY 4.0).
This converter downloads one corpus's three splits and emits GLiNER2 entities
JSONL, **preserving the canonical splits** (no random re-split): ``train.tsv`` ->
``<out>.train.jsonl``, ``devel.tsv`` -> ``<out>.val.jsonl``, ``test.tsv`` ->
``<out>.test.jsonl``. Each maximal ``B-/I-<type>`` run becomes an entity surface
grouped by ``<type>`` under ``output.entities`` (reusing the shared
``bio_to_entities`` decoder). Writes go through ``dumps_record`` (NFKC + UTF-8).

Usage::

    uv run python tools/data/convert_mtl_bio.py --dataset BC2GM --out data/bc2gm.jsonl
    uv run python tools/data/convert_mtl_bio.py --dataset NCBI-disease --out data/ncbi_disease.jsonl
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record, derive_split_paths
from convert_hf_token_ner import bio_to_entities

BASE_URL = "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data"
SRC_SPLITS = {"train": "train.tsv", "val": "devel.tsv", "test": "test.tsv"}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read().decode("utf-8")


def parse_conll(text: str):
    """Yield (tokens, tags) per sentence from CoNLL BIO text (tab- or space-delimited)."""
    tokens: list[str] = []
    tags: list[str] = []
    for line in text.splitlines():
        s = line.rstrip()
        if not s or s.startswith("-DOCSTART-"):
            if tokens:
                yield tokens, tags
                tokens, tags = [], []
            continue
        parts = s.split("\t")
        if len(parts) < 2:
            parts = s.split()
        if len(parts) < 2:
            continue
        tokens.append(parts[0])
        tags.append(parts[-1])
    if tokens:
        yield tokens, tags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True,
                    help="Corpus folder name, e.g. BC2GM, NCBI-disease, JNLPBA.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output base path (writes <base>.{train,val,test}.jsonl).")
    ap.add_argument("--encoding", default="IOB", choices=["IOB", "IOBES"],
                    help="Tag-encoding folder suffix (default IOB).")
    ap.add_argument("--input", type=Path, default=None,
                    help="Local dir holding train/devel/test.tsv (skips download).")
    ap.add_argument("--keep-empty", action="store_true",
                    help="Keep sentences with no entities (default: drop).")
    ap.add_argument("--base-url", default=BASE_URL, help="Raw base URL for the repo data folder.")
    args = ap.parse_args()

    paths = derive_split_paths(args.out)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    # Resolve the source folder: most corpora live in <dataset>-<encoding> (e.g.
    # BC2GM-IOB); a few (e.g. JNLPBA) ship BIO data in a plain <dataset> folder.
    train_text: str | None = None
    folder: str | None = None
    if args.input is None:
        for cand in (f"{args.dataset}-{args.encoding}", args.dataset):
            try:
                train_text = fetch(f"{args.base_url}/{cand}/train.tsv")
                folder = cand
                break
            except Exception:  # noqa: BLE001 - try the next candidate folder
                continue
        if folder is None:
            print(f"ERROR: no train.tsv for {args.dataset} "
                  f"(tried {args.dataset}-{args.encoding} and plain {args.dataset})")
            return 1
        print(f"Source folder: {folder}")

    counts = {s: 0 for s in paths}
    types: set[str] = set()
    total_entities = skipped_empty = 0

    for split, fname in SRC_SPLITS.items():
        if args.input is not None:
            src = args.input / fname
            if not src.exists():
                print(f"  {split}: missing {src}, skipping")
                continue
            text = src.read_text(encoding="utf-8")
        elif split == "train":
            text = train_text
        else:
            url = f"{args.base_url}/{folder}/{fname}"
            try:
                text = fetch(url)
            except Exception as e:  # noqa: BLE001 - network fetch, report and skip split
                print(f"  {split}: fetch failed ({e!r}), skipping")
                continue
        with paths[split].open("w", encoding="utf-8") as fh:
            for tokens, tags in parse_conll(text):
                out_text, entities = bio_to_entities(tokens, tags)
                if not entities and not args.keep_empty:
                    skipped_empty += 1
                    continue
                fh.write(dumps_record({"input": out_text, "output": {"entities": entities}}) + "\n")
                counts[split] += 1
                total_entities += sum(len(v) for v in entities.values())
                types.update(entities.keys())

    print(f"Done. {args.dataset}: " + " ".join(f"{s}={counts[s]}" for s in paths)
          + f" skipped_empty={skipped_empty} total_entities={total_entities} types={sorted(types)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
