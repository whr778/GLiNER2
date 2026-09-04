"""The Turkish candidate pool for field-level casualty annotation, across many outlets.

`build_turkish_candidates.py` buys TRT Haber alone (one outlet, one year) and says so:
a model trained on it may have learned one publisher's register rather than Turkish. This
builds the larger, multi-outlet pool that the extractor purchase draws from.

SHARD CHOICE IS OUTLET CHOICE. umutertugrul/turkish-news-1.8M-tokenized is ordered by
source in blocks -- 12 of its 31 shards hold a single outlet -- so a contiguous read is a
single-publisher corpus. The default shards below were picked by reading the `url` column
of every shard, and between them cover all ten outlets: cnnturk, aa, aljazeera, haberler,
dw, indyturk, sozcu, t24, euronews, yesilgazete.

Only `text` and `url` are read. The parquet is 5.16 GB because half of it is a `tokens`
column this never touches; columnar reads make the pull ~89 MB per shard instead of 166.

TWO EXCLUSIONS, both required before the data can be trusted:
  - the held-out gate evaluation documents, by normalised group key
  - whole HOLDOUT outlets, kept out of training so the extractor blind test measures
    generalisation across publishers rather than memorised register

    uv run python tools/data/build_turkish_pool.py --out /Volumes/Development/data/turkish_pool.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import fsspec
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record, normalize_group_key  # noqa: E402
from build_turkish_candidates import CUE, MAX_CHARS  # noqa: E402

BASE = ("https://huggingface.co/datasets/umutertugrul/turkish-news-1.8M-tokenized/"
        "resolve/main/data/train-{:05d}-of-00031.parquet")
# Between them these cover all ten outlets; see the shard/outlet map in the docstring.
SHARDS = [0, 2, 4, 7, 8, 20, 23, 25]
# One international Turkish service and one domestic independent, both with enough volume
# for a meaningful blind test. Held out WHOLE, so the extractor test set is a publisher the
# model never read.
HOLDOUT = ("dw.com", "t24.com.tr")
MIN_CHARS = 400


def outlet(url: str) -> str:
    return urlparse(url or "").netloc.replace("www.", "").split(":")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/Volumes/Development/data/turkish_pool.jsonl")
    ap.add_argument("--holdout-out", default="/Volumes/Development/data/turkish_pool_holdout.jsonl")
    ap.add_argument("--shards", type=int, nargs="+", default=SHARDS)
    ap.add_argument("--exclude", nargs="+",
                    default=["data/turkish_gate/gate_ann_tr_heldout.jsonl",
                             "data/turkish_gate/gate_ann_turkish.jsonl"],
                    help="documents already bought or reserved; not re-collected")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cue", choices=("require", "exclude", "any"), default="require",
                    help="require: cue-bearing only, the casualty-annotation default and "
                         "what built turkish_pool18 (measured 100%% cue-bearing). "
                         "exclude: the COMPLEMENT, for EVENT-TYPE annotation -- a pool "
                         "filtered to casualty cues carries no Turkish `sport competition` "
                         "or `Organization Fine`, so a stage-1 model bought from it would "
                         "learn the ~20 disaster types and never learn to say `not a "
                         "disaster`. any: no filter.")
    ap.add_argument("--exclude-keys", default="",
                    help="file of pre-hashed group keys, one per line. Lets a remote box "
                         "honour the exclusions without shipping it the documents: the "
                         "keys are ~200 KB where the corpora are ~33 MB, and an unexcluded "
                         "rebuild would re-collect held-out eval documents into training")
    args = ap.parse_args()

    done: set[str] = set()
    for path in args.exclude:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            done.add(normalize_group_key(json.loads(line)["input"])[:300])
    if args.exclude_keys:
        for line in Path(args.exclude_keys).open(encoding="utf-8"):
            if line.strip():
                done.add(line.strip())
    print(f"[pool] {len(done)} documents excluded (already bought or held out)")

    train_path, hold_path = Path(args.out), Path(args.holdout_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    outlets, held, kept, scanned, dropped = Counter(), Counter(), 0, 0, 0

    with train_path.open("w", encoding="utf-8") as tf, hold_path.open("w", encoding="utf-8") as hf:
        for shard in args.shards:
            try:
                with fsspec.open(BASE.format(shard)).open() as fh:
                    table = pq.ParquetFile(fh).read(columns=["text", "url"])
            except Exception as exc:
                print(f"[pool] shard {shard:2d} FAILED: {exc}", flush=True)
                continue
            texts, urls = table["text"].to_pylist(), table["url"].to_pylist()
            for text, url in zip(texts, urls):
                scanned += 1
                text = (text or "")[:MAX_CHARS]
                if len(text) < MIN_CHARS:
                    continue
                has_cue = bool(CUE.search(text))
                if (args.cue == "require" and not has_cue) or \
                   (args.cue == "exclude" and has_cue):
                    continue
                key = normalize_group_key(text)[:300]
                if key in done:
                    dropped += 1
                    continue
                done.add(key)
                site = outlet(url)
                row = {"input": text, "url": url, "outlet": site, "shard": shard}
                if site in HOLDOUT:
                    hf.write(dumps_record(row) + "\n")
                    held[site] += 1
                else:
                    tf.write(dumps_record(row) + "\n")
                    outlets[site] += 1
                    kept += 1
            print(f"[pool] shard {shard:2d} done  scanned={scanned:7d}  kept={kept:6d}", flush=True)
            if args.limit and kept >= args.limit:
                break

    print(f"[pool] {train_path}: {kept} cue-bearing candidates from {scanned} scanned "
          f"({kept / scanned:.1%}), {dropped} already-owned duplicates skipped")
    for site, n in outlets.most_common():
        print(f"[pool]   {site:22s} {n:6d}")
    print(f"[pool] {hold_path}: {sum(held.values())} held-out-outlet documents {dict(held)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
