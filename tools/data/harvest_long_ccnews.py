"""Harvest the LONG tail of CC-News -- the documents our own collector threw away.

`fetch_cc_news.py` defaults `--max-chars 12000` and SKIPS anything longer, so the pool it
built (488,710 documents, max 12,008 chars) contains nothing that reaches the model's
sliding window: at ~4.6 chars/token the window engages past ~18,800 chars and eb16's
`window_stride` is 3072 subwords. Every long-document evaluation this project might run was
excluded at collection time, by us, silently.

Measured on a 40,000-document stream of `vblagoje/cc_news` (unfiltered): median 1,709 chars,
p99 11,316, max 86,728. Documents >= 28,000 chars (~2 windows) are 0.072% of the corpus --
roughly 513 in the full 708,241 -- so this reads the parquet shards directly rather than
streaming, which took 15 minutes for 40,000 documents.

    uv run python tools/data/harvest_long_ccnews.py --min-chars 28000 --out data/ccnews_long.jsonl

Deduplicated on `_split.normalize_group_key`, the same key the splitter uses, and written
through `dumps_record` so the text is NFKC-normalized with line separators stripped -- which
matters more here than anywhere else, since one stray U+2028 in a 40,000-character article
fragments the record.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _split import dumps_record, normalize_group_key  # noqa: E402

REPO = "vblagoje/cc_news"


def harvest(min_chars: int, max_chars: int, out: Path, limit: int) -> dict:
    """Stream each parquet shard, keep documents in [min_chars, max_chars], dedup."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    stats = {"scanned": 0, "kept": 0, "dupes": 0, "too_short": 0, "too_long": 0}
    seen: set[str] = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i in range(5):
            name = f"plain_text/train-0000{i}-of-00005.parquet"
            print(f"[harvest] {name}")
            path = hf_hub_download(REPO, name, repo_type="dataset")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    stats["scanned"] += 1
                    text = (text or "").strip()
                    if len(text) < min_chars:
                        stats["too_short"] += 1
                        continue
                    if max_chars and len(text) > max_chars:
                        stats["too_long"] += 1
                        continue
                    key = normalize_group_key(text)
                    if key in seen:
                        stats["dupes"] += 1
                        continue
                    seen.add(key)
                    fh.write(dumps_record({"input": text}) + "\n")
                    stats["kept"] += 1
                    if limit and stats["kept"] >= limit:
                        print(f"[harvest] hit --limit {limit}")
                        return stats
            print(f"[harvest]   scanned={stats['scanned']:,} kept={stats['kept']:,}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-chars", type=int, default=28000,
                    help="~6,000 subwords: two sliding windows at stride 3072")
    ap.add_argument("--max-chars", type=int, default=0, help="0 = no upper bound")
    ap.add_argument("--limit", type=int, default=0, help="0 = take everything that qualifies")
    ap.add_argument("--out", type=Path, default=Path("data/ccnews_long.jsonl"))
    args = ap.parse_args()

    stats = harvest(args.min_chars, args.max_chars, args.out, args.limit)
    print(f"\n[harvest] scanned {stats['scanned']:,}")
    print(f"[harvest] KEPT    {stats['kept']:,}  -> {args.out}")
    print(f"[harvest] dropped: {stats['too_short']:,} short, "
          f"{stats['too_long']:,} long, {stats['dupes']:,} duplicate")


if __name__ == "__main__":
    main()
