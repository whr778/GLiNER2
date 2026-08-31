"""A general-purpose Simplified Chinese news sample for multi-task annotation.

Distinct from `build_chinese_candidates.py`, which selects the CASUALTY cue region. This
takes a broad sample for NER / relations / structures / classifications / events, so it
deliberately does NOT cue-filter: a corpus selected on death words would teach the model
that entities and relations only occur in disaster reporting.

Excludes everything already annotated for casualty, so the two corpora stay disjoint and
the multi-task model is not evaluated on text it saw under a different task.

Sampled with a STRIDE across the corpus rather than the head: news_zh is ordered by source
in blocks, so a contiguous read is a single-publisher sample.

    uv run python tools/data/build_zh_general_pool.py --limit 6000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record, normalize_group_key  # noqa: E402
from build_chinese_candidates import URL, is_simplified  # noqa: E402

MIN_CHARS, MAX_CHARS = 400, 4000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--out", default="data/chinese_gate/zh_general_6k.jsonl")
    ap.add_argument("--stride", type=int, default=97, help="prime, to spread the sample")
    ap.add_argument("--exclude", nargs="+", default=[
        "data/chinese_gate/zh_cas_candidates.jsonl",
        "data/chinese_gate/zh_cas_candidates_2.jsonl",
        "data/chinese_gate/zh_gate_sample.jsonl",
    ])
    args = ap.parse_args()

    import fsspec
    import pyarrow.parquet as pq

    done: set = set()
    for p in args.exclude:
        f = Path(p)
        if f.is_file():
            for line in f.open(encoding="utf-8"):
                done.add(normalize_group_key(json.loads(line)["input"])[:300])
    print(f"[zh-gen] {len(done):,} documents excluded (already annotated for casualty)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = scanned = 0
    sources: Counter = Counter()
    with fsspec.open(URL).open() as fh, out_path.open("w", encoding="utf-8") as out:
        for batch in pq.ParquetFile(fh).iter_batches(batch_size=5000,
                                                     columns=["content", "source", "title"]):
            d = batch.to_pydict()
            for content, source, title in zip(d["content"], d["source"], d["title"]):
                scanned += 1
                if scanned % args.stride:
                    continue
                text = str(content or "")[:MAX_CHARS]
                if len(text) < MIN_CHARS or not is_simplified(text):
                    continue
                key = normalize_group_key(text)[:300]
                if key in done:
                    continue
                done.add(key)
                out.write(dumps_record({"input": text, "source": str(source or "?"),
                                        "title": str(title or "")}) + "\n")
                kept += 1
                sources[str(source or "?")] += 1
                if kept >= args.limit:
                    print(f"[zh-gen] {kept:,} documents from {scanned:,} scanned -> {out_path}")
                    print(f"[zh-gen] {len(sources):,} distinct sources; "
                          f"top: {[k for k, _ in sources.most_common(5)]}")
                    return 0
    print(f"[zh-gen] exhausted at {kept:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
