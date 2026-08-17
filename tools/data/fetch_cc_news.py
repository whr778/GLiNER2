"""Collect real English news documents from `vblagoje/cc_news` for annotation.

Emits a GLiNER2 JSONL with `input` set to the article body and `output` empty --
the shape `synthetic/generate.py --annotate-from` consumes. This is the real-text
half of the paper's real/synthetic mixture: real documents, model-written labels.

**Licensing.** The dataset card declares `language: en` but `license: unknown`, and
the underlying articles stay copyright of their publishers. Treat the output as a
private research cache, like the other `-raw` corpora here -- not for redistribution.
Each row keeps its `source` (url, domain, date, title) so any downstream question about
provenance is answerable.

**Filtering.** Three passes, all measured rather than assumed:

* Language. The corpus is ~98.75% English by `lumi_language_id`, the rest `und` --
  short or garbled rows rather than another language. Keeping only `en` drops the junk.
* Length. Very short rows carry nothing to annotate; very long ones dominate the
  annotation bill for one document. Bounds are in characters, on the raw body.
* Duplicates. News syndication republishes the same wire story across outlets, so
  this is the corpus most likely to violate the split-uniqueness rule. Rows are
  deduplicated on `_split.normalize_group_key` -- the same document key the splitter
  and `check_leakage.py` use -- BEFORE anything is paid to annotate them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record, normalize_group_key  # noqa: E402

DATASET = "vblagoje/cc_news"


def load_keys(paths: List[Path]) -> Set[str]:
    """Document keys already collected, so a second pull cannot repeat the first.

    A fresh run of this script with the same seed returns the SAME documents --
    the shuffle is deterministic. Growing a corpus therefore means excluding what
    is already in it, not just changing the seed (which still overlaps heavily).
    """
    keys: Set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("input")
                if isinstance(text, str) and text.strip():
                    keys.add(normalize_group_key(text))
    return keys


def collect(count: int, min_chars: int, max_chars: int, seed: int,
            buffer_size: int, out: Path, exclude: Optional[Set[str]] = None) -> dict:
    """Stream, filter and deduplicate until `count` documents are written."""
    from datasets import load_dataset
    from lumi_language_id import detect_language

    ds = load_dataset(DATASET, split="train", streaming=True)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    stats = {"seen": 0, "empty": 0, "too_short": 0, "too_long": 0,
             "not_english": 0, "duplicate": 0, "already_held": 0, "written": 0}
    already: Set[str] = exclude or set()
    seen_keys: Set[str] = set()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in ds:
            stats["seen"] += 1
            text = (row.get("text") or "").strip()
            if not text:
                stats["empty"] += 1
                continue
            if len(text) < min_chars:
                stats["too_short"] += 1
                continue
            if len(text) > max_chars:
                stats["too_long"] += 1
                continue
            # LID on a prefix: enough signal, and it keeps the pass cheap on long rows.
            if detect_language(text[:1500])[0] != "en":
                stats["not_english"] += 1
                continue
            key = normalize_group_key(text)
            if key in already:
                stats["already_held"] += 1
                continue
            if key in seen_keys:
                stats["duplicate"] += 1
                continue
            seen_keys.add(key)

            fh.write(dumps_record({
                "input": text,
                "output": {},
                "source": {"url": row.get("url"), "domain": row.get("domain"),
                           "date": row.get("date"), "title": row.get("title")},
            }) + "\n")
            stats["written"] += 1
            if stats["written"] >= count:
                break
            if stats["written"] % 1000 == 0:
                print(f"  {stats['written']:,} written ({stats['seen']:,} seen)", flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=Path("data/cc_news_10k_raw.jsonl"))
    ap.add_argument("--min-chars", type=int, default=400,
                    help="below this a row has too little to annotate")
    ap.add_argument("--max-chars", type=int, default=12000,
                    help="caps the cost tail; ~3k tokens")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed; the raw stream is ordered by crawl")
    ap.add_argument("--buffer-size", type=int, default=10000)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="JSONL corpora already collected; their documents are skipped "
                         "so a second pull EXTENDS the first instead of repeating it")
    args = ap.parse_args()

    exclude = load_keys(args.exclude)
    if exclude:
        print(f"excluding {len(exclude):,} documents already held "
              f"({', '.join(str(p) for p in args.exclude)})")
    print(f"streaming {DATASET} -> {args.out} (target {args.count:,})")
    stats = collect(args.count, args.min_chars, args.max_chars,
                    args.seed, args.buffer_size, args.out, exclude)
    print(f"\nwrote {stats['written']:,} documents to {args.out}")
    for k in ("seen", "empty", "too_short", "too_long",
              "not_english", "duplicate", "already_held"):
        print(f"  {k:13s} {stats[k]:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
