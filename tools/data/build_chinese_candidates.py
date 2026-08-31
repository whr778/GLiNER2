"""Cue-bearing Simplified Chinese news, in the shape annotate_gate.py consumes.

Source: shaowenchen/news_zh -- 2,430,752 native full articles, median 794 chars, 7,409
distinct publishers including Xinhua and China News Service. Evaluated against two
rejected alternatives: ChineseNewsSummaryDaily is LLM summaries (median 284 chars, 15%
English, 3.6% Traditional) and ag_news_zh is machine-translated 2004 AG News headlines
(median 73 chars, visibly degraded -- "路透社路透社(路透社)", mangled numerals).

Only the AMBIGUOUS region is bought: a numeral within ~15 characters of a death or injury
word. Measured at 3.57% of articles over 300 chars, so roughly 68,000 cue-bearing
articles exist in the full corpus. Text with no cue is a free negative and never reaches
the API.

TRADITIONAL IS FILTERED OUT, not assumed absent. The corpus is 93.93% Simplified and only
0.03% Traditional, but a Traditional article would still train the model on characters the
Simplified deployment never sees.

    uv run python tools/data/build_chinese_candidates.py --limit 5000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record, normalize_group_key  # noqa: E402

URL = ("https://huggingface.co/datasets/shaowenchen/news_zh/resolve/main/"
       "data/train-news_zh.parquet")

# A numeral near a death/injury word. Deliberately broad: this selects what gets
# adjudicated, and the adjudication -- not this regex -- assigns the label.
DEATH = r"(?:死亡|遇难|丧生|身亡|死者|伤亡|受伤|失踪|罹难|遇害|重伤|轻伤)"
NUM = r"(?:[0-9０-９]+|[一二三四五六七八九十百千万]+)"
CUE = re.compile(NUM + r"(?:人|名|余人|多人)?.{0,15}?" + DEATH
                 + r"|" + DEATH + r".{0,15}?" + NUM + r"(?:人|名)")

# Characters that exist in only one script, used to reject Traditional articles.
SIMP = set("这个国说会时来对开发经济动车马门问题实现产业务爱与东书长风见龙飞习罗")
TRAD = set("這個國說會時來對開發經濟動車馬門問題實現產業務愛與東書長風見龍飛習羅")
CJK = re.compile(r"[一-鿿]")
MIN_CHARS, MAX_CHARS = 300, 6000


def is_simplified(text: str) -> bool:
    if len(CJK.findall(text)) < len(text) * 0.2:
        return False                      # latin-script contaminant
    return not any(c in TRAD for c in text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", default="data/chinese_gate/zh_candidates.jsonl")
    args = ap.parse_args()

    import fsspec
    import pyarrow.parquet as pq

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept, scanned, seen = 0, 0, set()
    with fsspec.open(URL).open() as fh, out_path.open("w", encoding="utf-8") as out:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(batch_size=5000, columns=["content", "source", "time"]):
            d = batch.to_pydict()
            for content, source, when in zip(d["content"], d["source"], d["time"]):
                scanned += 1
                text = str(content or "")[:MAX_CHARS]
                if len(text) < MIN_CHARS or not CUE.search(text) or not is_simplified(text):
                    continue
                key = normalize_group_key(text)[:300]
                if key in seen:
                    continue
                seen.add(key)
                out.write(dumps_record({"input": text, "source": str(source or "?"),
                                        "time": str(when or "")}) + "\n")
                kept += 1
                if kept >= args.limit:
                    print(f"[zh] {kept} candidates from {scanned:,} scanned "
                          f"({kept/scanned:.2%}) -> {out_path}")
                    return 0
    print(f"[zh] exhausted: {kept} candidates from {scanned:,} scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
