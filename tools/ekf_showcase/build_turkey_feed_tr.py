"""The Turkish-language counterpart to build_turkey_feed.py, for the same event.

`turkey2023/ground_truth.json` and `feed.jsonl` are both English (Al Jazeera). This adds
a document stream in the language the event actually happened in, so the pipeline's
Turkish support can be validated against a REAL, already-committed ground truth rather
than a new one built to match it.

The English feed re-samples ONE live-tracker page at successive Wayback snapshots. No
equivalent single Turkish page was found; what exists instead is a genuine multi-outlet,
multi-day stream -- `tr.euronews.com` publishes one dated article per URL
(`/2023/02/DD/...`), and 2023-02 coverage of this event already sits in
`/Volumes/Development/data/turkish_pool18.jsonl` (`umutertugrul/turkish-news-1.8M-tokenized`
via `build_turkish_pool.py`), which is why no network fetch is needed here. That pool
retains `url` where the downstream annotated corpora (`data/turkish_gate/*.jsonl`) do not.

NOT pre-filtered for relevance. The 48 rows in the 6-21 Feb window include unrelated
politics and sport (Manchester United's sale, Kılıçdaroğlu's remarks) alongside genuine
earthquake coverage -- same mix a real feed would have. Admission is the gate's job, not
this script's.

Only DAY resolution is available (no time-of-day in the URL), so every article on a given
day is stamped at that day's noon UTC. Coarser than the English feed's minute-level Wayback
timestamps, but ordering across days is exact and that is what the EKF tracker consumes.

Article text is NOT committed -- it lives in the pre-downloaded pool, itself outside the
repo. Only t_hours/date/url are written, alongside the text, under the gitignored cache.

    uv run python tools/ekf_showcase/build_turkey_feed_tr.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ONSET = datetime(2023, 2, 6, 1, 17, tzinfo=timezone.utc)
PREFIX = "https://tr.euronews.com/2023/02/"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="/Volumes/Development/data/turkish_pool18.jsonl")
    ap.add_argument("--first-day", type=int, default=6)
    ap.add_argument("--last-day", type=int, default=21)
    ap.add_argument("--out", default="datasets/turkey2023/_cache/feed_tr.jsonl")
    args = ap.parse_args()

    rows = []
    with open(args.pool, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            u = d.get("url", "")
            if not u.startswith(PREFIX):
                continue
            day = int(u[len(PREFIX):len(PREFIX) + 2])
            if not (args.first_day <= day <= args.last_day):
                continue
            stamp = datetime(2023, 2, day, 12, 0, tzinfo=timezone.utc)
            rows.append({
                "t_hours": round((stamp - ONSET).total_seconds() / 3600.0, 2),
                "text": d["input"],
                "date": f"2023-02-{day:02d}",
                "source_url": u,
                "language": "tr",
            })

    rows.sort(key=lambda r: r["t_hours"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["text"]) for r in rows]
    print(f"[feed-tr] {len(rows)} documents  t {rows[0]['t_hours']}h .. {rows[-1]['t_hours']}h")
    print(f"[feed-tr] chars: min {min(lens)}  median {sorted(lens)[len(lens) // 2]}  max {max(lens)}")
    by_day = {}
    for r in rows:
        by_day[r["date"]] = by_day.get(r["date"], 0) + 1
    print(f"[feed-tr] per day: {by_day}")
    print(f"[feed-tr] wrote {out}  (article text is not committed)")


if __name__ == "__main__":
    main()
