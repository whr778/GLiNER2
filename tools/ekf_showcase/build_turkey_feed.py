"""Turn the archived Turkiye-Syria tracker snapshots into a pipeline feed.

One document per day, 6-21 Feb 2023, each the article as it stood that morning. The
test this sets up is narrow but real, and it is aimed squarely at the thing the
multi-event corpus was built to fix: **every document describes two events at once**
-- Turkiye and Syria -- each with its own death toll in the same sentence. Binding the
Syrian figure to the Turkish stream is precisely the error measured at 22.6% before
retraining.

Three properties make it a fair test rather than a regex exercise:

1. **Two co-occurring events, one document.** Association must split them by location;
   the event TYPE is identical (both earthquakes), so type alone pools them.
2. **Real distractor numbers.** The body carries the magnitude (7.8), the hour (4:17),
   aftershock and province counts, and -- the sharp one -- "the official death toll
   stood at 17,500", which belongs to the 1999 Izmit earthquake. A pipeline that reads
   any toll-shaped number will bind a 1999 figure into a 2023 stream.
3. **Genuine reporting dynamics.** The tolls do not rise smoothly: 9,057 -> 17,674
   overnight on 10 Feb as access improved, then flat days where nothing was revised.

Article text is NOT committed -- it is Al Jazeera's. The feed is written under the
gitignored cache; only extracted figures and URLs are versioned.

    uv run python tools/ekf_showcase/build_turkey_feed.py
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from harvest_turkey_gt import TRACKER, WAYBACK, _get, _text, snapshots

ONSET = datetime(2023, 2, 6, 1, 17, tzinfo=timezone.utc)
START, END = "Live tracker", "Source : Al Jazeera"


def article(raw: str) -> str:
    """The article region only -- site navigation and the footer carry no event facts."""
    t = _text(raw)
    i = t.find(START)
    j = t.find(END, i + 1)
    return t[i + len(START):j if j > 0 else len(t)].strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", default="datasets/turkey2023/ground_truth.json")
    ap.add_argument("--cache", default="datasets/turkey2023/_cache")
    ap.add_argument("--out", default="datasets/turkey2023/_cache/feed.jsonl")
    args = ap.parse_args()

    gt = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    keep = {p["snapshot"][:10] for p in gt["points"]}      # the un-stale window only
    cache = Path(args.cache)

    rows = []
    for ts in snapshots("20230206", "20230415", cache):
        day = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        if day not in keep:
            continue
        stamp = datetime(int(ts[:4]), int(ts[4:6]), int(ts[6:8]), int(ts[8:10]),
                         int(ts[10:12]), int(ts[12:14]), tzinfo=timezone.utc)
        text = article(_get(WAYBACK.format(ts=ts, url=TRACKER), cache / f"{ts}.html"))
        rows.append({
            "t_hours": round((stamp - ONSET).total_seconds() / 3600.0, 2),
            "text": text,
            "date": day,
            "archive_url": WAYBACK.format(ts=ts, url=TRACKER),
        })

    rows.sort(key=lambda r: r["t_hours"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["text"]) for r in rows]
    print(f"[feed] {len(rows)} documents  t {rows[0]['t_hours']}h .. {rows[-1]['t_hours']}h")
    print(f"[feed] chars: min {min(lens)}  median {sorted(lens)[len(lens) // 2]}  max {max(lens)}")
    izmit = sum(1 for r in rows if "17,500" in r["text"])
    print(f"[feed] documents carrying the 1999 Izmit distractor (17,500): {izmit}/{len(rows)}")
    print(f"[feed] wrote {out}  (article text is not committed)")


if __name__ == "__main__":
    main()
