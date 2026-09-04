"""The Chinese-language counterpart to build_turkey_feed.py, for the same event.

Same purpose as build_turkey_feed_tr.py: a real, dated document stream in a language the
pipeline claims to support, scored against the already-committed English-sourced
`turkey2023/ground_truth.json` rather than a new answer key built to match it.

WHY WAYBACK, NOT A LIVE FETCH. The live chinanews.com.cn and news.cn hosts are
unreachable from this environment -- TCP connect fails on every one of their IPs
("Bad file descriptor") while unrelated hosts (Wikipedia, web.archive.org, the
xinhuanet.com root) connect fine, so this is an environment-level block on specific
China-hosted ranges, not a dead site. The Wayback Machine has clean, repeated captures
of every URL below, so this fetches those instead -- consistent with how
harvest_turkey_gt.py already sources the English side.

WHY chinanews.com.cn (中新网), NOT shaowenchen/news_zh. The training corpus already in
this repo (`data/chinese_gate/`, via shaowenchen/news_zh) is a real but useless source
for this specific task: its own HF metadata states "From 2014 to 2016", seven years
before this earthquake. chinanews.com.cn's dated URL path (`/gj/2023/02-DD/...`) is what
made a real, dated Chinese stream findable at all.

URL LIST WAS HAND-CURATED, not a CDX crawl of the whole `/gj/2023/02-*` directory --
that directory holds thousands of unrelated international stories, so each URL below was
found by searching for this event specifically and confirmed on-page (see the module's
companion investigation). Coverage is 7 of the 16 days in the ground-truth window, thinner
than the Turkish feed's 14 -- reported as-is rather than padded.

The EARLIEST Wayback capture of each URL is used, closest to the article's real publish
time. Article text is NOT committed -- only it and the raw HTML are cached, gitignored.

    uv run python tools/ekf_showcase/build_turkey_feed_zh.py
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ONSET = datetime(2023, 2, 6, 1, 17, tzinfo=timezone.utc)
CDX = "http://web.archive.org/cdx/search/cdx?url={url}&output=json&filter=statuscode:200"
WAYBACK = "https://web.archive.org/web/{ts}/{url}"
UA = {"User-Agent": "gliner2-ekf-validation/1.0 (research; contact via repository)"}

# (date, article id) -- found by searching for this event on chinanews.com.cn specifically.
ARTICLES = [
    ("2023-02-06", "9948295"), ("2023-02-06", "9948286"), ("2023-02-06", "9948553"),
    ("2023-02-07", "9948685"), ("2023-02-07", "9948748"),
    ("2023-02-09", "9950158"), ("2023-02-09", "9950137"),
    ("2023-02-10", "9950856"), ("2023-02-10", "9951521"), ("2023-02-10", "9951184"),
    ("2023-02-13", "9952381"),
    ("2023-02-15", "9953670"), ("2023-02-15", "9954132"),
    ("2023-02-21", "9957292"),
]
BASE = "https://www.chinanews.com.cn/gj/2023/{d}/{aid}.shtml"


def _get(url: str, cache: Path) -> str:
    """Fetch with an on-disk cache; retries transient archive.org 5xx/timeouts."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read().decode("utf-8", errors="replace")
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1}/3 after {exc} ({url[:80]})")
            time.sleep(5.0 * (attempt + 1))
    cache.write_text(body, encoding="utf-8")
    time.sleep(1.0)
    return body


def earliest_snapshot(url: str, cache: Path) -> str | None:
    rows = json.loads(_get(CDX.format(url=url), cache / "cdx.json"))
    return rows[1][1] if len(rows) > 1 else None


def article_text(raw: str) -> str:
    """The `left_zw` body region -- the template's nav/section menu carries no facts.

    Bounded by the `<!--正文end-->` comment, NOT by the next `</div>`: the body nests
    image and caption divs, so a non-greedy `</div>` match ends at the first photo
    wrapper and returns a ~100-char fragment of the standfirst. That truncation is
    silent -- it yields plausible-looking text, not an error.

    Photo captions are KEPT. They are real article prose and they carry figures, often
    a day stale ("as of now, over 12,000 dead" beside a piece reporting 20,000), which
    is exactly the in-document distractor this feed exists to test binding against.
    """
    i = raw.find('<div class="left_zw"')
    if i < 0:
        return ""
    j = raw.find("<!--正文end-->", i)
    body = raw[i:j if j > 0 else len(raw)]
    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body))).strip()
    return re.sub(r"【编辑[:：][^】]*】", "", text).strip()   # trailing editor byline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="datasets/turkey2023/_cache/zh")
    ap.add_argument("--out", default="datasets/turkey2023/_cache/feed_zh.jsonl")
    args = ap.parse_args()

    cache = Path(args.cache)
    rows, misses = [], []
    for date, aid in ARTICLES:
        d = date[5:7] + "-" + date[8:10]
        url = BASE.format(d=d, aid=aid)
        ts = earliest_snapshot(url, cache / f"{aid}_cdx")
        if not ts:
            misses.append(url)
            continue
        raw = _get(WAYBACK.format(ts=ts, url=url), cache / f"{aid}.html")
        text = article_text(raw)
        # 50, not 100: the shortest genuine document is a 67-char CCTV wire brief whose
        # whole content is one exact toll ("1541人死亡、9733人受伤"), an early trajectory
        # point. A length floor tuned to catch extraction failures was discarding it.
        if len(text) < 50:
            misses.append(url)
            continue
        stamp = datetime(int(ts[:4]), int(ts[4:6]), int(ts[6:8]), int(ts[8:10]),
                         int(ts[10:12]), int(ts[12:14]), tzinfo=timezone.utc)
        rows.append({
            "t_hours": round((stamp - ONSET).total_seconds() / 3600.0, 2),
            "text": text,
            "date": date,
            "source_url": url,
            "archive_url": WAYBACK.format(ts=ts, url=url),
            "language": "zh",
        })

    rows.sort(key=lambda r: r["t_hours"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["text"]) for r in rows]
    print(f"[feed-zh] {len(rows)} documents  t {rows[0]['t_hours']}h .. {rows[-1]['t_hours']}h")
    print(f"[feed-zh] chars: min {min(lens)}  median {sorted(lens)[len(lens) // 2]}  max {max(lens)}")
    if misses:
        print(f"[feed-zh] {len(misses)} misses: {misses}")
    print(f"[feed-zh] wrote {out}  (article text is not committed)")


if __name__ == "__main__":
    main()
