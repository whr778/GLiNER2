"""Build a 2020 Aegean Sea earthquake news feed from archived Turkish English-language wire copy.

The third event, and the first with genuinely independent sources on both sides:

    ground truth  <- English Wikipedia REVISION HISTORY infobox (harvest_aegean_gt.py)
    feed          <- Hurriyet Daily News / Daily Sabah article prose, this file

Helene already separates the two (Wikipedia table vs AP prose). Turkiye 2023 does not --
there the tracker supplies the figure and the tracker's own prose is what gets parsed, so
`est_last_value` scores 0.000 by construction. This feed keeps Helene's separation and
improves on it: the ground truth is a timestamped revision series rather than a static
table, so it carries the event's dynamics including a real downward reclassification
(Izmir 116 -> 114 on 5 November).

WHY THIS EVENT. The collapse (one HMM emission absorbing the date gate, the scope gate and
page furniture) is demonstrated on Helene and cannot be validated on Turkiye, because
Turkiye's contaminant is the same order of magnitude as the event (Izmit's 17,500 against
a 50,000-death event, and it CROSSES the trajectory). Here the event is 119 deaths and the
historical comparison journalists reach for is Izmit 1999 at ~17,000 -- a 143x separation.

Measured before building: 15 of 16 Al Jazeera articles about Turkiye 2023 mention Izmit
1999, while 0 of 55 contemporaneous Wikipedia revisions about THIS event do. Journalists
carry the historical comparison; encyclopedia stubs do not. That is why the documents come
from news copy and only the ground truth comes from Wikipedia.

Article text is NOT committed. It belongs to its publishers. Only the derived feed lives
under the gitignored cache, and this harvester regenerates it from the archive.

    uv run python tools/ekf_showcase/build_aegean_feed.py
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

HUBS = [
    ("hurriyetdailynews.com", r"https?://www\.hurriyetdailynews\.com/[a-z0-9\-]+-\d+"),
    ("dailysabah.com/turkey", r"https?://www\.dailysabah\.com/turkey/[a-z0-9\-/]+"),
]
CDX = ("http://web.archive.org/cdx/search/cdx?url={url}&output=json"
       "&filter=statuscode:200&collapse=timestamp:8{extra}")
WAYBACK = "https://web.archive.org/web/{ts}/{url}"
UA = {"User-Agent": "gliner2-ekf-validation/1.0 (research; contact via repository)"}
ONSET = datetime(2020, 10, 30, 11, 51, tzinfo=timezone.utc)
PLACES = ("Izmir", "İzmir", "Samos", "Turkey", "Greece", "Bayrakli", "Seferihisar")
TOLL = re.compile(r"kill|dead|death|toll|died|fatalit", re.I)
QUAKE = re.compile(r"earthquake|quake|deprem", re.I)


def _get(url: str, cache: Path, tries: int = 5) -> str:
    """Fetch with an on-disk cache. A FAILURE IS NEVER CACHED.

    The archive throttles hard under sustained use -- this harvest hit 429 on the third
    request during feasibility testing -- and caching a failure turns a transient 429 into
    a permanent one no re-run can clear. Backoff is deliberately generous.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=120) as r:
                body = r.read().decode("utf-8", errors="replace")
            cache.write_text(body, encoding="utf-8")
            time.sleep(2.5)
            return body
        except Exception:
            time.sleep(8.0 * (attempt + 1))
    return ""


def plain(raw: str) -> str:
    """Readable body text from an archived page."""
    raw = re.sub(r"(?is)<(script|style|noscript|nav|footer|header)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>", "\n", raw)
    txt = re.sub(r"(?s)<[^>]+>", " ", raw)
    txt = html.unescape(txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)
    return txt.strip()


def article_urls(cache: Path) -> set[str]:
    urls: set[str] = set()
    for hub, pat in HUBS:
        body = _get(CDX.format(url=hub, extra="&from=20201030&to=20201125"),
                    cache / f"cdx_hub_{re.sub(r'[^a-z]+','_',hub)}.json")
        rows = json.loads(body)[1:] if body.strip().startswith("[") else []
        print(f"[hub] {hub}: {len(rows)} snapshots")
        for r in rows:
            raw = _get(WAYBACK.format(ts=r[1], url="https://" + hub),
                       cache / f"hub_{r[1]}_{re.sub(r'[^a-z]+','_',hub)[:12]}.html")
            urls |= set(re.findall(pat, raw))
    return urls


def first_capture(url: str, cache: Path, key: str) -> str | None:
    body = _get(CDX.format(url=url.replace("https://", ""), extra="&limit=1"),
                cache / f"cdx_{key}.json")
    rows = json.loads(body)[1:] if body.strip().startswith("[") else []
    return rows[0][1] if rows else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="datasets/aegean2020/_cache")
    ap.add_argument("--out", default="datasets/aegean2020/_cache/feed.jsonl")
    ap.add_argument("--max-articles", type=int, default=250)
    args = ap.parse_args()
    cache = Path(args.cache)

    urls = sorted(article_urls(cache))
    quake = [u for u in urls if QUAKE.search(u) or re.search(r"izmir|aegean|samos", u, re.I)]
    print(f"[hub] {len(urls)} article URLs, {len(quake)} match the event by slug")
    urls = (quake + [u for u in urls if u not in quake])[: args.max_articles]

    rows, skipped = [], 0
    for url in urls:
        key = re.sub(r"[^a-z0-9]+", "_", url.split("//")[-1])[:70]
        ts = first_capture(url, cache, key)
        if not ts:
            skipped += 1
            continue
        text = plain(_get(WAYBACK.format(ts=ts, url=url), cache / f"art_{key}.html"))
        # Relevance from the STORY BODY, not the slug: hub pages link general navigation,
        # and the Helene build showed unrelated articles entering the feed and registering
        # places from page chrome.
        if len(text) < 400 or not QUAKE.search(text):
            skipped += 1
            continue
        stamp = datetime(int(ts[:4]), int(ts[4:6]), int(ts[6:8]), int(ts[8:10]),
                         int(ts[10:12]), int(ts[12:14]), tzinfo=timezone.utc)
        rows.append({
            "t_hours": round((stamp - ONSET).total_seconds() / 3600.0, 2),
            "text": text,
            "date": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}",
            "places_named": sorted({p for p in PLACES if p in text}),
            "has_toll": bool(TOLL.search(text)),
            "mentions_1999": bool(re.search(r"\b1999\b", text)),
            "archive_url": WAYBACK.format(ts=ts, url=url),
        })

    rows.sort(key=lambda r: r["t_hours"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n99 = sum(r["mentions_1999"] for r in rows)
    print(f"[feed] {len(rows)} articles written, {skipped} skipped -> {out}")
    print(f"[feed] {n99} mention 1999 (the Izmit contaminant this event was chosen for)")
    if rows:
        print(f"[feed] t_hours {rows[0]['t_hours']:.1f} .. {rows[-1]['t_hours']:.1f}")


if __name__ == "__main__":
    main()
