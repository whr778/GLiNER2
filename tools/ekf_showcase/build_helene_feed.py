"""Build a Hurricane Helene news feed from archived AP wire copy.

This is the half of the Helene validation that the Turkiye-Syria run could not do.
There, ground truth was read from the same sentence the extractor reads, so
``est_last_value`` scored 0.000 by construction and the filter was unmeasurable. Here
the two sources are deliberately different:

    ground truth  <- English Wikipedia's per-state casualty TABLE (harvest_helene_gt.py)
    feed          <- Associated Press articles, prose, this file

News prose lags and disagrees with the encyclopaedic table -- which is the point. A
figure repeated from yesterday's wire copy is no longer automatically correct, so
"repeat the last reading" stops being an oracle and a filter finally has something to do.

Design notes:

- **Many dated articles, not one page re-snapshotted.** Each article is a document at a
  time, which is what a real feed looks like. Publication time is taken as the EARLIEST
  Wayback capture, which is a lower bound on when the text existed.
- **Articles are kept only if they carry a state and a toll.** Helene generated a great
  deal of coverage about aid, politics and sport; those are gate negatives, and a handful
  are kept on purpose so the gate has something to reject.
- Article text is NOT committed. It is AP's. Only the derived feed lives under the
  gitignored cache, and the harvester regenerates it from the archive.

    uv run python tools/ekf_showcase/build_helene_feed.py
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

HUB = "apnews.com/hub/hurricane-helene"
CDX = ("http://web.archive.org/cdx/search/cdx?url={url}&output=json"
       "&filter=statuscode:200&collapse=timestamp:8{extra}")
WAYBACK = "https://web.archive.org/web/{ts}/{url}"
UA = {"User-Agent": "gliner2-ekf-validation/1.0 (research; contact via repository)"}
ONSET = datetime(2024, 9, 26, 23, 10, tzinfo=timezone.utc)
STATES = ("North Carolina", "South Carolina", "Tennessee", "Georgia", "Florida", "Virginia")
TOLL = re.compile(r"kill|dead|death|toll|died|fatalit", re.I)


def _get(url: str, cache: Path, tries: int = 4) -> str:
    """Fetch with an on-disk cache. A FAILURE IS NEVER CACHED.

    Caching failures is worse than not caching: the archive rate-limits under sustained
    use, and writing a placeholder turns a transient 429 into a permanent one that no
    re-run can clear. Retries back off; a capture that stays dead returns "" so one bad
    article cannot abort the harvest.
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
            time.sleep(1.5)
            return body
        except Exception:
            time.sleep(3.0 * (attempt + 1))       # back off; the archive throttles
    return ""


def plain(raw: str) -> str:
    """The AP story body only.

    Two things must go, and neither is optional. The Wayback Machine injects its own
    toolbar (`id="wm-ipp-base"`) into every capture, and AP pages carry navigation,
    related-story rails and a footer. Taking the whole page put BOTH into the feed: the
    median document was 26,598 characters and articles about a CIA misconduct case or a
    four-day workweek registered as naming Florida, Tennessee and Georgia -- chrome, not
    content. A feed like that measures page furniture, not extraction.

    So the body is selected structurally (`RichTextStoryBody`), not by regex over the
    whole document.
    """
    try:
        from lxml import html as lhtml
    except ImportError:
        raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S)
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))).strip()

    try:
        tree = lhtml.fromstring(raw)
    except Exception:
        return ""
    for bad in tree.xpath('//*[starts-with(@id,"wm-ipp")] | //script | //style | //nav | //footer'):
        bad.getparent().remove(bad) if bad.getparent() is not None else None
    nodes = tree.xpath('//div[contains(@class,"RichTextStoryBody")]')
    if not nodes:
        nodes = tree.xpath('//div[contains(@class,"Page-storyBody")]') or tree.xpath("//main")
    if not nodes:
        return ""
    head = tree.xpath("//h1//text()")
    body = " ".join(n.text_content() for n in nodes)
    return re.sub(r"\s+", " ", html.unescape(" ".join(head) + " " + body)).strip()


def article_urls(cache: Path) -> set[str]:
    """Every AP article linked from any archived snapshot of the Helene hub."""
    body = _get(CDX.format(url=HUB, extra="&from=20240928&to=20241110"),
                cache / "cdx_hub.json")
    rows = json.loads(body)[1:] if body.strip().startswith("[") else []
    urls: set[str] = set()
    for r in rows:
        raw = _get(WAYBACK.format(ts=r[1], url="https://" + HUB), cache / f"hub_{r[1]}.html")
        urls |= set(re.findall(r"https?://apnews\.com/article/[a-z0-9\-]+", raw))
    return urls


def first_capture(url: str, cache: Path) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "_", url.split("/article/")[-1])[:60]
    body = _get(CDX.format(url=url.replace("https://", ""), extra="&limit=1"),
                cache / f"cdx_{key}.json")
    rows = json.loads(body)[1:] if body.strip().startswith("[") else []
    return rows[0][1] if rows else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="datasets/helene2024/_cache")
    ap.add_argument("--out", default="datasets/helene2024/_cache/feed.jsonl")
    ap.add_argument("--max-articles", type=int, default=120)
    args = ap.parse_args()
    cache = Path(args.cache)

    urls = sorted(article_urls(cache))[: args.max_articles]
    print(f"[hub] {len(urls)} unique AP article URLs")

    rows, skipped = [], 0
    for url in urls:
        ts = first_capture(url, cache)
        if not ts:
            skipped += 1
            continue
        key = re.sub(r"[^a-z0-9]+", "_", url.split("/article/")[-1])[:60]
        text = plain(_get(WAYBACK.format(ts=ts, url=url), cache / f"art_{key}.html"))
        # Relevance: the hub links AP's general navigation too, so a capture is only a
        # Helene document if the STORY BODY says so. Without this, unrelated articles
        # (a CIA case, a four-day-workweek feature) entered the feed and registered
        # multiple states from page chrome.
        if len(text) < 400 or "helene" not in text.lower():
            skipped += 1
            continue
        stamp = datetime(int(ts[:4]), int(ts[4:6]), int(ts[6:8]), int(ts[8:10]),
                         int(ts[10:12]), int(ts[12:14]), tzinfo=timezone.utc)
        rows.append({
            "t_hours": round((stamp - ONSET).total_seconds() / 3600.0, 2),
            "text": text,
            "date": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}",
            "states_named": [s for s in STATES if s in text],
            "has_toll": bool(TOLL.search(text)),
            "archive_url": WAYBACK.format(ts=ts, url=url),
        })

    rows.sort(key=lambda r: r["t_hours"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    multi = [r for r in rows if len(r["states_named"]) >= 2 and r["has_toll"]]
    lens = sorted(len(r["text"]) for r in rows)
    print(f"[feed] {len(rows)} articles ({skipped} skipped), t {rows[0]['t_hours']:.1f}h "
          f".. {rows[-1]['t_hours']:.1f}h")
    print(f"[feed] chars: min {lens[0]}  median {lens[len(lens) // 2]}  max {lens[-1]}")
    print(f"[feed] carrying >=2 states AND a toll word: {len(multi)}")
    print(f"[feed] carrying no toll word at all (gate negatives): "
          f"{sum(1 for r in rows if not r['has_toll'])}")
    print(f"[feed] wrote {out}  (AP article text is not committed)")


if __name__ == "__main__":
    main()
