"""Harvest the 2023 Turkiye-Syria earthquake death-toll trajectory from archived pages.

Why this exists. The EKF showcase has so far been validated only on synthetic streams
whose ground truth we generated. A real event tests whether the tracker survives real
reporting dynamics -- lag, revision, competing sources, plateaus. Turkiye-Syria (6 Feb
2023) is the case: two countries reporting separately, a steep first week, and a long
tail of upward revision to a settled final figure.

Provenance is the whole point, so the rules are strict:

1. **One page, sampled over time.** Every figure comes from the SAME Al Jazeera live
   tracker URL, captured at successive Wayback snapshots. Stitching numbers from
   different outlets mixes reporting conventions (some count Turkiye only, some combine,
   some use rebel-held Syria figures) and produces a series whose jumps are artefacts of
   the source rather than the event.
2. **Read from page content, never from a search summary.** Search-result summaries were
   checked against each other and conflate dates -- they returned 31,643 for 9 Feb and
   17,134 for 10 Feb, which cannot both be true. Anything not read off fetched HTML is
   not admissible here.
3. **Anchored template, not any toll-shaped number.** The page also contains the 1999
   Izmit toll and headline links to other days. Matching "a big number near the word
   dead" picks those up. The standfirst is a fixed sentence naming both countries, so
   that is what gets matched.
4. **Turkiye and Syria stay separate.** They are separate reporting authorities with
   separate revision behaviour; merging them creates non-monotonic steps that are pure
   source artefact.

Monotonicity is the fabrication detector: a confirmed-death toll from a single authority
never decreases. Any point that breaks it is reported, not silently dropped.

    uv run python tools/ekf_showcase/harvest_turkey_gt.py \
        --out datasets/turkey2023/ground_truth.json
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request
from pathlib import Path

TRACKER = ("https://www.aljazeera.com/news/2023/2/6/"
           "turkey-syria-earthquake-death-toll-and-devastation-live-tracker")
CDX = ("http://web.archive.org/cdx/search/cdx?url={url}&from={frm}&to={to}"
       "&output=json&filter=statuscode:200&collapse=timestamp:8")
WAYBACK = "https://web.archive.org/web/{ts}/{url}"
UA = {"User-Agent": "gliner2-ekf-validation/1.0 (research; contact via repository)"}

# The standfirst names both countries in one sentence. Wording drifted over the month
# ("At least X deaths have been reported in Turkey, while Y people have died in Syria",
# later "More than X..."), so the anchor is the two country names around two numbers.
TOLL = re.compile(
    r"([\d,]{3,})\s+(?:deaths|people|have)[^.]{0,60}?Turkey[^.]{0,80}?([\d,]{3,})"
    r"[^.]{0,60}?Syria",
    re.I)


def _get(url: str, cache: Path) -> str:
    """Fetch with an on-disk cache so re-runs never re-hit the archive."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        body = r.read().decode("utf-8", errors="replace")
    cache.write_text(body, encoding="utf-8")
    time.sleep(2.0)                      # be polite to the archive
    return body


def _text(raw: str) -> str:
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))


def snapshots(frm: str, to: str, cache: Path):
    rows = json.loads(_get(CDX.format(url=TRACKER, frm=frm, to=to), cache / "cdx.json"))
    return [r[1] for r in rows[1:]]


def harvest(timestamps, cache: Path):
    points, misses = [], []
    for ts in timestamps:
        raw = _get(WAYBACK.format(ts=ts, url=TRACKER), cache / f"{ts}.html")
        m = TOLL.search(_text(raw))
        if not m:
            misses.append(ts)
            continue
        points.append({
            "snapshot": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:]}Z",
            "turkey": int(m.group(1).replace(",", "")),
            "syria": int(m.group(2).replace(",", "")),
            "source_url": TRACKER,
            "archive_url": WAYBACK.format(ts=ts, url=TRACKER),
        })
    return points, misses


def check_monotonic(points, field):
    """Indices where the series decreases -- a single authority's confirmed toll cannot."""
    bad, peak = [], 0
    for i, p in enumerate(points):
        if p[field] < peak:
            bad.append((i, p["snapshot"], p[field], peak))
        peak = max(peak, p[field])
    return bad


def truncate_stale(points, field="turkey"):
    """Drop the tail after the source stopped updating.

    The tracker freezes at 41,000 on 21 Feb and reports that unchanged into April, while
    the real toll went on to 53,537. That flat run is the PAGE going stale, not the event
    plateauing -- proven by Al Jazeera's own reporting elsewhere on 25 Feb, which gave
    44,218 as of 24 Feb while this page still said 41,000 on 26 Feb.

    Keeping it would be actively harmful: a tracker that simply stopped updating would
    score perfectly across six stale weeks. So the series ends at the last genuine change.
    """
    last = max(i for i in range(len(points)) if i == 0 or points[i][field] != points[i - 1][field])
    return points[:last + 1], points[last + 1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="datasets/turkey2023/ground_truth.json")
    ap.add_argument("--cache", default="datasets/turkey2023/_cache")
    ap.add_argument("--from-date", default="20230206")
    ap.add_argument("--to-date", default="20230415")
    args = ap.parse_args()

    cache = Path(args.cache)
    ts = snapshots(args.from_date, args.to_date, cache)
    print(f"[cdx] {len(ts)} snapshots {ts[0]} .. {ts[-1]}")

    points, misses = harvest(ts, cache)
    points.sort(key=lambda p: p["snapshot"])
    print(f"[harvest] extracted {len(points)}  template-miss {len(misses)}")
    if misses:
        print(f"[harvest] no template match: {', '.join(misses)}")

    for field in ("turkey", "syria"):
        bad = check_monotonic(points, field)
        status = "monotonic" if not bad else f"DECREASES at {bad}"
        print(f"[check] {field:7s}: {status}")

    points, stale = truncate_stale(points)
    print(f"[stale] dropped {len(stale)} snapshots after the source froze "
          f"(last genuine update {points[-1]['snapshot'][:10]})")

    first_week = [p for p in points if p["snapshot"] < "2023-02-14"]
    print(f"[check] points in steep first week (6-13 Feb): {len(first_week)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "event": "2023 Turkiye-Syria earthquakes",
        "onset_utc": "2023-02-06T01:17:00Z",
        "source": "Al Jazeera live tracker, sampled via the Wayback Machine",
        "note": "turkey and syria are separate reporting authorities; keep as separate streams",
        "coverage": {
            "window": [points[0]["snapshot"], points[-1]["snapshot"]],
            "truncated_after": "source stopped updating; flat tail to 2023-04-09 dropped",
            "staleness_evidence": (
                "Al Jazeera reported 44,218 for Turkiye as of 24 Feb 2023 "
                "(aljazeera.com/news/2023/2/25/death-toll-climbs-above-50000-after-"
                "turkey-syria-earthquakes) while this tracker still showed 41,000 on 26 Feb"),
            "settled_final_not_in_series": {
                "turkiye": 53537, "syria_government": 1414, "syria_rebel_held": 4537,
                "source": "en.wikipedia.org/wiki/2023_Turkey-Syria_earthquakes"},
        },
        "points": points,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[harvest] wrote {out}")
    for p in points:
        print(f"  {p['snapshot'][:10]}  Turkiye {p['turkey']:>6,}   Syria {p['syria']:>6,}")


if __name__ == "__main__":
    main()
