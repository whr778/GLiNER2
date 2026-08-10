"""Harvest per-state Hurricane Helene (2024) death tolls from archived Wikipedia.

Why this event, after Turkiye-Syria. Two limitations of that validation are fixed here.

1. **Aggregate versus parts.** Turkiye and Syria were two places with two independent
   tolls and no arithmetic between them. Helene is ONE event whose reporting carries a
   national total AND its state components in the same sentence -- "at least 227 deaths
   across six states, including 120 in North Carolina and 17 in Tennessee". Attribution
   must bind 120 to NC, 17 to TN, and refuse to file 227 under any single state. Nothing
   in the Turkiye feed tested that, and a naive extractor files the total under whichever
   state sits nearest.
2. **The oracle-baseline flaw.** In Turkiye the ground truth was read from the same
   sentence the extractor reads, so `est_last_value` scored 0.000 by construction and the
   filter could not be evaluated at all. Here ground truth comes from Wikipedia's casualty
   TABLE while the extraction feed comes from news prose, so the two are not the same text
   and repeating the last reading is no longer automatically correct.

Seven streams (six states plus Indiana) and a total, sampled daily. Monotonicity is the
fabrication check, with one honest exception: Wikipedia is edited, and a state's figure is
occasionally revised DOWNWARD when a death is reattributed or found indirect. Those are
reported rather than dropped -- a real revision is signal for a tracker, not noise.

    uv run python tools/ekf_showcase/harvest_helene_gt.py
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

PAGE = "https://en.wikipedia.org/wiki/Hurricane_Helene"
CDX = ("http://web.archive.org/cdx/search/cdx?url={url}&from={frm}&to={to}"
       "&output=json&filter=statuscode:200&collapse=timestamp:8")
WAYBACK = "https://web.archive.org/web/{ts}/{url}"
UA = {"User-Agent": "gliner2-ekf-validation/1.0 (research; contact via repository)"}
STATES = ("Florida", "Georgia", "South Carolina", "North Carolina", "Tennessee",
          "Virginia", "Indiana", "Total")


def _get(url: str, cache: Path) -> str:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        body = r.read().decode("utf-8", errors="replace")
    cache.write_text(body, encoding="utf-8")
    time.sleep(2.0)
    return body


def deaths(cell):
    """(value, low, high) from a casualty cell. The parenthetical is MISSING, not deaths.

    Cells are not all plain integers. Wikipedia carried `70-119[a] (200)` for North
    Carolina on 7 Oct while sources disagreed, and reading the leading integer turns that
    into a 119 -> 70 crash that looks like a revision and is not one. A range is genuine
    uncertainty and is kept as a range; the midpoint is the point estimate, so a
    disagreement does not masquerade as a confident low number.
    """
    text = re.sub(r"\[[^\]]*\]", " ", str(cell).replace("\xad", ""))
    text = text.split("(")[0]                                 # drop the missing count
    rng = re.search(r"(\d[\d,]*)\s*[-‒-―]\s*(\d[\d,]*)", text)
    if rng:
        lo, hi = (int(g.replace(",", "")) for g in rng.groups())
        return (lo + hi) // 2, lo, hi
    m = re.search(r"\d[\d,]*", text)
    if not m:
        return None
    v = int(m.group(0).replace(",", ""))
    return v, v, v


def casualty_table(raw: str):
    """state -> deaths, from the one table listing several states with a Deaths column."""
    import pandas as pd
    from io import StringIO
    for t in pd.read_html(StringIO(raw)):
        cols = [str(c) for c in t.columns]
        if not any("eath" in c for c in cols) or t.shape[1] < 2:
            continue
        first, death_col = t.columns[0], next(c for c in t.columns if "eath" in str(c))
        found = {}
        for _, row in t.iterrows():
            name = re.sub(r"\s+", " ", str(row[first])).strip()
            if name in STATES:
                d = deaths(row[death_col])
                if d is not None:
                    found[name] = d[0]
                    if d[1] != d[2]:
                        found.setdefault("_ranges", {})[name] = [d[1], d[2]]
        if len([k for k in found if not k.startswith("_")]) >= 3:
            return found
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="datasets/helene2024/ground_truth.json")
    ap.add_argument("--cache", default="datasets/helene2024/_cache")
    args = ap.parse_args()
    cache = Path(args.cache)

    rows = json.loads(_get(CDX.format(url=PAGE, frm="20240926", to="20241231"),
                           cache / "cdx.json"))[1:]
    print(f"[cdx] {len(rows)} snapshots")

    points = []
    for r in rows:
        ts = r[1]
        table = casualty_table(_get(WAYBACK.format(ts=ts, url=PAGE), cache / f"{ts}.html"))
        if not table:
            continue
        points.append({
            "snapshot": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:]}Z",
            "deaths": table,
            "archive_url": WAYBACK.format(ts=ts, url=PAGE),
        })
    points.sort(key=lambda p: p["snapshot"])

    print(f"[harvest] {len(points)} snapshots with a parsable casualty table\n")
    hdr = [s for s in STATES if any(s in p["deaths"] for p in points)]
    print("date        " + "".join(f"{s[:9]:>10}" for s in hdr))
    for p in points:
        print(f"{p['snapshot'][:10]}  " + "".join(f"{p['deaths'].get(s, '-'):>10}" for s in hdr))

    print("\n[check] downward revisions (real edits, reported not dropped):")
    peak, any_drop = {}, False
    for p in points:
        for s, v in p["deaths"].items():
            if s.startswith("_"):
                continue
            if s in peak and v < peak[s]:
                print(f"   {p['snapshot'][:10]}  {s}: {peak[s]} -> {v}")
                any_drop = True
            peak[s] = max(peak.get(s, 0), v)
    if not any_drop:
        print("   none - every state is monotonic non-decreasing")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "event": "Hurricane Helene (2024)",
        "onset_utc": "2024-09-26T23:10:00Z",
        "source": "English Wikipedia casualty table, sampled via the Wayback Machine",
        "note": "per-state streams plus a Total row; the total is an AGGREGATE of the parts",
        "points": points,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[harvest] wrote {out}")


if __name__ == "__main__":
    main()
