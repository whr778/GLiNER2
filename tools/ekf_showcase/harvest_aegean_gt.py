"""Ground truth for the 2020 Aegean Sea earthquake, from the Wikipedia revision history.

The article was edited heavily as the event unfolded -- 60 revisions in the first three
and a half hours -- and every revision carries an infobox `casualties` field with the
then-current figures split by country:

    [[Greece]]: 2 dead, 19 injured<br />[[Turkey]]: 114 dead, 1,035 injured

so each timestamped revision is one point on the trajectory. Public, versioned and
citable; a better source than the Wayback tracker used for Turkiye 2023, where the
snapshot times are whatever the crawler happened to take.

Same caveat as that feed, recorded plainly: ground truth and the document feed come from
the SAME source. The infobox supplies the figure, the prose is what the pipeline must
read, so this tests whether the extractor can recover a stated toll from surrounding text
carrying rival numbers -- not whether two independent sources agree.

Article prose is NOT committed. It caches under the gitignored data/ tree; only the
trajectory and the revision URLs are versioned.

    uv run python tools/ekf_showcase/harvest_aegean_gt.py
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TITLE = "2020 Aegean Sea earthquake"
ONSET = "2020-10-30T11:51:00Z"          # 14:51 local (UTC+3)
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "GLiNER2-research/1.0 (disaster-tracking research; whr778@gmail.com)"}
CACHE = Path("data/aegean2020_text_cache")

# "[[Greece]]: 2 dead" / "Greece: 2 dead" / "Turkey: 114 dead"
_CAS = re.compile(r"\[?\[?(Greece|Turkey|Turkiye)\]?\]?\s*:?\s*([\d,]+)\s*(?:people\s*)?dead",
                  re.I)
_INJ = re.compile(r"\[?\[?(Greece|Turkey|Turkiye)\]?\]?\s*:?\s*[\d,]+\s*(?:people\s*)?dead"
                  r"[^<]*?([\d,]+)\s*injured", re.I)


def _get(url: str, cache: Path) -> str:
    if cache.is_file():
        return cache.read_text(encoding="utf-8")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
        body = r.read().decode("utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(body, encoding="utf-8")
    time.sleep(0.4)                      # be polite to the API
    return body


def revisions(frm: str, to: str):
    """Every revision id + timestamp in the window, oldest first."""
    out, cont = [], None
    while True:
        q = {"action": "query", "prop": "revisions", "titles": TITLE, "rvlimit": "500",
             "rvstart": frm, "rvend": to, "rvdir": "newer",
             "rvprop": "timestamp|ids", "format": "json"}
        if cont:
            q["rvcontinue"] = cont
        url = f"{API}?{urllib.parse.urlencode(q)}"
        d = json.loads(_get(url, CACHE / f"revlist_{cont or 'first'}.json"))
        page = list(d["query"]["pages"].values())[0]
        out.extend(page.get("revisions", []))
        cont = (d.get("continue") or {}).get("rvcontinue")
        if not cont:
            return out


def sample(revs, bucket_hours: float):
    """Last revision of each time bucket -- a trajectory, not every keystroke."""
    by_bucket = {}
    for r in revs:
        ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
        by_bucket[int(ts.timestamp() // (bucket_hours * 3600))] = r
    return [by_bucket[k] for k in sorted(by_bucket)]


def content(revids):
    """Fetch wikitext for specific revisions, batched."""
    out = {}
    for i in range(0, len(revids), 20):
        chunk = revids[i:i + 20]
        q = {"action": "query", "prop": "revisions", "revids": "|".join(map(str, chunk)),
             "rvprop": "timestamp|ids|content", "rvslots": "main", "format": "json"}
        url = f"{API}?{urllib.parse.urlencode(q)}"
        d = json.loads(_get(url, CACHE / f"content_{chunk[0]}.json"))
        for page in d["query"]["pages"].values():
            for r in page.get("revisions", []):
                out[r["revid"]] = (r["timestamp"], r["slots"]["main"].get("*", ""))
    return out


def parse_casualties(wikitext: str):
    """Deaths and injured per country from the infobox casualties field."""
    m = re.search(r"\|\s*casualties\s*=(.{0,900}?)(?:\n\s*\||\n\}\})", wikitext, re.S | re.I)
    if not m:
        return None
    field = m.group(1)
    dead, inj = {}, {}
    for country, n in _CAS.findall(field):
        dead[_norm(country)] = int(n.replace(",", ""))
    for country, n in _INJ.findall(field):
        inj[_norm(country)] = int(n.replace(",", ""))
    return (dead, inj) if dead else None


def _norm(c: str) -> str:
    c = c.lower()
    return "Izmir" if c.startswith("turk") else "Samos"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="frm", default="2020-10-30T00:00:00Z")
    ap.add_argument("--to", default="2020-12-01T00:00:00Z")
    ap.add_argument("--bucket-hours", type=float, default=6.0)
    ap.add_argument("--out", default="datasets/aegean2020/ground_truth.json")
    a = ap.parse_args()

    revs = revisions(a.frm, a.to)
    print(f"{len(revs)} revisions in window")
    picked = sample(revs, a.bucket_hours)
    print(f"{len(picked)} sampled at {a.bucket_hours}h buckets")
    body = content([r["revid"] for r in picked])

    points, skipped = [], 0
    for r in picked:
        ts, wt = body.get(r["revid"], (None, ""))
        got = parse_casualties(wt)
        if not got:
            skipped += 1
            continue
        dead, inj = got
        dead = dict(dead)
        dead["Total"] = sum(v for k, v in dead.items() if k != "Total")
        points.append({"snapshot": ts, "revid": r["revid"], "deaths": dead,
                       "injured": inj,
                       "revision_url": f"https://en.wikipedia.org/w/index.php?oldid={r['revid']}"})
    print(f"{len(points)} points parsed, {skipped} revisions had no parsable casualties")

    prev = {}
    for p in points:
        for k, v in p["deaths"].items():
            if k in prev and v < prev[k]:
                print(f"  NOTE non-monotone {k}: {prev[k]} -> {v} at {p['snapshot']}")
        prev.update(p["deaths"])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "event": "2020 Aegean Sea earthquake",
        "onset_utc": ONSET,
        "source": "English Wikipedia revision history, infobox casualties field",
        "note": "Izmir = Turkiye's toll, Samos = Greece's. GT and the document feed share "
                "a source; see the module docstring.",
        "points": points,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(points)} points)")
    if points:
        print(f"  first {points[0]['snapshot']} {points[0]['deaths']}")
        print(f"  last  {points[-1]['snapshot']} {points[-1]['deaths']}")


if __name__ == "__main__":
    main()
