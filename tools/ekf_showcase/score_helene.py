"""Score tracked Hurricane Helene streams against the per-state Wikipedia trajectory.

Two things this measures that Turkiye-Syria could not.

**Aggregate versus parts.** Helene reporting carries a national total and its state
components in the same sentence. Binding 227 to North Carolina is a specific, checkable
error, and it is reported separately from ordinary miss-binding because it is a different
mistake: the number is real, correctly extracted, and attached to a scope that does not
exist as a stream.

**A baseline that is not an oracle.** Ground truth comes from Wikipedia's casualty table
while the feed is AP prose, so `est_last_value` no longer reproduces the truth by
construction and the EKF/baseline comparison is finally meaningful. In Turkiye it was not:
the truth was read from the sentence the extractor reads, and the baseline scored 0.000.

    uv run python tools/ekf_showcase/score_helene.py \
        --tracked datasets/helene2024/_cache/tracked.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

ONSET = datetime(2024, 9, 26, 23, 10, tzinfo=timezone.utc)
STATES = ("Florida", "Georgia", "South Carolina", "North Carolina", "Tennessee",
          "Virginia", "Indiana")


def gt(path: Path):
    """state -> [(t_hours, deaths)], plus the Total series."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    series: dict[str, list] = {}
    for p in raw["points"]:
        ts = datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00"))
        t = (ts - ONSET).total_seconds() / 3600.0
        for name, v in p["deaths"].items():
            if not name.startswith("_"):
                series.setdefault(name, []).append((t, v))
    return series


def at(series, t):
    best = None
    for tt, v in series:
        if tt <= t:
            best = v
    return best


def nrmse(pred, series, grid):
    pairs = [(p, at(series, t)) for p, t in zip(pred, grid)]
    pairs = [(p, g) for p, g in pairs if g is not None]
    if not pairs:
        return None
    vals = [g for _, g in pairs]
    rng = max(vals) - min(vals)
    err = sqrt(sum((p - g) ** 2 for p, g in pairs) / len(pairs))
    return err / rng if rng > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracked", default="datasets/helene2024/_cache/tracked.json")
    ap.add_argument("--truth", default="datasets/helene2024/ground_truth.json")
    ap.add_argument("--mode", default="heuristic")
    ap.add_argument("--role", default="dead")
    args = ap.parse_args()

    res = json.loads(Path(args.tracked).read_text(encoding="utf-8"))
    truth, grid = gt(Path(args.truth)), res["grid"]

    obs = [o for a in res["articles"] for o in a["observations"]
           if o["role"] == args.role and o["mode"] == args.mode]
    print(f"[obs] {len(obs)} '{args.role}' observations over {res['n_articles']} articles")

    # Aggregate-vs-parts: a national total filed under one state is a specific error.
    totals = truth.get("Total", [])
    total_vals = {v for _, v in totals}
    mis = [o for o in obs if int(o["value"]) in total_vals
           and "|" in str(o.get("event_key", "")) ]
    print(f"[aggregate] observations whose value equals a NATIONAL TOTAL but are filed "
          f"under a single state: {len(mis)}")
    for o in mis[:6]:
        print(f"    t={o['t_hours']:>7.1f}h  value={int(o['value']):>4}  key={o['event_key']}")

    streams = res["tracked_by_event"][args.mode]
    print(f"\n[streams] {len(streams)} association key(s)")
    print(f"{'stream':<34}{'n':>4}  {'best-matching state':>20} {'nRMSE':>8} {'last_value':>11}")
    for key, s in streams.items():
        r = s.get(args.role, {})
        if not r.get("ekf"):
            continue
        scored = [(nrmse(r["ekf"], truth[st], grid), st) for st in STATES if st in truth]
        scored = [(v, st) for v, st in scored if v is not None]
        if not scored:
            continue
        best, st = min(scored)
        lv = nrmse(r["last_value"], truth[st], grid)
        print(f"{key[:34]:<34}{r['n_obs']:>4}  {st:>20} {best:>8.3f} "
              f"{('n/a' if lv is None else f'{lv:.3f}'):>11}")

    print("\n[truth] final per-state:", {k: v[-1][1] for k, v in truth.items() if v})


if __name__ == "__main__":
    main()
