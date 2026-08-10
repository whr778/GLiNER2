"""Score tracked Turkiye-Syria streams against the sourced trajectory.

Kept separate from `run_pipeline.py` on purpose. The pipeline's built-in `--truth`
handling pools truth across streams, which is exactly the assumption under test here:
this event has TWO reporting authorities whose tolls differ by ~7x over the same window,
so a pooled score cannot say whether association worked.

Each tracked stream is scored against BOTH national trajectories and reported against
both. Which country a stream actually followed is a finding, not an input -- assigning
streams to whichever trajectory flatters them would manufacture the result.

nRMSE is normalized by the truth's range, so 1.0 is roughly the score of predicting a
constant. The `est_last_value` baseline is carried alongside because a filter that cannot
beat "repeat the last number you read" has earned nothing.

    uv run python tools/ekf_showcase/score_turkey.py \
        --tracked datasets/turkey2023/_cache/tracked_frozen.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

ONSET = datetime(2023, 2, 6, 1, 17, tzinfo=timezone.utc)
IZMIT = 17500          # the 1999 quake's toll, present in 15/16 documents


def gt_series(path: Path):
    """(t_hours, turkiye, syria) per sourced point."""
    gt = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for p in gt["points"]:
        ts = datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00"))
        out.append(((ts - ONSET).total_seconds() / 3600.0, p["turkey"], p["syria"]))
    return out


def at(series, t, idx):
    """Last reported value at or before t -- reporting is a step function."""
    best = None
    for row in series:
        if row[0] <= t:
            best = row[idx]
    return best


def nrmse(pred, series, idx, grid):
    pairs = [(p, at(series, t, idx)) for p, t in zip(pred, grid)]
    pairs = [(p, g) for p, g in pairs if g is not None]
    if not pairs:
        return None
    vals = [g for _, g in pairs]
    rng = max(vals) - min(vals)
    err = sqrt(sum((p - g) ** 2 for p, g in pairs) / len(pairs))
    return err / rng if rng > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracked", default="datasets/turkey2023/_cache/tracked_frozen.json")
    ap.add_argument("--truth", default="datasets/turkey2023/ground_truth.json")
    ap.add_argument("--mode", default="heuristic")
    ap.add_argument("--role", default="dead")
    args = ap.parse_args()

    res = json.loads(Path(args.tracked).read_text(encoding="utf-8"))
    series, grid = gt_series(Path(args.truth)), res["grid"]

    obs = [o for a in res["articles"] for o in a["observations"]
           if o["role"] == args.role and o["mode"] == args.mode]
    print(f"[obs] {len(obs)} '{args.role}' observations over {res['n_articles']} documents")
    print(f"[obs] value range {min(o['value'] for o in obs):,.0f} .. "
          f"{max(o['value'] for o in obs):,.0f}")

    bound = [o for o in obs if abs(o["value"] - IZMIT) < 1]
    print(f"[izmit] 1999 toll ({IZMIT:,}) bound as a 2023 observation: "
          f"{'YES -- ' + str(len(bound)) + ' times' if bound else 'no'}")

    streams = res["tracked_by_event"][args.mode]
    print(f"\n[streams] {len(streams)} association key(s): {list(streams)}")
    print(f"{'stream':<34} {'n_obs':>5}  {'nRMSE vs Turkiye':>17}  {'nRMSE vs Syria':>15}")
    for key, s in streams.items():
        r = s.get(args.role, {})
        if not r.get("ekf"):
            continue
        tur, syr = nrmse(r["ekf"], series, 1, grid), nrmse(r["ekf"], series, 2, grid)
        print(f"{key[:34]:<34} {r['n_obs']:>5}  {_f(tur):>17}  {_f(syr):>15}")

    pooled = res["tracked"][args.mode][args.role]
    print(f"\n[pooled: all observations, one stream]")
    for est in ("ekf", "last_value"):
        tur, syr = nrmse(pooled[est], series, 1, grid), nrmse(pooled[est], series, 2, grid)
        print(f"  {est:<11} nRMSE vs Turkiye {_f(tur):>9}   vs Syria {_f(syr):>9}")

    final_t, final_s = series[-1][1], series[-1][2]
    print(f"\n[endpoint] truth at {grid[-1]:.0f}h: Turkiye {final_t:,}  Syria {final_s:,}")
    print(f"           ekf {pooled['ekf'][-1]:,.0f}   last_value {pooled['last_value'][-1]:,.0f}")


def _f(v):
    return "n/a" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
