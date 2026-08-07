"""Track the Venezuela dead toll from extracted observations vs official GT (V4 mini-run).

Irregular-timestamp, real-data eval. Reuses `evaluate.py`'s estimators (they already handle
arbitrary grids -- they grow P by elapsed time). `dead` only; `missing` is unreliable on real
news (README). `--max-rate` drops dynamics-implausible UPWARD spikes -- e.g. the USGS
">10,000" *prediction* mis-bound to dead (a 188->10,000 jump in ~0h) -- which the one-sided
innovation gate ADMITS (it only rejects implausibly-low readings, to allow real jumps).

  uv run python datasets/venezuela_2026/track.py --min-conf 0.5 [--max-rate 100]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "datasets/disaster_streams")
import evaluate as E  # noqa: E402


def load(root: Path, role: str, min_conf: float):
    obs = [o for o in (json.loads(l) for l in (root / "observations.jsonl").open(encoding="utf-8"))
           if o["role"] == role and o.get("confidence", 1.0) >= min_conf]
    obs.sort(key=lambda o: o["t_hours"])
    gt = {}
    for line in (root / "trajectory.jsonl").open(encoding="utf-8"):
        t = json.loads(line)
        if t.get(role) is not None:
            gt[t["t_hours"]] = t[role]
    return obs, gt


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/venezuela_2026")
    ap.add_argument("--role", default="dead")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--max-rate", type=float, default=None, help="drop upward spikes > this/hour")
    ap.add_argument("--reject-sigma", type=float, default=None)
    args = ap.parse_args(argv)

    root = Path(args.data)
    obs, gt = load(root, args.role, args.min_conf)
    if args.max_rate is not None:
        n0 = len(obs); obs = E.rate_filter(obs, args.max_rate)
        print(f"[rate-filter] dropped {n0 - len(obs)} implausible upward spike(s) (>{args.max_rate}/h)")
    grid = sorted(gt)
    E.REJECT_SIGMA = args.reject_sigma

    lv = E.est_last_value(obs, grid)
    rm = E.est_running_max(obs, grid)
    ekf = E.est_ekf(obs, grid, args.role)

    print(f"{args.role} obs (conf>={args.min_conf}): " +
          ", ".join(f"{o['t_hours']}h:{o['value']}" for o in obs))
    print(f"\n{'t_h':>6}{'GT':>9}{'last_val':>10}{'run_max':>9}{'EKF':>9}")
    for i, t in enumerate(grid):
        print(f"{t:6d}{gt[t]:9}{lv[i]:10.0f}{rm[i]:9.0f}{ekf[i]:9.0f}")


if __name__ == "__main__":
    main()
