"""Does the EKF absorb a genuine downward REVISION better than repeating the last reading?

The filter's core claim, tested on the only real revision data this project has. Wikipedia's
Helene table records North Carolina falling 123 -> 102 -> 96 as deaths were reclassified
from direct to indirect -- not noise, not a correction of a typo, but the authority changing
its mind. Turkiye-Syria was purely monotone and never exercised this.

Why it is a fair fight rather than a rigged one: a revision is the case where
``est_last_value`` looks BEST, because it jumps to the new figure immediately while a filter
lags. If the EKF wins here it wins because the observation stream is noisy and smoothing
beats chasing; if it loses, that is the honest answer and the claim in section 3 needs
narrowing.

Both estimators are imported from ``evaluate`` -- the ones that actually ship -- so this
measures the implementation and not a convenient re-derivation.

Truth is real. Only the reporting process is simulated (which snapshots get reported, with
what noise), because the real feed's per-state recall is still too thin to drive a filter.

    uv run python tools/ekf_showcase/revision_test.py --trials 60
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "datasets" / "disaster_streams"))
import evaluate as ekf  # noqa: E402

ONSET = datetime(2024, 9, 26, 23, 10, tzinfo=timezone.utc)
STATES = ("Florida", "Georgia", "South Carolina", "North Carolina", "Tennessee", "Virginia")


def series(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for p in raw["points"]:
        ts = datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00"))
        t = (ts - ONSET).total_seconds() / 3600.0
        for s, v in p["deaths"].items():
            if s in STATES:
                out.setdefault(s, []).append((t, float(v)))
    return out


def revisions(seq):
    """Indices where the authority revised DOWNWARD -- the case under test."""
    return [i for i in range(1, len(seq)) if seq[i][1] < seq[i - 1][1]]


def at(seq, t):
    best = None
    for tt, v in seq:
        if tt <= t:
            best = v
    return best


def nrmse(pred, seq, grid, only=None):
    pairs = [(p, at(seq, t)) for p, t in zip(pred, grid)]
    pairs = [(p, g) for (p, g), k in zip(pairs, range(len(pairs)))
             if g is not None and (only is None or k in only)]
    if not pairs:
        return None
    vals = [g for _, g in pairs]
    rng = max(vals) - min(vals)
    err = sqrt(sum((p - g) ** 2 for p, g in pairs) / len(pairs))
    return err / rng if rng > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", default="datasets/helene2024/ground_truth.json")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--p-report", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = series(Path(args.truth))
    print(f"{'state':<16}{'revisions':>10}{'ekf':>9}{'last_value':>12}{'winner':>10}"
          f"{'  ekf@revision':>15}{'last@revision':>15}")

    tot = {"ekf": 0.0, "last": 0.0, "ekf_rev": 0.0, "last_rev": 0.0, "n": 0, "wins": 0}
    for state in STATES:
        seq = data[state]
        revs = revisions(seq)
        grid = [t for t, _ in seq]
        rng = random.Random(args.seed)
        acc = {"ekf": [], "last": [], "ekf_rev": [], "last_rev": []}
        for _ in range(args.trials):
            obs = []
            for t, v in seq:
                if rng.random() < args.p_report:
                    obs.append({"t_hours": t, "role": "dead",
                                "value": max(v * (1 + rng.gauss(0, args.noise)), 0.0),
                                "qualifier": "point", "source": "official"})
            if not obs:
                continue
            e = ekf.est_ekf(obs, grid, "dead")
            l = ekf.est_last_value(obs, grid)
            # a small window AFTER each revision: where the two estimators disagree most
            win = {i + k for i in revs for k in range(0, 3)}
            for key, pred, only in (("ekf", e, None), ("last", l, None),
                                    ("ekf_rev", e, win), ("last_rev", l, win)):
                val = nrmse(pred, seq, grid, only)
                if val is not None:
                    acc[key].append(val)
        m = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
        wins = sum(1 for a, b in zip(acc["ekf"], acc["last"]) if a < b)
        winner = "EKF" if m["ekf"] < m["last"] else "last_value"
        print(f"{state:<16}{len(revs):>10}{m['ekf']:>9.3f}{m['last']:>12.3f}{winner:>10}"
              f"{m['ekf_rev']:>15.3f}{m['last_rev']:>15.3f}")
        for k in ("ekf", "last", "ekf_rev", "last_rev"):
            if m[k] == m[k]:               # states with no revision contribute no nan
                tot[k] += m[k]
                tot.setdefault(f"n_{k}", 0)
                tot[f"n_{k}"] += 1
        tot["n"] += 1; tot["wins"] += wins

    def mean(k):
        c = tot.get(f"n_{k}", 0)
        return tot[k] / c if c else float("nan")
    print(f"\n{'MEAN':<16}{'':>10}{mean('ekf'):>9.3f}{mean('last'):>12.3f}"
          f"{('EKF' if mean('ekf') < mean('last') else 'last_value'):>10}"
          f"{mean('ekf_rev'):>15.3f}{mean('last_rev'):>15.3f}")
    print("\nA revision is where last_value looks BEST -- it adopts the new figure at once "
          "while a\nfilter lags. The EKF can only win by smoothing noise the baseline chases.")


if __name__ == "__main__":
    main()
