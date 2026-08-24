"""Does a national total CONSTRAIN the per-state estimates, or is it just noise to discard?

The aggregate-vs-parts question, made testable. Helene reporting carries a national total
and its state components in the same sentence -- "227 across six states, including 120 in
North Carolina and 17 in Tennessee". The current pipeline files that 227 under whichever
state is nearest, which is wrong: it is not a rival claim about North Carolina, it is
information about all six states at once.

The claim under test is that this is a MEASUREMENT-MODEL question, not a clustering one.
Make the state a vector and every report becomes a linear observation that differs only in
its H row:

    "120 in North Carolina"      H = e_NC        z = 120
    "227 across six states"      H = [1,1,...]   z = 227

A Kalman filter fuses both natively. Neither a Gaussian mixture nor mean shift can express
the sum at all, and both would cluster in VALUE space -- where 12 (Florida) and 17
(Tennessee) look like one cluster while 120 and 227 look like two, which is backwards.

**Two arms, same filter, same dynamics, same per-state observations.** The only difference
is whether aggregate reports are used:

    parts-only   6 independent scalar filters; totals DISCARDED   (what ships today)
    vector       one 6-dim filter; totals enter as a SUM ROW      (proposed)

**Ground truth is real**, including its revisions: Wikipedia's per-state casualty table
across 31 dated snapshots, in which North Carolina genuinely falls 123 -> 102 -> 96 as
deaths were reclassified. Only the reporting PROCESS is simulated -- which reports arrive
when, and with what noise -- because the real feed's recall (25 observations from 70
articles) is too thin to test a filter, and that is a separate problem.

The simulated regime is the measured one: totals are frequent, per-state breakdowns are
sparse. That is what AP actually does, and it is precisely where an aggregate should earn
its keep -- the total tracks the sum quickly while sparse per-state reports pin the split.

    uv run python tools/ekf_showcase/vector_state_test.py --trials 40
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import numpy as np

ONSET = datetime(2024, 9, 26, 23, 10, tzinfo=timezone.utc)
STATES = ("Florida", "Georgia", "South Carolina", "North Carolina", "Tennessee", "Virginia")


def truth_series(path: Path):
    """(hours, x) per snapshot, x = per-state vector in STATES order."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for p in raw["points"]:
        ts = datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00"))
        d = p["deaths"]
        if all(s in d for s in STATES):
            out.append(((ts - ONSET).total_seconds() / 3600.0,
                        np.array([float(d[s]) for s in STATES])))
    return out


def kalman(obs, n_steps, dim, q, seed_x, seed_P, q_prop: float = 0.15,
           q_rho: float = 0.0):
    """Plain linear KF over a random-walk state; obs = [(step, H, z, r), ...].

    Deliberately the simplest filter that can express the question, and IDENTICAL for both
    arms -- the arms differ only in which observations they are handed, so any difference
    is the measurement model and not the tuning.
    """
    x = seed_x.astype(float).copy()
    P = np.eye(dim) * seed_P
    by_step: dict[int, list] = {}
    for step, H, z, r in obs:
        by_step.setdefault(step, []).append((H, z, r))
    traj = []
    for k in range(n_steps):
        # Isotropic Q is WRONG when components differ by orders of magnitude (Virginia
        # ranges 1->2, North Carolina 6->123): with H = [1,1,...] an aggregate's correction
        # is then spread EQUALLY, so a national update shoves Virginia as hard as North
        # Carolina. q_prop scales process noise with the current estimate, which is what
        # makes the gain distribute proportionally instead.
        Qd = np.eye(dim) * q
        if q_prop:
            sig = q_prop * np.maximum(np.abs(x), 1.0)
            if q_rho:
                # Both diagonal options assert state tolls accrue INDEPENDENTLY. One storm
                # does not, and the aggregate row H=[1..1] is where that bites, since
                # Var(sum) = sum_ij P_ij. C keeps the marginals and adds uniform
                # correlation, so any change is the correlation and not a bigger Q.
                C = (1 - q_rho) * np.eye(dim) + q_rho * np.ones((dim, dim))
                Qd = np.diag(sig) @ C @ np.diag(sig)
            else:
                Qd = np.diag(sig ** 2)
        P = P + Qd                                    # predict (random walk)
        for H, z, r in by_step.get(k, []):
            H = H.reshape(1, -1)
            S = float((H @ P @ H.T).item()) + r
            K = (P @ H.T) / S
            x = x + (K * (z - float((H @ x).item()))).ravel()
            P = P - K @ H @ P
        traj.append(x.copy())
    return np.array(traj)


def run_trial(truth, rng, p_state, noise, q, q_prop=0.15, q_rho=0.0):
    steps = list(range(len(truth)))
    x_true = np.array([x for _, x in truth])
    dim = len(STATES)

    part_obs, agg_obs = [], []
    for k in steps:
        for j in range(dim):                                   # sparse per-state reports
            if rng.random() < p_state:
                v = x_true[k, j] * (1 + rng.gauss(0, noise))
                H = np.zeros(dim); H[j] = 1.0
                r = max((noise * max(x_true[k, j], 1.0)) ** 2, 1.0)
                part_obs.append((k, H, v, r))
        total = float(x_true[k].sum()) * (1 + rng.gauss(0, noise))   # frequent aggregate
        agg_obs.append((k, np.ones(dim), total, max((noise * max(x_true[k].sum(), 1.0)) ** 2, 1.0)))

    seed_x = x_true[0].copy()
    kw = {"q_prop": q_prop, "q_rho": q_rho}
    parts_only = kalman(part_obs, len(steps), dim, q, seed_x, seed_P=25.0, **kw)
    vector = kalman(part_obs + agg_obs, len(steps), dim, q, seed_x, seed_P=25.0, **kw)

    def nrmse(est):
        rng_ = x_true.max(axis=0) - x_true.min(axis=0)
        rng_[rng_ == 0] = 1.0
        return float(np.mean(np.sqrt(((est - x_true) ** 2).mean(axis=0)) / rng_))

    return nrmse(parts_only), nrmse(vector), len(part_obs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", default="datasets/helene2024/ground_truth.json")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--q", type=float, default=9.0)
    ap.add_argument("--q-prop", type=float, default=0.15,
                    help="process noise proportional to |x|. DEFAULT, and a precondition "
                         "rather than a tuning knob: Virginia ranges 1->2 while North "
                         "Carolina ranges 6->123, so equal ABSOLUTE accrual noise fits "
                         "badly and makes the vector arm look 7.7x worse than it is. "
                         "Pass 0 for isotropic (sigma^2 I) only to reproduce that.")
    ap.add_argument("--q-rho", type=float, default=0.0,
                    help="uniform correlation in Q, marginals preserved. Tests whether "
                         "the aggregate row needs off-diagonal process noise. MEASURED "
                         "NEGATIVE: monotonically worse in rho, and it degrades parts-only "
                         "too, so these trajectories really are near-independent.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1,
                    help="Independent RNG streams (seed, seed+1, ...). With 1 the output "
                         "is the historical single-stream table. With N>1 each density "
                         "reports the mean delta and its SPREAD across streams -- the "
                         "noise floor. Re-running one seed reproduces, it does not "
                         "measure variance, and a delta smaller than the floor is not a "
                         "result.")
    args = ap.parse_args()

    truth = truth_series(Path(args.truth))
    print(f"[truth] {len(truth)} snapshots, {len(STATES)} states, REAL trajectories "
          f"(North Carolina revises 123 -> 102 -> 96)")
    print(f"[sim]   reporting process only: totals every snapshot, per-state sparse, "
          f"{args.noise:.0%} noise\n")
    print(f"{'per-state report rate':>22} {'parts-only':>12} {'vector':>10} {'delta':>9} "
          f"{'vector wins':>12}")

    for p_state in (0.10, 0.20, 0.35, 0.50, 0.80):
        per_seed = []
        for sd in range(args.seed, args.seed + args.seeds):
            rng = random.Random(sd)
            a, b, wins = [], [], 0
            for _ in range(args.trials):
                pa, vb, n = run_trial(truth, rng, p_state, args.noise, args.q,
                                      args.q_prop, args.q_rho)
                a.append(pa); b.append(vb); wins += vb < pa
            per_seed.append((sum(a) / len(a), sum(b) / len(b), wins))
        ma = sum(x[0] for x in per_seed) / len(per_seed)
        mb = sum(x[1] for x in per_seed) / len(per_seed)
        wins = sum(x[2] for x in per_seed)
        tot = args.trials * args.seeds
        if args.seeds == 1:
            print(f"{p_state:>21.0%} {ma:>12.4f} {mb:>10.4f} {mb - ma:>+9.4f} "
                  f"{wins}/{tot:<11}")
        else:
            deltas = [x[1] - x[0] for x in per_seed]
            spread = max(deltas) - min(deltas)
            print(f"{p_state:>21.0%} {ma:>12.4f} {mb:>10.4f} {mb - ma:>+9.4f} "
                  f"{wins}/{tot:<11} spread={spread:.4f} "
                  f"[{min(deltas):+.4f},{max(deltas):+.4f}]"
                  f"{'  CLEARS' if abs(mb - ma) > spread else '  WITHIN FLOOR'}")

    print("\nThe aggregate cannot disaggregate on its own -- it constrains the SUM, not the\n"
          "split -- so the gain should be largest where per-state reports are sparse and\n"
          "shrink as they become dense enough to pin each state directly.")


if __name__ == "__main__":
    main()
