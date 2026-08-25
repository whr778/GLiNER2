"""Does a fourth HMM state for downward reclassification earn its place?

Helene's North Carolina truth falls four times (25, 14, 21, 7 deaths -- the largest 21%
of the running value) and the national total three times. Every other mechanism here
treats a falling toll as an error. This asks whether modelling it explicitly helps, or
whether the `own` state already absorbs drops that small.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "datasets/disaster_streams")
import evaluate as ekf                                             # noqa: E402
from scope_gate import gate, hmm_gate, hmm_gate4                   # noqa: E402
from hmm_collapse_test import build, weights                       # noqa: E402
from scope_gate_test import score, pooled_rmse, oracle_gate_three_way, DATASETS  # noqa: E402


def main():
    cfg, states, res, series, obs = build()
    grid = res["grid"]
    w = weights(obs, "flat")
    for o in obs:
        o["_reject_logodds"] = sum(w[f] for f in w if o[f])
    P = lambda kept: pooled_rmse(score(kept, series, grid, states), cfg["places"])
    NC = lambda kept: (score(kept, series, grid, states).get("North Carolina") or (None,))[0]

    orc = P(oracle_gate_three_way(obs, states, series, 0.25))
    base = P(gate(obs, 2.0, 2, states, cfg["reference"])[0])
    k3 = hmm_gate(obs, states, cfg["reference"], sigma=0.3, reject_cost=4.0,
                  stay=0.1, part_ratio=2.0)[0]
    print(f"helene: shipped gate {base:.1f} | 3-state HMM {P(k3):.1f} | oracle {orc:.1f}")
    print(f"        North Carolina nRMSE: gate "
          f"{NC(gate(obs, 2.0, 2, states, cfg['reference'])[0]):.3f}  3-state {NC(k3):.3f}\n")
    print(f"  {'revise_cost':>12}{'REV fires':>11}{'kept':>7}{'drop':>6} || "
          f"{'POOLED':>8}{'NC nRMSE':>10}")
    for rc in (5.0, 6.0, 8.0, 12.0, 20.0):
        kept, moved, dropped = hmm_gate4(obs, states, cfg["reference"], sigma=0.3,
                                         reject_cost=4.0, stay=0.1, part_ratio=2.0,
                                         revise_cost=rc)
        nrev = sum(1 for v in kept.values() for o in v if o.get("_revised"))
        nk = sum(len(v) for k, v in kept.items() if k != "__aggregate__")
        nc = NC(kept)
        print(f"  {rc:>12.1f}{nrev:>11}{nk:>7}{len(dropped):>6} || {P(kept):>8.1f}"
              f"{(nc if nc is not None else float('nan')):>10.3f}")
    print(f"\n  3-state reference for the same columns: "
          f"pooled {P(k3):.1f}, NC {NC(k3):.3f}")


if __name__ == "__main__":
    main()
