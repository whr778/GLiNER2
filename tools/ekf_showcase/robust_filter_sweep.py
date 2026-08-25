"""Does a Student-t measurement model beat the hard gates it is meant to retire?

REJECT_SIGMA, MAX_RATE and the scope gate's drop branch are three thresholds standing in
for one fat tail. This sweeps the Student-t degrees of freedom on both real events, with
the scope gate ON and OFF, so the two questions are separable:

  * does the robust filter help at the shipped operating point?
  * does it let the scope gate be turned DOWN or off -- i.e. does it retire a knob?

Pooled micro-RMSE in deaths is the headline, as everywhere else in this programme.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "datasets/disaster_streams")
import evaluate as ekf                                        # noqa: E402
from scope_gate import gate                                   # noqa: E402
from scope_gate_test import (DATASETS, truth, score, pooled_rmse,  # noqa: E402
                             oracle_gate_three_way)

NUS = (None, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0)


def run(ds, ratio, nu, symmetric=False):
    cfg = DATASETS[ds]
    states = {cfg["key_of"](p): p for p in cfg["places"]}
    res = json.loads(Path(cfg["tracked"]).read_text(encoding="utf-8"))
    series = truth(Path(cfg["truth"]), cfg["onset"])
    obs = [o for a in res["articles"] for o in a["observations"]
           if o["mode"] == "heuristic" and o["role"] == "dead"]
    ekf.STUDENT_T_NU = nu
    ekf.STUDENT_T_SYMMETRIC = symmetric
    kept, _, _ = gate(obs, ratio, 2, states, cfg["reference"])
    out = pooled_rmse(score(kept, series, res["grid"], states), cfg["places"])
    ekf.STUDENT_T_NU = None
    ekf.STUDENT_T_SYMMETRIC = False
    return out


def main():
    for ds in ("helene", "turkey"):
        cfg = DATASETS[ds]
        states = {cfg["key_of"](p): p for p in cfg["places"]}
        res = json.loads(Path(cfg["tracked"]).read_text(encoding="utf-8"))
        series = truth(Path(cfg["truth"]), cfg["onset"])
        obs = [o for a in res["articles"] for o in a["observations"]
               if o["mode"] == "heuristic" and o["role"] == "dead"]
        orc = pooled_rmse(score(oracle_gate_three_way(obs, states, series, 0.25),
                                series, res["grid"], states), cfg["places"])
        print(f"\n=== {ds} === three-way oracle ceiling {orc:.1f} deaths")
        print(f"  {'nu':>6} | {'gate 2.0':>10} {'gate OFF':>10} | {'sym 2.0':>10}")
        for nu in NUS:
            g_on = run(ds, 2.0, nu)
            g_off = run(ds, 0.0, nu)
            sym = run(ds, 2.0, nu, symmetric=True) if nu else g_on
            tag = "gaussian" if nu is None else f"{nu:.0f}"
            print(f"  {tag:>6} | {g_on:>10.1f} {g_off:>10.1f} | {sym:>10.1f}")


if __name__ == "__main__":
    main()
