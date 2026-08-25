"""Does soft (PDA) association beat hard assignment -- greedy OR global?

Three arms on the same feed and the same scorer:
  gate        greedy, hard, commits per observation   (shipped)
  viterbi     global, hard, commits per sequence      (item 2, 2-for-2)
  imm         soft, never commits                     (this)

The soft weight reaches the estimate through CONF_R, which the filter already implements
as R /= confidence**2 -- so a weight of 0.1 inflates that reading's noise 100x. CONF_R is
enabled ONLY for the imm arm and restored afterwards, so the controls are untouched.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "datasets/disaster_streams")
import evaluate as ekf                                          # noqa: E402
from scope_gate import gate, viterbi_gate, imm_gate             # noqa: E402
from scope_gate_test import (DATASETS, truth, score, pooled_rmse,  # noqa: E402
                             oracle_gate_three_way)

BEST_RATIO = {"helene": 2.0, "turkey": 1.5}


def main():
    for ds in ("helene", "turkey"):
        cfg = DATASETS[ds]
        states = {cfg["key_of"](p): p for p in cfg["places"]}
        res = json.loads(Path(cfg["tracked"]).read_text(encoding="utf-8"))
        series = truth(Path(cfg["truth"]), cfg["onset"])
        grid = res["grid"]
        obs = [o for a in res["articles"] for o in a["observations"]
               if o["mode"] == "heuristic" and o["role"] == "dead"]
        P = lambda kept: pooled_rmse(score(kept, series, grid, states), cfg["places"])

        orc = P(oracle_gate_three_way(obs, states, series, 0.25))
        base = P(gate(obs, BEST_RATIO[ds], 2, states, cfg["reference"])[0])
        vit = P(viterbi_gate(obs, states, cfg["reference"], sigma=0.3, reject_cost=4.0,
                             stay=0.1, warmup=0, part_ratio=BEST_RATIO[ds])[0])
        print(f"\n=== {ds} ===  gate {base:.1f}   viterbi {vit:.1f}   oracle {orc:.1f}")
        print(f"  {'down':>6}{'clutter':>9} | {'kept':>5}{'moved':>6}{'drop':>6}"
              f" || {'POOLED':>9}")
        for down in (1.0, 0.5, 0.25):
            for clut in (1.0, 1e2, 1e4, 1e6):
                ekf.CONF_R = True
                try:
                    kept, moved, dropped = imm_gate(
                        obs, states, cfg["reference"], down_factor=down,
                        clutter_scale=clut)
                    pool = P(kept)
                finally:
                    ekf.CONF_R = False
                n_kept = sum(len(v) for k, v in kept.items() if k != "__aggregate__")
                flags = []
                if pool < base:
                    flags.append("beats gate")
                if pool < vit:
                    flags.append("beats viterbi")
                print(f"  {down:>6.2f}{clut:>9.0e} | {n_kept:>5}{len(moved):>6}"
                      f"{len(dropped):>6} || {pool:>9.1f}  {', '.join(flags)}")


if __name__ == "__main__":
    main()
