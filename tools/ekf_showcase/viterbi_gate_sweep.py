"""Does a global Viterbi scope decode beat the greedy gate?

The shipped gate walks each stream in time order and commits per observation. Viterbi
decides the sequence jointly, which is the textbook fix for the failure we reproduced on
Turkiye (one large figure admitted early poisons the running maximum, and every
legitimate later reading then looks stale).

Scored against each event's BEST shipped setting, not the shipped default -- comparing an
arm against a mistuned control is how you manufacture a win.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scope_gate import gate, viterbi_gate                          # noqa: E402
from scope_gate_test import (DATASETS, truth, score, pooled_rmse,  # noqa: E402
                             oracle_gate_three_way)

BEST_RATIO = {"helene": 2.0, "turkey": 1.5}


def load(ds):
    cfg = DATASETS[ds]
    states = {cfg["key_of"](p): p for p in cfg["places"]}
    res = json.loads(Path(cfg["tracked"]).read_text(encoding="utf-8"))
    series = truth(Path(cfg["truth"]), cfg["onset"])
    obs = [o for a in res["articles"] for o in a["observations"]
           if o["mode"] == "heuristic" and o["role"] == "dead"]
    return cfg, states, res, series, obs


def main():
    for ds in ("helene", "turkey"):
        cfg, states, res, series, obs = load(ds)
        grid = res["grid"]
        P = lambda kept: pooled_rmse(score(kept, series, grid, states), cfg["places"])
        orc = P(oracle_gate_three_way(obs, states, series, 0.25))
        base = P(gate(obs, BEST_RATIO[ds], 2, states, cfg["reference"])[0])
        print(f"\n=== {ds} ===  shipped gate @{BEST_RATIO[ds]} = {base:.1f}"
              f"   oracle = {orc:.1f}")
        print(f"  {'sigma':>6}{'reject':>8}{'stay':>6} | {'kept':>5}{'moved':>6}{'drop':>6}"
              f" || {'POOLED':>9}")
        for sigma in (0.3, 0.5, 0.8):
            for rc in (1.0, 2.0, 4.0, 8.0):
                kept, moved, dropped = viterbi_gate(
                    obs, states, cfg["reference"], sigma=sigma, reject_cost=rc,
                    stay=0.1, warmup=0, part_ratio=BEST_RATIO[ds])
                n_kept = sum(len(v) for k, v in kept.items() if k != "__aggregate__")
                pool = P(kept)
                mark = "  <-- beats gate" if pool < base else ""
                print(f"  {sigma:>6.1f}{rc:>8.1f}{0.1:>6.1f} | {n_kept:>5}{len(moved):>6}"
                      f"{len(dropped):>6} || {pool:>9.1f}{mark}")


if __name__ == "__main__":
    main()
