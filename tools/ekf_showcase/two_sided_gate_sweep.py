"""Sweep the new downward (stale) side of the scope gate on both real events."""
import json, sys
from pathlib import Path
sys.path.insert(0, "tools/ekf_showcase")
from scope_gate import gate
from scope_gate_test import DATASETS, truth, score, pooled_rmse, oracle_gate_three_way

for ds in ("helene", "turkey"):
    cfg = DATASETS[ds]
    states = {cfg["key_of"](p): p for p in cfg["places"]}
    res = json.loads(Path(cfg["tracked"]).read_text(encoding="utf-8"))
    series = truth(Path(cfg["truth"]), cfg["onset"])
    grid = res["grid"]
    obs = [o for a in res["articles"] for o in a["observations"]
           if o["mode"] == "heuristic" and o["role"] == "dead"]
    orc = pooled_rmse(score(oracle_gate_three_way(obs, states, series, 0.25),
                            series, grid, states), cfg["places"])
    print(f"\n=== {ds}: {len(obs)} 'dead' observations "
          f"| three-way oracle (tol 0.25) = {orc:.1f} deaths ===")
    print(f"{'ratio':>6}{'down':>7}{'moved':>7}{'drop':>6}{'stale':>7} || {'POOLED':>8}")
    for ratio in (2.0,):
        for down in (0.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5):
            kept, moved, dropped = gate(obs, ratio, 2, states,
                                        cfg["reference"], down_ratio=down)
            stale = sum(1 for d in dropped if d.get("_stale"))
            pool = pooled_rmse(score(kept, series, grid, states), cfg["places"])
            flag = "  <- one-sided baseline" if down == 0.0 else ""
            print(f"{ratio:>6.1f}{down:>7.1f}{len(moved):>7}{len(dropped):>6}"
                  f"{stale:>7} || {pool:>8.1f}{flag}")
