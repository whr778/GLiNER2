"""The pre-registered third-event test: does the collapse pay where scale separation is 143x?

Predictions registered in THIRD_EVENT_AEGEAN2020.md before the feed existed:
  1. the collapse shows a LARGE gain here, unlike Turkiye's zero
  2. the rejection is carried by the DATE feature, not scope membership
  3. sparsity does not reverse it, because a 143x contaminant is never accidentally useful
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import out_of_window                              # noqa: E402
from scope_gate import gate, viterbi_gate, hmm_gate                 # noqa: E402
from scope_gate_test import (DATASETS, truth, score, pooled_rmse,   # noqa: E402
                             oracle_gate_three_way)

cfg = DATASETS["aegean"]
states = {cfg["key_of"](p): p for p in cfg["places"]}
roll = json.load(open("datasets/aegean2020/rollup.json", encoding="utf-8"))
IN = {str(x).lower() for x in set(roll["aliases"]) | set(roll["aliases"].values()) if x}
IN |= {"__aggregate__", "izmir", "samos"}

res = json.loads(Path(cfg["tracked"]).read_text(encoding="utf-8"))
series = truth(Path(cfg["truth"]), cfg["onset"])
grid = res["grid"]

obs = []
for a in res["articles"]:
    for o in a["observations"]:
        if o["role"] != "dead":
            continue
        o = dict(o)
        sp = o.get("span") or ""
        key = str(o.get("event_key", "")).lower()
        o["_f_scope"] = key not in IN
        # mode="any": nearest-by-distance picks the CURRENT year here (2020 at +117 chars
        # beats 1999 at -152) and misses a 143x contaminant entirely.
        o["_f_date"] = bool(out_of_window(a["text"], sp, a.get("events") or {}, 2020,
                                          mode="any"))
        obs.append(o)

P = lambda kept: pooled_rmse(score(kept, series, grid, states), cfg["places"])
big = [o for o in obs if float(o["value"]) > 500]
print(f"aegean: {len(obs)} dead observations, true peak 117")
print(f"  contaminants >500: {sorted(int(o['value']) for o in big)}")
print(f"  bound to a gated place: {sum(1 for o in obs if str(o.get('event_key','')).lower() in states)}"
      f" of {len(obs)}   (the rest are 'unknown')")
print(f"  _f_date fires on {sum(o['_f_date'] for o in obs)}, _f_scope on {sum(o['_f_scope'] for o in obs)}")

orc = P(oracle_gate_three_way(obs, states, series, 0.25))
base = P(gate(obs, 2.0, 2, states, cfg["reference"])[0])
vit = P(viterbi_gate(obs, states, cfg["reference"], sigma=0.3, reject_cost=4.0,
                     stay=0.1, warmup=0, part_ratio=2.0)[0])
print(f"\n  shipped gate @2.0   {base:9.2f}")
print(f"  viterbi (magnitude) {vit:9.2f}")
print(f"  oracle              {orc:9.2f}")
print(f"\n  {'feature weight':>15}{'dropped':>9}{'17000 gone':>12} || {'POOLED':>9}")
for w in (0.0, 1.0, 2.0, 3.0):
    for o in obs:
        o["_reject_logodds"] = w * ((o["_f_date"]) + (o["_f_scope"]))
    kept, moved, dropped = hmm_gate(obs, states, cfg["reference"], sigma=0.3,
                                    reject_cost=4.0, stay=0.1, part_ratio=2.0)
    n17 = sum(1 for d in dropped if float(d["value"]) > 500)
    print(f"  {w:>15.1f}{len(dropped):>9}{n17:>12} || {P(kept):>9.2f}")
