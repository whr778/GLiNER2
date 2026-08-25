"""Does the emission collapse hold on Turkiye, or is it a Helene-only result?

No audit labels exist for this event, so cross-event catch/FP cannot be measured here.
What CAN be measured is whether folding the date, scope and boilerplate features into the
emission costs anything on the trajectory -- which is the question that decides whether
the collapse is safe to ship on both events or only one.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, "tools/ekf_showcase")
from run_pipeline import out_of_window
from scope_gate import gate, viterbi_gate, hmm_gate
from scope_gate_test import DATASETS, truth, score, pooled_rmse, oracle_gate_three_way

BOILER = re.compile(r"RELATED COVERAGE|MORE COVERAGE|READ MORE|SEE ALSO|Related:", re.I)
cfg = DATASETS["turkey"]
states = {cfg["key_of"](p): p for p in cfg["places"]}
roll = json.load(open("datasets/turkey2023/rollup.json", encoding="utf-8"))
IN = set(roll["aliases"]) | set(roll["aliases"].values())
h = roll["hierarchy"]
IN |= {h.get("aggregate")} | set(h.get("parts") or [])
IN = {str(x).lower() for x in IN if x}

res = json.loads(Path(cfg["tracked"]).read_text(encoding="utf-8"))
series = truth(Path(cfg["truth"]), cfg["onset"])
grid = res["grid"]
obs = []
for a in res["articles"]:
    for o in a["observations"]:
        if o["mode"] != "heuristic" or o["role"] != "dead":
            continue
        o = dict(o)
        span = o.get("span") or ""
        key = str(o.get("event_key", "")).lower()
        key_place = key.split("|")[-1]
        i = a["text"].find(span)
        o["_f_scope"] = key_place not in IN
        o["_f_date"] = bool(out_of_window(a["text"], span, a.get("events") or {}, 2023))
        o["_f_boiler"] = bool(i >= 0 and BOILER.search(a["text"][max(0, i - 200):i]))
        obs.append(o)

n = len(obs)
print(f"turkey: {n} 'dead' observations")
for f in ("_f_scope", "_f_date", "_f_boiler"):
    print(f"  {f:10} fires on {sum(o[f] for o in obs):>3}/{n}")

P = lambda kept: pooled_rmse(score(kept, series, grid, states), cfg["places"])
orc = P(oracle_gate_three_way(obs, states, series, 0.25))
base = P(gate(obs, 1.5, 2, states, cfg["reference"])[0])
vit = P(viterbi_gate(obs, states, cfg["reference"], sigma=0.3, reject_cost=4.0,
                     stay=0.1, warmup=0, part_ratio=1.5)[0])
print(f"\n  shipped gate @1.5   {base:9.1f}")
print(f"  viterbi (magnitude) {vit:9.1f}")
print(f"  oracle              {orc:9.1f}")
print(f"\n  {'flat weight':>12} | {'POOLED':>9}")
for w in (0.0, 1.0, 2.0, 4.0):
    for o in obs:
        o["_reject_logodds"] = w * sum(1 for f in ("_f_scope", "_f_date", "_f_boiler") if o[f])
    kept, _, dropped = hmm_gate(obs, states, cfg["reference"], sigma=0.3,
                                reject_cost=4.0, stay=0.1, part_ratio=1.5)
    print(f"  {w:>12.1f} | {P(kept):>9.1f}   ({len(dropped)} dropped)")
