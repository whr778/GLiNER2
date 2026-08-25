"""Can a LEARNED reject beat the ratio gate? Measure separability before building.

The three-way oracle takes Helene from the shipped gate's 29.3 pooled deaths to 17.6 by
DROPPING more (106 kept -> 76). That ~11.7-death prize is the whole remaining association
headroom (assignment headroom is zero: two-way oracle 29.3 == shipped gate 29.3).

But the oracle drops on RELATIVE ERROR AGAINST TRUTH. The question a learned reject has to
answer is whether the features available at INFERENCE separate those same observations --
and specifically whether text/metadata add anything BEYOND the magnitude ratio the shipped
gate already uses. If they do not, there is nothing to learn.
"""
import json, sys, math, collections
from pathlib import Path
sys.path.insert(0, "tools/ekf_showcase")
from scope_gate import reference_for, scale_at
from scope_gate_test import oracle_gate_three_way, DATASETS, truth, at

cfg = DATASETS["helene"]
states = {cfg["key_of"](p): p for p in cfg["places"]}
res = json.loads(Path(cfg["tracked"]).read_text(encoding="utf-8"))
series = truth(Path(cfg["truth"]), cfg["onset"])
obs = [o for a in res["articles"] for o in a["observations"]
       if o["mode"] == "heuristic" and o["role"] == "dead"]

TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
# The oracle keeps every non-gated observation by construction, so a "gated" feature
# separates perfectly for free. Restrict to the gated population -- that is the only
# place the reject decision is actually made.
obs = [o for o in obs if str(o.get("event_key")) in states]
kept = oracle_gate_three_way(obs, states, series, TOL)
keep_ids = {id(o) for v in kept.values() for o in v}
y = [0 if id(o) in keep_ids else 1 for o in obs]          # 1 = oracle REJECTS
print(f"oracle tol={TOL}: {len(obs)} observations, {sum(y)} rejected ({sum(y)/len(y):.0%})")

# the feature the shipped gate already has: value / running reference
scale_cache = {}
def ratio_feat(o):
    key = str(o.get("event_key"))
    if key not in states:
        return 0.0
    if key not in scale_cache:
        scale_cache[key] = reference_for(obs, key, states, cfg["reference"])
    natl = scale_at(scale_cache[key], o["t_hours"])
    return float(o["value"]) / max(natl, 1.0)

rows = []
for o in obs:
    span = (o.get("span") or "").lower()
    rows.append({
        "ratio": ratio_feat(o),
        "logv": math.log10(max(float(o["value"]), 1)),
        "conf": float(o.get("confidence") or 0),
        "official": 1.0 if o.get("source") == "official" else 0.0,
        "atleast": 1.0 if o.get("qualifier") == "at_least" else 0.0,
        "t": o["t_hours"] / 1000.0,
        "span_num": 1.0 if any(c.isdigit() for c in span) else 0.0,
        "span_len": len(span) / 40.0,
    })

def auc(scores, labels):
    pairs = [(s, l) for s, l in zip(scores, labels)]
    pos = [s for s, l in pairs if l == 1]; neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))

print("\n(restricted to GATED observations -- non-gated are kept by construction)")
print("\nsingle-feature AUC for predicting the oracle's REJECT (0.5 = no signal):")
for f in rows[0]:
    a = auc([r[f] for r in rows], y)
    print(f"  {f:10} {a:5.3f}   {'<- gate already uses this' if f=='ratio' else ''}")

# does anything add to `ratio`? residualise: within ratio-matched bands, does the
# feature still separate?
print("\nwithin ratio-matched bands (does any feature add to what the gate has?):")
band = lambda r: min(int(r["ratio"] * 4), 6)
by = collections.defaultdict(list)
for r, l in zip(rows, y):
    by[band(r)].append((r, l))
print("  (AUC below 0.5 means the feature predicts KEEP; 1-AUC is its strength)")
for f in ("official", "atleast", "conf", "logv", "span_num"):
    tot = n = 0
    for b, items in by.items():
        if len({l for _, l in items}) < 2:
            continue
        a = auc([r[f] for r, _ in items], [l for _, l in items])
        if a == a:
            tot += a * len(items); n += len(items)
    print(f"  {f:10} pooled within-band AUC {tot/max(n,1):5.3f}  (n={n})")


print("\n--- what the oracle actually rejects ---")
rej=[o for o,l in zip(obs,y) if l==1]; kep=[o for o,l in zip(obs,y) if l==0]
import statistics
for name,grp in (("REJECTED",rej),("KEPT",kep)):
    v=[float(o["value"]) for o in grp]
    dig=sum(1 for o in grp if any(c.isdigit() for c in (o.get("span") or "")))
    print(f"  {name:9} n={len(grp):3}  median value {statistics.median(v):7.1f}  "
          f"digit-span {dig/max(len(grp),1):5.0%}")
print("\n  sample rejects:")
for o in sorted(rej,key=lambda o:float(o["value"]))[:10]:
    print(f"    v={o['value']:>6}  t={o['t_hours']:8.1f}h  {o['event_key']:16} "
          f"{o.get('source','?'):14} {(o.get('span') or '')[:34]!r}")
