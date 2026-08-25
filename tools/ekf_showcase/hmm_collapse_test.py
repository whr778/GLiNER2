"""Can ONE decode absorb the date gate [5], the scope gate [6] and page furniture?

Today those are three independent hard filters applied in series, and the series is a bad
trade: measured, the date gate adds +1 cross-event catch for +10 false rejections over
scope membership alone, because a hard reject cannot be outvoted by strong magnitude
evidence. This folds all three into the item-2 Viterbi as additive REJECT evidence, so
they argue instead of veto, and decodes every stream so scope membership can act inside
the emission rather than before streams exist.

Weights are log-likelihood ratios from the 86 hand-audited labels. THAT IS FITTING ON THE
EVALUATION SET -- there is no held-out split and only six positives -- so the cross-event
catch/FP numbers below are optimistic by construction. The `flat` arm uses one round
hand-set weight for all three features and peeks at nothing; treat it as the honest one.
Pooled RMSE is scored against trajectory truth, not the audit labels, so it is unaffected.
"""
import json, math, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import out_of_window                             # noqa: E402
from event_binding_probe import _ctx_key, window, load_labels      # noqa: E402
from scope_gate import gate, viterbi_gate, hmm_gate                # noqa: E402
from scope_gate_test import (DATASETS, truth, score, pooled_rmse,  # noqa: E402
                             oracle_gate_three_way)

BOILER = re.compile(r"RELATED COVERAGE|MORE COVERAGE|READ MORE|SEE ALSO|Related:", re.I)


def build():
    cfg = DATASETS["helene"]
    states = {cfg["key_of"](p): p for p in cfg["places"]}
    roll = json.load(open("datasets/helene2024/rollup.json", encoding="utf-8"))
    IN = set(roll["aliases"]) | set(roll["aliases"].values())
    h = roll["hierarchy"]
    IN |= {h.get("aggregate")} | set(h.get("parts") or [])
    IN = {str(x).lower() for x in IN if x}
    labels = load_labels()
    res = json.loads(Path(cfg["tracked"]).read_text(encoding="utf-8"))
    obs = []
    for a in res["articles"]:
        for o in a["observations"]:
            if o["mode"] != "heuristic" or o["role"] != "dead":
                continue
            ctx, off = window(a["text"], o.get("span") or "")
            lab = None
            if ctx is not None:
                lb = labels.get(_ctx_key(ctx, int(o["value"])))
                lab = (lb or {}).get("label") if isinstance(lb, dict) else lb
            o = dict(o)
            o["_f_scope"] = str(o.get("event_key", "")).lower() not in IN
            o["_f_date"] = bool(out_of_window(a["text"], o.get("span") or "",
                                              a.get("events") or {}, 2024))
            o["_f_boiler"] = bool(ctx is not None and BOILER.search(ctx[:off]))
            o["_label"] = lab or "unlabelled"
            obs.append(o)
    return cfg, states, res, truth(Path(cfg["truth"]), cfg["onset"]), obs


def weights(obs, mode):
    """log P(feature|cross-event) / P(feature|genuine), or a flat hand-set value."""
    if mode == "flat":
        return {"_f_scope": 2.0, "_f_date": 2.0, "_f_boiler": 2.0}
    cross = [o for o in obs if o["_label"] == "cross-event"]
    good = [o for o in obs if o["_label"] == "helene"]
    w = {}
    for f in ("_f_scope", "_f_date", "_f_boiler"):
        pc = sum(o[f] for o in cross) / max(len(cross), 1)
        pg = sum(o[f] for o in good) / max(len(good), 1)
        w[f] = math.log(max(pc, 1e-3) / max(pg, 1e-3))
    return w


def main():
    cfg, states, res, series, obs = build()
    grid = res["grid"]
    P = lambda kept: pooled_rmse(score(kept, series, grid, states), cfg["places"])
    orc = P(oracle_gate_three_way(obs, states, series, 0.25))
    base = P(gate(obs, 2.0, 2, states, cfg["reference"])[0])
    vit = P(viterbi_gate(obs, states, cfg["reference"], sigma=0.3, reject_cost=4.0,
                         stay=0.1, warmup=0, part_ratio=2.0)[0])
    n_cross = sum(1 for o in obs if o["_label"] == "cross-event")
    n_good = sum(1 for o in obs if o["_label"] == "helene")
    print(f"helene: {len(obs)} obs, {n_good} genuine, {n_cross} cross-event")
    print(f"  shipped gate {base:.1f} | viterbi (magnitude only) {vit:.1f} | oracle {orc:.1f}")
    print(f"  series-of-hard-gates reference: 5/{n_cross} caught, 16/{n_good} false = 19.8%\n")
    print(f"  {'arm':16}{'scope':>7}{'date':>7}{'boil':>7} | {'caught':>7}{'falseRej':>9}"
          f" || {'POOLED':>8}")
    for mode in ("flat", "measured"):
        w = weights(obs, mode)
        for o in obs:
            o["_reject_logodds"] = sum(w[f] for f in w if o[f])
        kept, moved, dropped = hmm_gate(obs, states, cfg["reference"], sigma=0.3,
                                        reject_cost=4.0, stay=0.1, part_ratio=2.0)
        drop_ids = {id(d) for d in dropped}
        caught = sum(1 for o in obs if o["_label"] == "cross-event"
                     and any(id(d) == id(o) or (d.get("t_hours") == o["t_hours"]
                             and d["value"] == o["value"]) for d in dropped))
        false_rej = sum(1 for o in obs if o["_label"] == "helene"
                        and any(d.get("t_hours") == o["t_hours"]
                                and d["value"] == o["value"] for d in dropped))
        pool = P(kept)
        print(f"  {mode:16}{w['_f_scope']:>7.2f}{w['_f_date']:>7.2f}{w['_f_boiler']:>7.2f}"
              f" | {caught:>4}/{n_cross:<2}{false_rej:>6}/{n_good:<2}"
              f" || {pool:>8.1f}")


if __name__ == "__main__":
    main()
