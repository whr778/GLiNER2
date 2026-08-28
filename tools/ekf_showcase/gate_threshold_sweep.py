"""Which stage-0 gate, at which threshold, minimises tracking error on the real events?

Every gate number this project has recorded was taken at --gate-threshold 0.5, and the
gate benchmark that ranks the models scores SMS messages with an indicative label. This
scores the thing that actually matters: pooled RMSE in deaths against trajectory truth,
on the three real events, with the shipped association decode.

Why one pipeline run per (event, gate) suffices. The gate's only effect is which articles
reach extraction, and raising a threshold only ever REMOVES articles. So a run at 0.5
contains every article that gate could admit at any threshold >= 0.5, and the sweep is
pure filtering -- no re-extraction, no re-run.

Scoring reuses the shipped harness (scope_gate_test.score / pooled_rmse) and the shipped
association gate (scope_gate.hmm_gate at its defaults), so the only thing varying is the
stage-0 decision.

READ THE PAIR: error beside streams-scored. pooled_rmse only counts streams that HAVE
observations, so raising the threshold can lower the error purely by scoring fewer streams.
On Helene the mmBERT gate goes 17.5 -> 9.6 between thresholds 0.9 and 0.998, and coverage
goes 4/6 streams -> 2/6 in the same step: most of that "gain" is the metric measuring less.

WHAT THIS DOES NOT COMPARE TO. The absolute numbers here do not reproduce the published
Helene/Turkiye/Aegean figures, because those were scored on FROZEN observation sets built
with a different pipeline configuration (see DATASETS in scope_gate_test.py). Comparisons
BETWEEN gates within one run of this script are valid -- same feed, same config, only the
gate varies. Comparisons against the published numbers are not.

    uv run python tools/ekf_showcase/gate_threshold_sweep.py <dir-of-run-files>
"""
import json, sys
from math import isnan
from pathlib import Path

sys.path.insert(0, "tools/ekf_showcase")
from scope_gate import hmm_gate
from scope_gate_test import DATASETS, pooled_rmse, score, truth

SP = Path(sys.argv[1])
THRESHOLDS = (0.5, 0.9, 0.99, 0.998, 0.9999)
GATES = {"casualty-docee": "casualty-docee", "gate2-mmbert-v2": "gate2-mmbert-v2"}
EVENTS = {"helene": "helene", "turkey": "turkey", "aegean": "aegean"}

# Streams scored is reported beside the error, because pooled_rmse only counts streams
# that HAVE observations: a higher threshold can look better purely by scoring fewer of
# them. Read the pair, never the error alone.
print(f"{'event':9s}{'gate':18s}{'thresh':>8s}{'arts':>6s}{'obs':>6s}"
      f"{'streams':>9s}{'POOLED RMSE (deaths)':>22s}")
for ev_tag, ds in EVENTS.items():
    cfg = DATASETS[ds]
    series = truth(Path(cfg["truth"]), cfg["onset"])
    states = {cfg["key_of"](p): p for p in cfg["places"]}
    for gate_tag in GATES:
        path = SP / f"sweep_{ev_tag}_{gate_tag}.jsonl"
        if not path.is_file():
            print(f"{ev_tag:9s}{gate_tag:18s}  (no run file)")
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        grid = res["grid"]
        for t in THRESHOLDS:
            arts = [a for a in res["articles"]
                    if a.get("relevant") and a.get("relevance_confidence", 0.0) >= t]
            obs = [o for a in arts for o in a["observations"]
                   if o["mode"] == "heuristic" and o["role"] == "dead"]
            if not obs:
                print(f"{ev_tag:9s}{gate_tag:18s}{t:>8.4f}{len(arts):>6d}{0:>6d}"
                      f"{'':>9s}{'no observations':>22s}")
                continue
            kept, _, _ = hmm_gate(obs, states, cfg["reference"])
            sc = score(kept, series, grid, states)
            n_str = sum(1 for pl in cfg["places"] if len(sc.get(pl, ())) > 4)
            p = pooled_rmse(sc, cfg["places"])
            cell = "nan" if isnan(p) else f"{p:.1f}"
            print(f"{ev_tag:9s}{gate_tag:18s}{t:>8.4f}{len(arts):>6d}{len(obs):>6d}"
                  f"{n_str:>4d}/{len(cfg['places']):<4d}{cell:>22s}")
