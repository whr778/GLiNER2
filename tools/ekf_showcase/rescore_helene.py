"""Re-score the recorded Helene run: do the disabled gates reach its contamination?

Offline, from `_cache/tracked_rollup.json` -- no model, no GPU. Companion to
`rescore_recorded_run.py` (Turkiye).

EXPLORATORY, not pre-registered: switches are turned on after seeing the failure.

**The prediction going in was WRONG and is kept as the reason to measure.** Turkiye's
contamination is a stale LOW constant (the 1999 Izmit 17,500, overtaken by the real toll),
which the one-sided rise gate rejects by construction. Helene's is the mirror image -- the
outliers are HIGH (1400, 1400, 3000, 300) against a total reaching ~250 -- so the same gate
ADMITS all of them. Its own docstring says so. The rate filter is the lever here.

After the scope filter the survivors are in-scope high outliers: North Carolina 1400 and
250, Florida 300. Those are exactly the residual EKF_MHT_DESIGN 7.3 describes -- "it misses
cross-event figures whose place is IN scope".
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "datasets/disaster_streams"))
sys.path.insert(0, str(REPO / "tools/ekf_showcase"))
import evaluate as ekf  # noqa: E402
from run_pipeline import scope_filter  # noqa: E402

track = json.loads((REPO / "datasets/helene2024/_cache/tracked_rollup.json").read_text())
gt = json.loads((REPO / "datasets/helene2024/ground_truth.json").read_text())
rollup = json.loads((REPO / "datasets/helene2024/rollup.json").read_text())
grid = track["grid"]
onset = datetime.fromisoformat(gt["onset_utc"].replace("Z", "+00:00"))

STREAMS = ["florida", "georgia", "south carolina", "north carolina", "tennessee",
           "virginia", "__aggregate__"]
TRUTH_KEY = {s: s.title() for s in STREAMS}
TRUTH_KEY["__aggregate__"] = "Total"

t_pts = [(datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00")) - onset).total_seconds() / 3600
         for p in gt["points"]]


def truth_series(name):
    vals = [float(p["deaths"][TRUTH_KEY[name]]) for p in gt["points"]]
    out, prev = [], vals[0]
    for t in grid:
        for a, b in zip(t_pts, vals):
            if a > t:
                break
            prev = b
        out.append(prev)
    return out


obs_all = [o for a in track["articles"] for o in a["observations"] if o["role"] == "dead"]
kept, rejected, _ = scope_filter(obs_all, rollup)
print(f"{len(obs_all)} `dead` observations; scope filter rejects {len(rejected)} "
      f"out-of-scope, keeps {len(kept)}")
survivors = sorted((o["value"] for o in kept), reverse=True)[:6]
print(f"largest surviving values: {survivors}\n")

ARMS = [
    ("default (as recorded)", dict(REJECT_SIGMA=None, MAX_RATE=None)),
    ("innovation gate 3-sigma", dict(REJECT_SIGMA=3.0, MAX_RATE=None)),
    ("rate filter 20/h", dict(REJECT_SIGMA=None, MAX_RATE=20.0)),
    ("rate filter 5/h", dict(REJECT_SIGMA=None, MAX_RATE=5.0)),
    ("rate filter 1/h", dict(REJECT_SIGMA=None, MAX_RATE=1.0)),
    ("gate 3s + rate 5/h", dict(REJECT_SIGMA=3.0, MAX_RATE=5.0)),
]


def rmse(est, tr):
    return math.sqrt(sum((e - t) ** 2 for e, t in zip(est, tr)) / len(tr))


rows = {}
for name, cfg in ARMS:
    for k, v in cfg.items():
        setattr(ekf, k, v)
    tot_abs, tot_n, per = 0.0, 0.0, {}
    for s in STREAMS:
        tr = truth_series(s)
        o = [x for x in kept if x.get("event_key") == s]
        if cfg["MAX_RATE"]:
            o = ekf.rate_filter(sorted(o, key=lambda z: z["t_hours"]), cfg["MAX_RATE"])
        est = ekf.est_ekf(o, grid, "dead") if o else [0.0] * len(grid)
        a = rmse(est, tr)
        rg = max(max(tr) - min(tr), 1.0)
        per[s] = (a, a / rg, est[-1], tr[-1], len(o))
        tot_abs += a
        tot_n += a / rg
    rows[name] = (per, tot_abs / len(STREAMS), tot_n / len(STREAMS))

for k, v in dict(REJECT_SIGMA=None, MAX_RATE=None).items():
    setattr(ekf, k, v)

print(f"{'arm':26}{'mean nRMSE':>12}{'mean RMSE':>11}{'NC final':>10}{'FL final':>10}"
      f"{'total final':>13}")
for name, (per, abs_, n_) in rows.items():
    print(f"{name:26}{n_:>12.3f}{abs_:>11.1f}{per['north carolina'][2]:>10.0f}"
          f"{per['florida'][2]:>10.0f}{per['__aggregate__'][2]:>13.0f}")
p0 = rows["default (as recorded)"][0]
print(f"{'TRUTH':26}{'':>12}{'':>11}{p0['north carolina'][3]:>10.0f}"
      f"{p0['florida'][3]:>10.0f}{p0['__aggregate__'][3]:>13.0f}")

# The mean-nRMSE column above macro-averages streams whose ranges differ ~100x (Virginia
# ~1, North Carolina ~117) -- the same defect that inverted the aggregate verdict. Per
# stream, in deaths, is what a reader should judge on.
print(f"\n{'stream':18}{'truth':>7}{'n_obs':>7}", end="")
for name in ("default (as recorded)", "innovation gate 3-sigma", "rate filter 5/h"):
    print(f"{name.split(' (')[0][:14]:>16}", end="")
print()
for st in STREAMS:
    tr = rows["default (as recorded)"][0][st][3]
    n = rows["default (as recorded)"][0][st][4]
    print(f"{st:18}{tr:>7.0f}{n:>7}", end="")
    for name in ("default (as recorded)", "innovation gate 3-sigma", "rate filter 5/h"):
        a, _, fin, _, _ = rows[name][0][st]
        print(f"{fin:>8.0f}/{a:>7.0f}", end="")
    print()
print("\n  columns are final estimate / RMSE in deaths")
