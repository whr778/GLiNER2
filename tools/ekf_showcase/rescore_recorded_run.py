"""Re-score the recorded Turkiye run with this week's lessons. No model, no GPU.

EXPLORATORY, not a pre-registered result: the gates below are switched on AFTER seeing
the failure they address, which is exactly the post-hoc move that inflated gate 1. Read
these as "is there a lever here", not "the EKF wins".

Three lessons applied:
  1. Report ABSOLUTE deaths, not just nRMSE. RESULTS.md already warns that a wrong
     constant (the 1999 Izmit 17,500) scores well mid-range; absolute error says so.
  2. Report the filter's own sigma, now that it is exposed.
  3. The documented failure is heavy-tailed contamination dragging a smoother down.
     REJECT_SIGMA and MAX_RATE were BUILT for that and default to None.
"""
import json, math, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "datasets/disaster_streams"))
import evaluate as ekf

track = json.loads((REPO / "datasets/turkey2023/_cache/tracked_default.json").read_text())
gt = json.loads((REPO / "datasets/turkey2023/ground_truth.json").read_text())
grid = track["grid"]
obs = [o for a in track["articles"] for o in a["observations"] if o["role"] == "dead"]
obs.sort(key=lambda o: o["t_hours"])

from datetime import datetime
onset = datetime.fromisoformat(gt["onset_utc"].replace("Z", "+00:00"))
pts = [p for p in gt["points"] if p.get("turkey") is not None]
tt = [(datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00")) - onset).total_seconds() / 3600
      for p in pts]
tv = [float(p["turkey"]) for p in pts]
def truth_at(t):
    prev = tv[0]
    for a, b in zip(tt, tv):
        if a > t: return prev
        prev = b
    return prev
truth = [truth_at(t) for t in grid]
rng = max(truth) - min(truth)

print(f"{len(obs)} `dead` observations, truth {min(truth):.0f} -> {max(truth):.0f} "
      f"(range {rng:.0f})")
n_izmit = sum(1 for o in obs if o["value"] == 17500)
print(f"of which the 1999 Izmit 17,500 appears {n_izmit} times\n")

def score(est):
    se = sum((e - t) ** 2 for e, t in zip(est, truth)) / len(truth)
    return math.sqrt(se) / rng, math.sqrt(se), est[-1]

arms = [
    ("default (as recorded)",      dict(REJECT_SIGMA=None, MAX_RATE=None, CONF_R=False)),
    ("innovation gate 3-sigma",    dict(REJECT_SIGMA=3.0,  MAX_RATE=None, CONF_R=False)),
    ("innovation gate 2-sigma",    dict(REJECT_SIGMA=2.0,  MAX_RATE=None, CONF_R=False)),
    ("rate filter 300/h",          dict(REJECT_SIGMA=None, MAX_RATE=300.0, CONF_R=False)),
    ("gate 3s + rate 300/h",       dict(REJECT_SIGMA=3.0,  MAX_RATE=300.0, CONF_R=False)),
    ("gate 3s + rate + conf-R",    dict(REJECT_SIGMA=3.0,  MAX_RATE=300.0, CONF_R=True)),
]
lv = ekf.est_last_value(obs, grid)
n, a, f = score(lv)
print(f"{'arm':30}{'nRMSE':>9}{'RMSE deaths':>13}{'final':>9}{'sigma_end':>11}")
print(f"{'last_value baseline':30}{n:>9.3f}{a:>13.0f}{f:>9.0f}{'--':>11}")
for name, cfg in arms:
    for k, v in cfg.items():
        setattr(ekf, k, v)
    o2 = ekf.rate_filter(obs, cfg["MAX_RATE"]) if cfg["MAX_RATE"] else obs
    ci = ekf.est_ekf_ci(o2, grid, "dead")
    n, a, f = score([c["mean"] for c in ci])
    sig = ci[-1]["sigma"]
    print(f"{name:30}{n:>9.3f}{a:>13.0f}{f:>9.0f}{sig:>11.0f}")
for k, v in dict(REJECT_SIGMA=None, MAX_RATE=None, CONF_R=False).items():
    setattr(ekf, k, v)
print(f"\ntruth final: {truth[-1]:.0f}")
