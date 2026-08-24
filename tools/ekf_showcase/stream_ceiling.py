"""How much of a stream's error is METHOD, and how much is source coverage?

Before spending on association or filtering for a stream, price the ceiling: at each grid
point take the observation closest to truth among those seen so far. No method can beat
that from these observations. If the oracle is still far from truth, the residual is what
the feed never said, and no gate reaches it.

Measured on Helene after the scope gate:

    North Carolina   achieved 52.4   oracle 23.0   -> 29.4 addressable, 23.0 a floor
    Total            achieved 26.7   oracle 16.0   -> 10.7 addressable

North Carolina's largest reported figure is 98 against a truth of 123, so 25 deaths never
appear in the feed at all. The national stream has full coverage (250 vs 250) and is
within 10.7 deaths of its own ceiling -- close to solved.

Original question:

An ORACLE bound: at each grid point take the observation closest to truth among those
available so far. No method can do better from these observations. If the oracle is still
far from truth, the residual is source coverage, not association or filtering.
"""
import json, math, sys
from datetime import datetime
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "datasets/disaster_streams"))
sys.path.insert(0, str(REPO / "tools/ekf_showcase"))
from run_pipeline import apply_rollup, merge_prefix_keys, scope_filter
from scope_gate import gate as scope_gate

track = json.loads((REPO / "datasets/helene2024/_cache/tracked_rollup.json").read_text())
rollup = json.loads((REPO / "datasets/helene2024/rollup.json").read_text())
gt = json.loads((REPO / "datasets/helene2024/ground_truth.json").read_text())
grid = track["grid"]
onset = datetime.fromisoformat(gt["onset_utc"].replace("Z", "+00:00"))
tp = [(datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00")) - onset).total_seconds() / 3600
      for p in gt["points"]]

def truth_series(name):
    vals = [float(p["deaths"][name]) for p in gt["points"]]
    out, prev = [], vals[0]
    for t in grid:
        for a, b in zip(tp, vals):
            if a > t: break
            prev = b
        out.append(prev)
    return out

obs = [o for a in track["articles"] for o in a["observations"] if o["role"] == "dead"]
apply_rollup(obs, rollup); merge_prefix_keys(obs)
kept, _, _ = scope_filter(obs, rollup)
states = {str(k).lower(): str(k) for k in (rollup.get("hierarchy") or {}).get("parts") or []}
gated, _, _ = scope_gate(kept, 2.0, 2, states, "aggregate")

for key, name in (("north carolina", "North Carolina"), ("__aggregate__", "Total")):
    rows = sorted(gated.get(key, []), key=lambda o: o["t_hours"])
    tr = truth_series(name)
    oracle, avail = [], []
    for t, g in zip(grid, tr):
        avail += [float(o["value"]) for o in rows if o["t_hours"] <= t
                  and o["t_hours"] > (grid[grid.index(t) - 1] if grid.index(t) else -1)]
        oracle.append(min(avail, key=lambda v: abs(v - g)) if avail else 0.0)
    orc = math.sqrt(sum((a - b) ** 2 for a, b in zip(oracle, tr)) / len(tr))
    vals = [float(o["value"]) for o in rows]
    print(f"{name}: {len(rows)} obs, values {min(vals):.0f}-{max(vals):.0f}, "
          f"truth final {tr[-1]:.0f}")
    print(f"   ORACLE best-available RMSE = {orc:.1f} deaths   "
          f"(EKF+gates achieved 52.4 / 26.7)")
    print(f"   max observation {max(vals):.0f} vs truth max {max(tr):.0f} -> "
          f"{max(0.0, max(tr) - max(vals)):.0f} deaths never appear in the feed\n")
