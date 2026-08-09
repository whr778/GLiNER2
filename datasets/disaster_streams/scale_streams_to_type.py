"""Rescale synthetic stream magnitudes to the event type they are paired with.

The generator produces one magnitude regime, tuned to large disasters (the Venezuela 2026
toll ran ~920 -> ~6,000+; median stream peak here is 8,481). That is right for an
earthquake and absurd for a nightclub fire. Pairing those streams with real DocEE
contexts made 80% of them implausible: a Road Crash peaking at 52,447 casualties, a Mine
Collapse at 41,713.

The types that survive DocEE's location filter are precisely the SMALL-scale ones --
Earthquakes and Floods usually have no specific Location annotated, so they drop out of
the context pool, leaving Road Crash, Fire, Mine Collapses and friends holding
earthquake-sized tolls.

Fixing it by re-assigning types would collapse type diversity (nearly every stream would
have to become an earthquake), and diversity is exactly what association testing needs.
So rescale instead: multiply each stream's trajectory and observations by a per-type
factor, keeping the SHAPE of the dynamics (rise, decay, hedging, source noise) that the
tracker is being tested on, and changing only the scale.

Both observations.jsonl and trajectory.jsonl are scaled by the same factor, so ground
truth stays consistent with its observations.

    uv run python datasets/disaster_streams/scale_streams_to_type.py \
        --contexts datasets/disaster_streams/contexts.json \
        --src datasets/disaster_streams --split train \
        --out datasets/disaster_streams_scaled
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Plausible PEAK death toll by event type, from real-world disasters. Used only to derive
# a scale factor, so order of magnitude is what matters.
TYPE_PEAK = {
    "Road Crash": 60,
    "Fire": 200,
    "Gas Explosion": 150,
    "Mine Collapses": 120,
    "Train Collisions": 200,
    "Air Crash": 300,
    "Shipwreck": 800,
    "Armed Conflict": 20000,
    "Earthquakes": 50000,
    "Floods": 20000,
}
DEFAULT_PEAK = 500


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contexts", default="datasets/disaster_streams/contexts.json")
    ap.add_argument("--src", default="datasets/disaster_streams")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="datasets/disaster_streams_scaled")
    ap.add_argument("--min-value", type=int, default=1,
                    help="floor after scaling; a report of 0 dead is not a report")
    args = ap.parse_args()

    ctx = json.loads(Path(args.contexts).read_text(encoding="utf-8"))
    src = Path(args.src) / args.split
    obs = [json.loads(l) for l in (src / "observations.jsonl").open(encoding="utf-8") if l.strip()]
    traj = [json.loads(l) for l in (src / "trajectory.jsonl").open(encoding="utf-8") if l.strip()]

    peak = defaultdict(float)
    for o in obs:
        peak[o["stream_id"]] = max(peak[o["stream_id"]], float(o["value"]))

    factor = {}
    for sid, c in ctx.items():
        target = TYPE_PEAK.get(c["event_type"], DEFAULT_PEAK)
        current = peak.get(sid, 0.0)
        factor[sid] = (target / current) if current > 0 else 1.0

    def scale(sid, value):
        return max(args.min_value, int(round(float(value) * factor.get(sid, 1.0))))

    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    with (out / "observations.jsonl").open("w", encoding="utf-8") as f:
        for o in obs:
            if o["stream_id"] not in ctx:
                continue
            f.write(json.dumps({**o, "value": scale(o["stream_id"], o["value"])},
                               ensure_ascii=False) + "\n")
            kept += 1
    with (out / "trajectory.jsonl").open("w", encoding="utf-8") as f:
        for t in traj:
            if t["stream_id"] not in ctx:
                continue
            row = dict(t)
            for role in ("dead", "injured", "missing"):
                if role in row:
                    row[role] = round(float(row[role]) * factor.get(t["stream_id"], 1.0), 3)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    new_peak = defaultdict(float)
    for l in (out / "observations.jsonl").open(encoding="utf-8"):
        o = json.loads(l)
        new_peak[o["stream_id"]] = max(new_peak[o["stream_id"]], o["value"])
    bad = sum(1 for sid, c in ctx.items()
              if new_peak.get(sid, 0) > TYPE_PEAK.get(c["event_type"], DEFAULT_PEAK) * 1.5)
    print(f"wrote {out}")
    print(f"  streams scaled : {len(ctx)}   observations: {kept}")
    print(f"  peak range     : {min(new_peak.values()):.0f} .. {max(new_peak.values()):.0f}")
    print(f"  still implausible for their type: {bad}/{len(ctx)}")


if __name__ == "__main__":
    main()
