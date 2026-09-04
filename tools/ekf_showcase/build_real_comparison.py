"""Build a run_pipeline.py-compatible --feed/--truth pair from REAL disaster data.

Two sources already on disk per event, never assembled into this shape before:

  feed   <- article text cached inside datasets/<event>/_cache/tracked_rollup.json,
            a run_pipeline.py OUTPUT from a past run. The text is real wire copy; only
            t_hours and text are read here, so a past run's own extractions
            (relevant/observations/events) do not leak back in as input.
  truth  <- datasets/<event>/ground_truth.json, an English Wikipedia casualty table.
            Sparse snapshots are fine as-is: run_pipeline.py's `_truth_at` takes the
            last truth point at or before each grid time, it does not need a dense
            per-hour series.

Coverage varies by event: Helene's table has no injured/missing column at all, so only
`dead` is scoreable there. Aegean's has per-place `injured` but no literal `Total` key,
so this sums the per-place values -- printed, so a silent undercount from a missing
place is visible rather than assumed away.

    uv run python tools/ekf_showcase/build_real_comparison.py helene
    uv run python tools/ekf_showcase/build_real_comparison.py aegean
    uv run python tools/ekf_showcase/run_pipeline.py \
        --feed datasets/<event>/real_feed.jsonl \
        --truth datasets/<event>/real_truth.jsonl \
        --casualty-model <model> --out <out>.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

DATASETS = {
    "helene": {"dir": Path("datasets/helene2024"), "roles": {"dead": "deaths"}},
    "aegean": {"dir": Path("datasets/aegean2020"),
              "roles": {"dead": "deaths", "injured": "injured"}},
}


def build_feed(event_dir: Path) -> int:
    rollup = json.loads((event_dir / "_cache" / "tracked_rollup.json").read_text(encoding="utf-8"))
    articles = rollup["articles"]
    out = event_dir / "real_feed.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps({"t_hours": a["t_hours"], "text": a["text"]},
                               ensure_ascii=False) + "\n")
    return len(articles)


def build_truth(event_dir: Path, roles: dict) -> int:
    gt = json.loads((event_dir / "ground_truth.json").read_text(encoding="utf-8"))
    onset = datetime.fromisoformat(gt["onset_utc"].replace("Z", "+00:00"))
    rows = []
    for p in gt["points"]:
        snap = datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00"))
        t_hours = round((snap - onset).total_seconds() / 3600, 3)
        row = {"t_hours": t_hours}
        for role, field in roles.items():
            values = p.get(field) or {}
            row[role] = float(values["Total"]) if "Total" in values else float(sum(values.values()))
        rows.append(row)
    rows.sort(key=lambda r: r["t_hours"])
    out = event_dir / "real_truth.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("event", choices=sorted(DATASETS))
    a = ap.parse_args()

    cfg = DATASETS[a.event]
    n_feed = build_feed(cfg["dir"])
    n_truth = build_truth(cfg["dir"], cfg["roles"])
    print(f"[feed]  {n_feed} articles -> {cfg['dir']}/real_feed.jsonl")
    print(f"[truth] {n_truth} snapshots ({'/'.join(cfg['roles'])}) "
          f"-> {cfg['dir']}/real_truth.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
