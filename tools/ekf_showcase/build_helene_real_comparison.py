"""Build a run_pipeline.py-compatible --feed/--truth pair from REAL Helene data.

Two sources already on disk, never assembled into this shape before:

  feed   <- article text cached inside datasets/helene2024/_cache/tracked_rollup.json,
            a run_pipeline.py OUTPUT from a past run. The text is real AP wire copy;
            only t_hours and text are read here, so a past run's own extractions
            (relevant/observations/events) do not leak back in as input.
  truth  <- datasets/helene2024/ground_truth.json, English Wikipedia's per-state
            casualty table. Sparse snapshots are fine as-is: run_pipeline.py's
            `_truth_at` takes the last truth point at or before each grid time, it
            does not need a dense per-hour series.

Ground truth here covers `dead` (Total) ONLY -- Wikipedia's table has no injured/missing
column, so those roles cannot be scored against real Helene truth with this file.

    uv run python tools/ekf_showcase/build_helene_real_comparison.py
    uv run python tools/ekf_showcase/run_pipeline.py \
        --feed datasets/helene2024/real_feed.jsonl \
        --truth datasets/helene2024/real_truth.jsonl \
        --casualty-model <model> --out <out>.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

CACHE = Path("datasets/helene2024/_cache/tracked_rollup.json")
GROUND_TRUTH = Path("datasets/helene2024/ground_truth.json")
FEED_OUT = Path("datasets/helene2024/real_feed.jsonl")
TRUTH_OUT = Path("datasets/helene2024/real_truth.jsonl")


def build_feed() -> int:
    rollup = json.loads(CACHE.read_text(encoding="utf-8"))
    articles = rollup["articles"]
    with FEED_OUT.open("w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps({"t_hours": a["t_hours"], "text": a["text"]},
                               ensure_ascii=False) + "\n")
    return len(articles)


def build_truth() -> int:
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    onset = datetime.fromisoformat(gt["onset_utc"].replace("Z", "+00:00"))
    rows = []
    for p in gt["points"]:
        snap = datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00"))
        t_hours = (snap - onset).total_seconds() / 3600
        rows.append({"t_hours": round(t_hours, 3), "dead": float(p["deaths"]["Total"])})
    rows.sort(key=lambda r: r["t_hours"])
    with TRUTH_OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    n_feed = build_feed()
    n_truth = build_truth()
    print(f"[feed]  {n_feed} articles -> {FEED_OUT}")
    print(f"[truth] {n_truth} snapshots (dead/Total only) -> {TRUTH_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
