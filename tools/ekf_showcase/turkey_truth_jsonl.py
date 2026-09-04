"""Convert `turkey2023/ground_truth.json` into the `--truth` JSONL run_pipeline consumes.

The GT stores one point per snapshot with SEPARATE `turkey` and `syria` figures, because
they are separate reporting authorities with separate revision behaviour. run_pipeline's
`--truth` wants flat `{"t_hours": x, "dead": y}` records, so a choice has to be made about
which trajectory "the" answer is -- and for the multilingual feeds that choice is not
neutral:

  turkey    what RESULTS.md scores the English feed against ("nRMSE vs Turkiye").
  syria     the smaller stream, poorly tracked by every run so far.
  combined  turkey + syria. The CHINESE feed reports this constantly -- "土耳其和叙利亚
            两国超2万人遇难" (over 20,000 dead across BOTH countries) -- and the Turkish
            feed sometimes does too. Scoring a combined-reporting feed against the
            Turkiye-only trajectory would charge it for being accurate.

So all three are emitted and results should be reported against all three, rather than
picking the flattering one after the fact.

    uv run python tools/ekf_showcase/turkey_truth_jsonl.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ONSET = datetime(2023, 2, 6, 1, 17, tzinfo=timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", default="datasets/turkey2023/ground_truth.json")
    ap.add_argument("--out-dir", default="datasets/turkey2023")
    args = ap.parse_args()

    gt = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)

    for arm in ("turkey", "syria", "combined"):
        rows = []
        for p in gt["points"]:
            stamp = datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00"))
            value = p["turkey"] + p["syria"] if arm == "combined" else p[arm]
            rows.append({"t_hours": round((stamp - ONSET).total_seconds() / 3600.0, 3),
                         "dead": float(value)})
        out = out_dir / f"truth_{arm}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[truth] {arm:9} {len(rows)} points  "
              f"{rows[0]['dead']:.0f} -> {rows[-1]['dead']:.0f}  -> {out}")


if __name__ == "__main__":
    main()
