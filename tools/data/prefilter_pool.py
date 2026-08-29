"""Apply the free numeric-toll regex before the pool ever reaches the GPU.

The model is the expensive half; this halves what it must score, at 83.7% recall of
positives and BETTER purity than the gate's own usable cut (40.8% vs 37.3%). See
tools/data/notes/ANNOTATION_ECONOMICS.md.

    uv run python tools/data/prefilter_pool.py --pool ... --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402
from gate_purity_curve import TOLL_NEAR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="/Volumes/Development/data/turkish_pool18.jsonl")
    ap.add_argument("--out", default="/Volumes/Development/data/turkish_pool18_prefiltered.jsonl")
    a = ap.parse_args()

    kept, total, outlets = 0, 0, Counter()
    with Path(a.out).open("w", encoding="utf-8") as fh:
        for line in Path(a.pool).open(encoding="utf-8"):
            total += 1
            row = json.loads(line)
            if TOLL_NEAR.search(row["input"]):
                fh.write(dumps_record(row) + "\n")
                kept += 1
                outlets[row.get("outlet")] += 1
    print(f"[prefilter] {kept}/{total} = {kept / total:.1%} kept -> {a.out}")
    for site, n in outlets.most_common():
        print(f"[prefilter]   {site:22s} {n:7,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
