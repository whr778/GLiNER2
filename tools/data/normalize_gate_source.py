"""Collapse a per-outlet `source` field to one value, for build_gate_corpus.py's balance().

`balance()` equalises positive/negative classes inside every (source, length-decile)
cell, which is what stops a source's language or register from leaking into the label.
That only works when a source IS one stratum. `data/chinese_gate/zh_gate_sample.jsonl`
carries 1,909 distinct per-outlet source names over 4,994 rows -- nearly one source per
document -- so fed in as-is, almost every row lands in a singleton cell with no
positive/negative counterpart and is silently dropped. `turkish_news` avoided this by
construction: all 5,638 Turkish rows already share one source value.

Collapses to ONE value, preserving the original under `_orig_source` for provenance
(read by nothing downstream; `build_gate_corpus.load_annotated` only reads
input/label/source). Not language-specific -- any adjudicated pool with per-document
source names needs this before it can be added as a stratum.

    uv run python tools/data/normalize_gate_source.py \
        data/chinese_gate/zh_gate_sample.jsonl data/chinese_gate/gate_ann_chinese.jsonl zh_news
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("source", help="the single source value every row is collapsed to")
    a = ap.parse_args()

    n = 0
    with open(a.outfile, "w", encoding="utf-8") as f:
        for line in open(a.infile, encoding="utf-8"):
            rec = json.loads(line)
            rec = {**rec, "source": a.source, "_orig_source": rec["source"]}
            f.write(dumps_record(rec) + "\n")
            n += 1
    print(f"[normalize] {n} rows -> {a.outfile}, source={a.source!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
