"""Build a LANGUAGE-BALANCED casualty structure corpus: English, Turkish, Simplified Chinese.

Every casualty corpus in this repo is 100% English and 100% LLM-generated text, so the
extractor learns the register of synthetic disaster copy. Meanwhile 31,263 Turkish and
20,901 Chinese field-level casualty records -- REAL news, already annotated and already
paid for -- sit unused in turkish_gate/ and chinese_gate/, in exactly the schema the EKF
consumes (`casualty_report` with mode/anchor).

Balanced by DOWNSAMPLING to the smallest language, so no arm can dominate by volume. The
English arm is still synthetic; that is the remaining gap, not a property of the design.
Swap in a real English corpus here when one exists and the balance holds.

    uv run python tools/train/build_casualty_multilingual.py --out data/casualty_ml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from _split import SplitWriter  # noqa: E402

SOURCES = {
    # casualty_natural ships train-only; it is re-split here with the others.
    "en": ["data/casualty_natural.train.jsonl"],         # synthetic text, real schema
    "tr": ["data/turkish_gate/cas_ann_tr.jsonl"],        # REAL Turkish news
    "zh": ["data/chinese_gate/cas_ann_zh.jsonl",
           "data/chinese_gate/cas_ann_zh2.jsonl"],       # REAL Chinese news
}


def load(paths):
    rows = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise SystemExit(f"missing source: {p}")
        rows += [json.loads(line) for line in path.open(encoding="utf-8")]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/casualty_ml", help="split base path")
    parser.add_argument("--per-language", type=int, default=0,
                        help="records per language; 0 = the smallest language's count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pools = {lang: load(paths) for lang, paths in SOURCES.items()}
    for lang, rows in pools.items():
        print(f"  {lang}: {len(rows)} available")

    n = args.per_language or min(len(r) for r in pools.values())
    rng = random.Random(args.seed)
    counts = {}
    with SplitWriter(Path(args.out), seed=args.seed) as writer:
        for lang, rows in pools.items():
            chosen = rows if len(rows) <= n else rng.sample(rows, n)
            per = {"train": 0, "val": 0, "test": 0}
            for record in chosen:
                per[writer.write(record)] += 1
            counts[lang] = per
            print(f"  {lang}: wrote {sum(per.values())} -> {per}")
    print(f"\nbalanced at {n} per language, {n * len(pools)} total")


if __name__ == "__main__":
    main()
