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
    "en": ["data/cas_ann_en.jsonl",                      # REAL English news (DocEE)
           "data/cas_ann_ccnews.jsonl"],                 # REAL English news (CC-News)
    "tr": ["data/turkish_gate/cas_ann_tr.jsonl"],        # REAL Turkish news
    "zh": ["data/chinese_gate/cas_ann_zh.jsonl",
           "data/chinese_gate/cas_ann_zh2.jsonl"],       # REAL Chinese news
}

# The 137k base TRAINED on docee.train, so an English casualty row derived from it is not
# blind -- the base has already seen that text, with different labels. Those rows are
# still fine to TRAIN on; they must not reach val or test. CC-News is absent from the base
# mix entirely, so it is safe throughout.
BASE_TRAINED_ON = ["data/docee.train.jsonl"]


def load(paths):
    rows = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise SystemExit(f"missing source: {p}")
        rows += [json.loads(line) for line in path.open(encoding="utf-8")]
    return rows


def base_seen_keys():
    """Document keys the 137k base trained on, by the splitter's own key rule."""
    from _split import normalize_group_key
    seen = set()
    for p in BASE_TRAINED_ON:
        path = Path(p)
        if not path.is_file():
            raise SystemExit(f"missing base corpus: {p}")
        for line in path.open(encoding="utf-8"):
            seen.add(normalize_group_key(json.loads(line).get("input") or ""))
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/casualty_ml", help="split base path")
    parser.add_argument("--per-language", type=int, default=0,
                        help="records per language; 0 = the smallest language's count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-per-language", type=int, default=1400,
                        help="val AND test size per language; equal by construction")
    args = parser.parse_args()

    pools = {lang: load(paths) for lang, paths in SOURCES.items()}
    for lang, rows in pools.items():
        print(f"  {lang}: {len(rows)} available")

    from _split import dumps_record, normalize_group_key
    n = args.per_language or min(len(r) for r in pools.values())
    rng = random.Random(args.seed)
    seen = base_seen_keys()

    # val/test are sized EQUALLY per language and drawn only from rows the base never
    # trained on. Both constraints matter and they pull against each other: pinning
    # base-seen English rows to train shrinks the English eval pool, and letting the split
    # ratio absorb that leaves a test set weighted 3:1 against English -- which lets a
    # model strong in the other two languages score well. Fix the eval size instead and
    # let train take the remainder.
    eval_each = args.eval_per_language
    splits = {"train": [], "val": [], "test": []}
    for lang, rows in pools.items():
        chosen = rows if len(rows) <= n else rng.sample(rows, n)
        safe = [r for r in chosen if normalize_group_key(r.get("input") or "") not in seen]
        pinned = len(chosen) - len(safe)
        rng.shuffle(safe)
        if len(safe) < 2 * eval_each:
            raise SystemExit(f"{lang}: only {len(safe)} base-unseen rows, need "
                             f"{2 * eval_each} for a balanced val+test")
        splits["val"] += safe[:eval_each]
        splits["test"] += safe[eval_each:2 * eval_each]
        splits["train"] += safe[2 * eval_each:] + [r for r in chosen if r not in safe[:2 * eval_each]
                                                   and normalize_group_key(r.get("input") or "") in seen]
        note = f"   ({pinned} base-seen -> train)" if pinned else ""
        print(f"  {lang}: {len(chosen)} chosen, {len(safe)} base-unseen{note}")

    counts = {}
    for name, rows in splits.items():
        path = Path(str(args.out) + f".{name}.jsonl")
        with path.open("w", encoding="utf-8") as fh:
            for record in rows:
                fh.write(dumps_record(record) + "\n")
        counts[name] = len(rows)
    print(f"  wrote {counts}")

    print(f"\nbalanced at {n} per language, {n * len(pools)} total")


if __name__ == "__main__":
    main()
