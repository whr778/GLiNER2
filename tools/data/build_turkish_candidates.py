"""Cue-bearing Turkish news, in the shape annotate_gate.py consumes.

The gate reads English and Chinese and scores AUC 0.4733 on Turkish -- below chance --
because its corpus contains no Turkish at all. Adjudicated Turkish text is what would fix
that; this selects the text worth paying to adjudicate.

Only the AMBIGUOUS region is bought, same rule as the English annotator: a document with
no casualty cue is a free negative and never reaches the API. Measured on
denizzhansahin/Turkish_News-2024 (TRT Haber, 2024): of 17,746 articles at least 400 chars,
4,880 carry a cue -- 27.5% -- and 860 of those also carry a disaster word.

WHAT THIS CANNOT FIX. One outlet, one year. A gate trained only on this learns TRT's
register alongside Turkish, and the register confound is exactly what invalidated the v1
gate corpus. Treat a model trained on it as evidence Turkish is learnable, not as a
shippable multilingual gate.

    uv run python tools/data/build_turkish_candidates.py --limit 1000
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record, normalize_group_key  # noqa: E402

# The Turkish counterpart of annotate_gate.CUE: words that put a document in the region
# where the four-way label is genuinely in doubt. Deliberately broad -- it selects what
# gets adjudicated, and the adjudication, not this regex, assigns the label.
CUE = re.compile(
    r"(öldü|ölü|ölüm|can kaybı|hayatını kaybet|yaralı|yaraland|hayatını yitir|"
    r"kayıp|enkaz|göçük|kurtarıl|tahliye|yaşamını yitir)", re.I)
MAX_CHARS = 6000


def candidates(limit: int, seed: int) -> list[str]:
    from datasets import load_dataset
    rows, seen = [], set()
    for record in load_dataset("denizzhansahin/Turkish_News-2024", split="train"):
        text = ((record["Baslik"] or "") + ". " + (record["Icerik"] or "")).strip()
        if len(text) < 400 or not CUE.search(text):
            continue
        key = normalize_group_key(text)[:300]
        if key in seen:
            continue
        seen.add(key)
        rows.append(text[:MAX_CHARS])
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="0 = every cue-bearing article")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/turkish_news_candidates.jsonl")
    a = ap.parse_args()

    rows = candidates(a.limit, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for text in rows:
            handle.write(dumps_record({"input": text, "source": "turkish_news"}) + "\n")
    print(f"[turkish-cands] {len(rows)} cue-bearing candidates -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
