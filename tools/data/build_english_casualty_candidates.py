"""Select English casualty-annotation candidates from DocEE, cheapest-first.

Annotation is billed per DOCUMENT and valued per POSITIVE, so the filter's precision is
the price. Measured on DocEE's own labels: 30.4% of its 27,307 documents carry a casualty
label, and the free toll regex lifts that to 53.0% purity while keeping 95.5% of the
positives. CC-News is the wrong pool for this by a factor of 38 (0.47% base rate).

SPLIT DESIGN MATTERS HERE. Every document in docee.train was trained on by the 137k base,
so a casualty test split drawn from it would not be blind. Candidates are therefore tagged
with their source split, and `--eval-only` selects from docee.val/test -- text the base
only ever blind-tested -- so a clean held-out set can be carved from the purchase.

    uv run python tools/data/build_english_casualty_candidates.py --out data/en_cas_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_purity_curve import PREFILTERS  # noqa: E402

SOURCES = {
    "data/docee.train.jsonl": "docee-train",
    "data/docee.val.jsonl": "docee-val",
    "data/docee.test.jsonl": "docee-test",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/en_cas_candidates.jsonl")
    parser.add_argument("--eval-only", action="store_true",
                        help="only docee val/test, which the base never trained on")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rx = PREFILTERS["en"]
    kept, seen, by_source = [], set(), {}
    for path, tag in SOURCES.items():
        if args.eval_only and tag == "docee-train":
            continue
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"missing source: {path}")
        for line in p.open(encoding="utf-8"):
            rec = json.loads(line)
            text = (rec.get("input") or "").strip()
            if not text or text in seen or not rx.search(text):
                continue
            seen.add(text)
            kept.append({"input": text, "source": tag})
            by_source[tag] = by_source.get(tag, 0) + 1
            if args.limit and len(kept) >= args.limit:
                break

    # SHUFFLE, deterministically. DocEE is ordered by event type, so file order is not a
    # sample: annotating the first 60 candidates drew 0/60 casualty-positive documents
    # where a random 400 drew 49.5%. Any --limit run, and any head-of-file inspection,
    # silently selects one event type unless the pool is shuffled first.
    random.Random(args.seed).shuffle(kept)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    per_doc = (1373 * 0.50 + 260 * 2.50) / 1e6
    print(f"wrote {out}: {len(kept)} candidates  {by_source}")
    print(f"  estimated batch cost: ${len(kept) * per_doc:,.2f} (Haiku 4.5 batch)")
    print(f"  expected positives at the measured 53.0% purity: ~{int(len(kept) * 0.53):,}")


if __name__ == "__main__":
    main()
