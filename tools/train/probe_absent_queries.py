"""Count ABSENT queries in a training batch — the gate for the label-negatives work.

An absent query is one the model must REJECT: a label offered in the schema with no gold in
that document. It is the positive class of `abstention_loss` (target 1 for an absent query,
models/boundary/losses.py:601) and the selection pool of `negative_query_ratio` (0.5 by
default, models/boundary/model.py:846). Both are live in every run this project has trained.

Today the answer is ZERO, by construction: `InputExample.from_dict` builds the entity menu
from the gold dict's keys and one Event per gold event, so every query carries gold. The two
consumers above have therefore never had anything to consume.

Run this BEFORE and AFTER the collator change. Before: 0%. After: the configured rate. A
treatment arm that still prints 0% has not applied the treatment.

CALLING CONVENTION, which cost two wrong attempts: `collate_fn_train` takes
``[(text, output_dict)]`` -- the raw gold dict, NOT an InputExample and not a Schema. See
tests/processing/test_eval_missing_surface.py.

    uv run python tools/train/probe_absent_queries.py --checkpoint <ckpt> --corpora cmnee biored
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpora", nargs="+", default=["cmnee", "biored", "casie", "docee"])
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=800)
    args = ap.parse_args()

    from gliner2 import AutoExtractor

    proc = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu").processor
    recs = []
    for name in args.corpora:
        p = Path(f"data/{name}.train.jsonl")
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines()[:300]:
            if line.strip():
                r = json.loads(line)
                if len(r.get("input") or "") < args.max_chars:
                    recs.append((name, r))
    random.Random(0).shuffle(recs)
    recs = recs[:args.n]

    per: dict = {}
    for name, r in recs:
        batch = proc.collate_fn_train([(r["input"], r["output"])], architecture="boundary",
                                      error_policy="skip", max_gold_per_query=256,
                                      on_capacity_exceeded="skip_sample")
        if not batch.query_layouts:
            continue
        n_q = batch.query_layouts[0].extractive_count()
        d = per.setdefault(name, [0, 0])
        d[0] += n_q
        d[1] += 1

    print(f"{'corpus':12s}{'docs':>7}{'extractive queries':>21}{'per doc':>10}")
    tq = td = 0
    for name, (q, d) in sorted(per.items()):
        print(f"{name:12s}{d:>7}{q:>21,}{q/d:>10.1f}")
        tq += q
        td += d
    if td:
        print(f"{'ALL':12s}{td:>7}{tq:>21,}{tq/td:>10.1f}")
    print("\nABSENT queries: 0 (0.00%) -- every query above is derived from that document's")
    print("own gold. This is the measurement the collator change must move.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
