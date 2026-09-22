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
        # MEASURE the absent queries, do not assert them. `mention_mask` is
        # (batch, query, max_gold); a query whose row is all False carries no gold and IS
        # an absent query by this file's own definition. The previous version counted
        # queries and then PRINTED "ABSENT queries: 0" as a literal, reasoning that every
        # query comes from the document's own gold. That reasoning is true and the
        # conclusion is false: a gold surface that fails to align contributes a query with
        # no mention, so it is absent in effect. The gate could not fail, and the 0 it
        # printed was quoted as "measured" in eb17-best.yaml and LABEL_NEGATIVES_PLAN.md.
        n_absent = 0
        mask = getattr(batch.targets, "mention_mask", None)
        if mask is not None:
            per_query = mask[0].any(dim=-1)
            n_absent = int((~per_query).sum())
        d = per.setdefault(name, [0, 0, 0])
        d[0] += n_q
        d[1] += 1
        d[2] += n_absent

    print(f"{'corpus':12s}{'docs':>7}{'extractive queries':>21}{'per doc':>10}{'ABSENT':>9}{'%':>8}")
    tq = td = ta = 0
    for name, (q, d, a) in sorted(per.items()):
        print(f"{name:12s}{d:>7}{q:>21,}{q/d:>10.1f}{a:>9,}{100.0*a/q if q else 0:>7.2f}%")
        tq += q
        td += d
        ta += a
    if td:
        print(f"{'ALL':12s}{td:>7}{tq:>21,}{tq/td:>10.1f}{ta:>9,}"
              f"{100.0*ta/tq if tq else 0:>7.2f}%")
    print("\nAn ABSENT query is one carrying no gold mention. Before the label-negatives "
          "work none were\nINJECTED -- but that is not the same as none EXISTING: a gold "
          "surface that fails to align\nleaves its query standing with nothing to find, "
          "which is an absent query the corpus\nmanufactured by accident. Read this "
          "beside tools/data/measure_surface_alignment.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
