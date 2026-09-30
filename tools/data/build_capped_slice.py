"""Write a seeded slice of whole documents from several corpora, capped by event-argument gold.

Used to hold LLM-annotated data at a fixed real-text : synthetic-text ratio measured in
ARGUMENT GOLD (the unit the eb18 balance is set in), rather than in documents: the
synthetic corpora differ ~11x in arguments per document (sonnet5_1k 15.6, haiku45_5k 1.4),
so a document count would say nothing about the supervision bought.

Documents are pooled, shuffled with a fixed seed and taken whole until the running argument
gold reaches the cap, so the slice keeps the pool's natural mix -- including documents with
no events -- instead of favouring whichever generator is densest. Arguments are counted with
the evaluator's own key set, so the cap means what eval scores.

    uv run python tools/data/build_capped_slice.py --cap 24809 --seed 0 \\
        --out data/synthetic_events_capped.train.jsonl \\
        data/synthetic_sonnet5_1k.train.jsonl data/synthetic_haiku45_5k.train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

from gliner2.training.eval_metrics import _gold_event_argument_set  # noqa: E402


def load(paths):
    """Every record of every source, tagged with its source file name."""
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            out += [(Path(path).name, json.loads(line)) for line in fh if line.strip()]
    return out


def take_until(records, cap, seed):
    """Seeded shuffle, then whole documents until argument gold reaches `cap`."""
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    picked, total = [], 0
    for i in order:
        if total >= cap:
            break
        picked.append(records[i])
        total += len(_gold_event_argument_set(records[i][1].get("output") or {}))
    return picked, total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--cap", type=int, required=True, help="argument gold to reach")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = load(args.sources)
    picked, total = take_until(records, args.cap, args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        for _, rec in picked:
            fh.write(dumps_record(rec) + "\n")
    by_source = Counter(name for name, _ in picked)
    print(f"{len(picked):,} of {len(records):,} documents, {total:,} argument gold "
          f"(cap {args.cap:,}) -> {args.out}")
    for name, n in sorted(by_source.items()):
        print(f"  {name}: {n:,} documents")


if __name__ == "__main__":
    main()
