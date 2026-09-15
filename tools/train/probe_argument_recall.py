"""Is the event-records model's low argument RECALL undertraining, or a missing mechanism?

The epoch-1 sweep found strict argument PRECISION up 1.3-1.7x over the incumbent at every
matched threshold, and recall down 4-10x. Those have opposite implications: better binding is
the record head working, while a recall floor could be structural -- a cap the decoder cannot
exceed no matter how long it trains.

The discriminator is a CURVE, not a level. Run the same documents through two checkpoints of
the SAME run and look at argument recall. Climbing materially between epochs means the floor
is training; flat means it is built in.

Recall is scored RELAXED -- (event_type, role, entity), no trigger, no instance binding --
because that is the metric that collapsed (0.7059 -> 0.1446 on the blind test) and because it
is blind to the binding question, so it isolates "did the model propose the argument at all".

    uv run python tools/train/probe_argument_recall.py \
        --checkpoints A=/path/ep1 B=/path/ep2 --n 200
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def gold_pairs(rec) -> set:
    out = set()
    for ev in ((rec.get("output") or {}).get("events") or []):
        if not isinstance(ev, dict):
            continue
        t = ev.get("event_type")
        for a in ev.get("arguments") or []:
            if isinstance(a, dict) and a.get("role") and a.get("entity"):
                out.add((t, a["role"], str(a["entity"]).strip()))
    return out


def pred_pairs(out) -> set:
    got = set()
    evs = out.get("event_extraction") if isinstance(out, dict) else None
    if not isinstance(evs, dict):
        return got
    for t, items in evs.items():
        for inst in (items if isinstance(items, list) else [items]):
            if not isinstance(inst, dict):
                continue
            for a in inst.get("arguments") or []:
                if isinstance(a, dict) and a.get("role") and a.get("entity"):
                    got.add((t, a["role"], str(a["entity"]).strip()))
    return got


def schema_from_train(train: Path) -> dict:
    from collections import defaultdict
    roles = defaultdict(set)
    for line in train.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for ev in ((json.loads(line).get("output") or {}).get("events") or []):
            if isinstance(ev, dict) and ev.get("event_type"):
                for a in ev.get("arguments") or []:
                    if isinstance(a, dict) and a.get("role"):
                        roles[ev["event_type"]].add(a["role"])
    return {t: sorted(r) for t, r in sorted(roles.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="LABEL=/path ...")
    ap.add_argument("--test", default="data/cmnee.test.jsonl")
    ap.add_argument("--train", default="data/cmnee.train.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-chars", type=int, default=900)
    args = ap.parse_args()

    import torch
    from gliner2 import AutoExtractor

    recs = []
    for line in Path(args.test).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if gold_pairs(r) and len(r.get("input") or "") <= args.max_chars:
            recs.append(r)
    random.Random(0).shuffle(recs)          # fixed seed: same documents for every checkpoint
    recs = recs[:args.n]
    schema = schema_from_train(Path(args.train))
    gold_total = sum(len(gold_pairs(r)) for r in recs)
    print(f"[recall] {len(recs)} documents, {gold_total:,} gold (type, role, entity) triples, "
          f"threshold {args.threshold}, device {args.device}\n")

    print(f"{'checkpoint':28s}{'recall':>9}{'precision':>11}{'F1':>8}{'predicted':>11}{'hits':>8}")
    for spec in args.checkpoints:
        label, path = spec.split("=", 1)
        m = AutoExtractor.from_pretrained(path, map_location=args.device)
        getattr(m, "model", m).to(torch.device(args.device)).eval()
        hits = npred = 0
        for r in recs:
            g = gold_pairs(r)
            p = pred_pairs(m.extract_events(r["input"], schema, threshold=args.threshold))
            hits += len(g & p)
            npred += len(p)
        rec = hits / gold_total if gold_total else 0.0
        prec = hits / npred if npred else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{label:28s}{rec:>9.4f}{prec:>11.4f}{f1:>8.4f}{npred:>11,}{hits:>8,}")
    print("\nrelaxed = (event_type, role, entity); no trigger, no instance binding, so this")
    print("isolates whether the argument was PROPOSED at all from whether it was bound right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
