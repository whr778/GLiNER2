"""Measure event-type FALSE POSITIVES, which the blind test cannot see.

WHY IT CANNOT SEE THEM. `_schema_from_gold` (training/eval_metrics.py:367) builds the event
menu from the DOCUMENT'S OWN GOLD -- `schema["events"]` contains exactly the types present in
that record and no others. The model is asked "which of these types are here?" where every
option is there by construction, so `event_type` precision is pinned at 1.0000 and
`F1 = 2R/(1+R)` exactly. Verified against all twelve event_type readings on file: 12/12 match
to 1e-4. `event_type` F1 therefore carries no information that recall does not.

Entities, relations and arguments get a gold-restricted menu too, but they must also land the
SPAN, so their precision stays free to be wrong (0.53, 0.32, 0.69). `event_type` is the only
head with nothing else to get wrong.

This probe offers the corpus's FULL taxonomy instead -- every event type the training split
uses -- which is what an actual deployment does, and counts the types the model invents.

    uv run python tools/train/probe_event_type_fp.py --checkpoints A=/path B=/path --n 150
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def schema_from_train(train: Path) -> dict:
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


def gold_types(rec) -> set:
    return {ev.get("event_type") for ev in ((rec.get("output") or {}).get("events") or [])
            if isinstance(ev, dict) and ev.get("event_type")}


def pred_types(out) -> set:
    evs = out.get("event_extraction") if isinstance(out, dict) else None
    if not isinstance(evs, dict):
        return set()
    got = set()
    for t, items in evs.items():
        items = items if isinstance(items, list) else [items]
        if any(isinstance(i, dict) for i in items):
            got.add(t)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="LABEL=/path ...")
    ap.add_argument("--test", default="data/cmnee.test.jsonl")
    ap.add_argument("--train", default="data/cmnee.train.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-chars", type=int, default=900)
    args = ap.parse_args()

    import torch
    from gliner2 import AutoExtractor

    recs = [json.loads(l) for l in Path(args.test).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    recs = [r for r in recs if gold_types(r) and len(r.get("input") or "") <= args.max_chars]
    random.Random(0).shuffle(recs)
    recs = recs[:args.n]
    full = schema_from_train(Path(args.train))
    gold_total = sum(len(gold_types(r)) for r in recs)

    print(f"[type-fp] {len(recs)} documents, FULL menu of {len(full)} event types "
          f"(gold menus average {gold_total/len(recs):.2f} types/doc), threshold "
          f"{args.threshold}\n")
    print(f"{'checkpoint':26s}{'precision':>11}{'recall':>9}{'F1':>8}{'invented':>10}{'predicted':>11}")
    for spec in args.checkpoints:
        label, path = spec.split("=", 1)
        m = AutoExtractor.from_pretrained(path, map_location=args.device)
        getattr(m, "model", m).to(torch.device(args.device)).eval()
        hits = npred = 0
        for r in recs:
            g = gold_types(r)
            p = pred_types(m.extract_events(r["input"], full, threshold=args.threshold))
            hits += len(g & p)
            npred += len(p)
        prec = hits / npred if npred else 0.0
        rec = hits / gold_total if gold_total else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{label:26s}{prec:>11.4f}{rec:>9.4f}{f1:>8.4f}{npred-hits:>10,}{npred:>11,}")
    print("\nprecision here is what the blind test reports as 1.0000 by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
