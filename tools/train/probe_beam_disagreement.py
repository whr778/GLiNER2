"""Is the joint beam INERT, or is it a coin flip? The decode-arms run cannot tell.

`decode_mode: joint` matched greedy on all seven heads at 2.5x the wall clock. Aggregate
parity has two very different explanations:

  (a) the constraints rarely BIND -- the beam returns what greedy returned, so there is
      nothing for a beam-aware loss to optimise toward and the train/test mismatch is moot;
  (b) the beam changes plenty and WINS AND LOSES EQUALLY -- the candidate scores carry no
      information about which constraint-consistent assignment is correct, which is exactly
      what a structured or beam-aware loss would fix.

Only a per-document comparison separates them. This decodes the SAME checkpoint twice and, on
every disagreement, asks which arm matched gold.

    uv run python tools/train/probe_beam_disagreement.py --checkpoint <ckpt> --n 100
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def triples(pred: dict):
    """Flatten a prediction to comparable (task, label, surface) tuples."""
    out = set()
    if not isinstance(pred, dict):
        return out
    for lab, items in (pred.get("entities") or {}).items():
        for it in (items if isinstance(items, list) else [items]):
            sfc = it.get("text") if isinstance(it, dict) else it
            if isinstance(sfc, str):
                out.add(("entity", lab, sfc.strip()))
    for t, insts in (pred.get("event_extraction") or {}).items():
        for inst in (insts if isinstance(insts, list) else [insts]):
            if not isinstance(inst, dict):
                continue
            for tr in inst.get("triggers") or []:
                if isinstance(tr, str):
                    out.add(("trigger", t, tr.strip()))
            for a in inst.get("arguments") or []:
                if isinstance(a, dict) and a.get("entity"):
                    out.add(("argument", f"{t}/{a.get('role')}", str(a["entity"]).strip()))
    return out


def gold_triples(output: dict):
    out = set()
    for lab, ss in (output.get("entities") or {}).items():
        for s in ss or []:
            if isinstance(s, str):
                out.add(("entity", lab, s.strip()))
    for ev in output.get("events") or []:
        if not isinstance(ev, dict):
            continue
        for tr in ev.get("triggers") or []:
            if isinstance(tr, str):
                out.add(("trigger", ev.get("event_type"), tr.strip()))
        for a in ev.get("arguments") or []:
            if isinstance(a, dict) and a.get("entity"):
                out.add(("argument", f"{ev.get('event_type')}/{a.get('role')}",
                         str(a["entity"]).strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test", default="data/cmnee.test.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from gliner2 import AutoExtractor
    from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head
    from gliner2.training.eval_metrics import _schema_from_gold

    def load(mode):
        m = AutoExtractor.from_pretrained(args.checkpoint, map_location=args.device)
        merged = dict(getattr(m.config, "boundary_head", None) or {})
        merged["decode_mode"] = mode
        m.config.boundary_head = merged
        settings = BoundaryHeadSettings(**validate_boundary_head(merged))
        m.boundary_settings = settings
        if getattr(m, "boundary_head", None) is not None:
            m.boundary_head.settings = settings
        getattr(m, "model", m).to(torch.device(args.device)).eval()
        return m

    recs = [json.loads(l) for l in Path(args.test).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    recs = [r for r in recs if (r.get("output") or {}).get("events")][:args.n]
    texts, schemas, golds = [], [], []
    for r in recs:
        sch = _schema_from_gold(r["output"])
        if sch:
            texts.append(r["input"]); schemas.append(sch); golds.append(r["output"])

    preds = {}
    for mode in ("greedy", "joint"):
        m = load(mode)
        preds[mode] = m.batch_extract(texts, schemas, batch_size=8, threshold=args.threshold)
        del m

    stats = Counter()
    by_task = Counter()
    for pg, pj, gold in zip(preds["greedy"], preds["joint"], golds):
        g, j, y = triples(pg), triples(pj), gold_triples(gold)
        if g == j:
            stats["identical documents"] += 1
            continue
        stats["documents where they differ"] += 1
        for item in (g - j):            # greedy had it, joint dropped it
            by_task[(item[0], "greedy-only")] += 1
            stats["greedy-only RIGHT" if item in y else "greedy-only wrong"] += 1
        for item in (j - g):            # joint added it
            by_task[(item[0], "joint-only")] += 1
            stats["joint-only RIGHT" if item in y else "joint-only wrong"] += 1

    n_doc = stats["identical documents"] + stats["documents where they differ"]
    print(f"\n{len(texts)} documents, threshold {args.threshold}\n")
    print(f"  identical output          : {stats['identical documents']:>5} "
          f"({100*stats['identical documents']/n_doc if n_doc else 0:.1f}%)")
    print(f"  differ                    : {stats['documents where they differ']:>5} "
          f"({100*stats['documents where they differ']/n_doc if n_doc else 0:.1f}%)")
    gr, gw = stats["greedy-only RIGHT"], stats["greedy-only wrong"]
    jr, jw = stats["joint-only RIGHT"], stats["joint-only wrong"]
    print(f"\n  greedy-only items : {gr+gw:>5}   of which RIGHT {gr:>5} "
          f"({100*gr/(gr+gw) if gr+gw else 0:.1f}%)")
    print(f"  joint-only items  : {jr+jw:>5}   of which RIGHT {jr:>5} "
          f"({100*jr/(jr+jw) if jr+jw else 0:.1f}%)")
    print(f"\n  net for joint     : {jr-gr:+d} correct items")
    if by_task:
        print("\n  where they differ, by task:")
        for (task, side), n in sorted(by_task.items()):
            print(f"     {task:10s} {side:12s} {n:>5}")
    print("\n  INERT  = mostly identical -> nothing for a beam-aware loss to optimise.")
    print("  COIN FLIP = differs often, RIGHT rates similar -> the scores carry no")
    print("  information about which constraint-consistent assignment is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
