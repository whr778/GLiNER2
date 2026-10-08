"""Does a model read entity LABELS or their POSITION? Same model, same records, same widened menu; only the order of
the entity labels changes.

    uv run tools/train/probe_label_order.py --model out/x/best --config tools/train/config/ab/menusched-const20.yaml

Training before be581a1 put gold entity labels first and appended the absent ones, and models learned the position:
whr778/gliner2-menudose-treatment scored entity strict F1 0.000 absent-first, 0.646 gold-first, 0.231 shuffled
(60 sonnet55 test docs, widened:20). A model that reads the labels scores the three orders alike (within ~0.02).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

ORDERS = ("absent_first", "gold_first", "shuffled")


def reorder(menu: dict, gold: dict, how: str, seed: int) -> dict:
    """``menu`` with its entity labels in one order: absent labels first, gold first, or shuffled."""
    ents = list(menu["entities"])
    present = [e for e in ents if e in (gold.get("entities") or {})]
    absent = [e for e in ents if e not in present]
    order = {"absent_first": absent + present, "gold_first": present + absent,
             "shuffled": random.Random(seed).sample(ents, len(ents))}[how]
    return dict(menu, entities={e: menu["entities"][e] for e in order})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="local checkpoint dir or Hub repo id")
    ap.add_argument("--config", required=True, help="the training config (its test split, labels and decode settings)")
    ap.add_argument("--menu", default="widened:20", help="widened:K or corpus_full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)

    from gliner2 import AutoExtractor
    from gliner2.training.eval_metrics import score_predictions

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ev = T._parse_eval_settings(cfg, args.config, corpus_data=[])
    model = AutoExtractor.from_pretrained(args.model)
    recs, menus, rep = T.menu_split(cfg, args.config, args.menu, args.split)
    recs, menus = recs[:args.limit], menus[:args.limit]
    print(f"[order] {args.model}: {len(recs)} {args.split} records under {args.menu} ({rep}), threshold {args.threshold}")
    for how in ORDERS:
        ms = [reorder(m, r["output"], how, i) for i, (m, r) in enumerate(zip(menus, recs))]
        preds = model.batch_extract_long([r["input"] for r in recs], ms, batch_size=args.batch_size, threshold=args.threshold,
                                         chunk_size=ev["chunk_size"], chunk_overlap=ev["chunk_overlap"],
                                         global_decode=ev["global_decode"], global_decode_config=ev["global_decode_config"])
        s = score_predictions([r["output"] for r in recs], preds, report=False)
        g = lambda h, k: s.get(f"eval_{h}_strict_micro_{k}", 0.0)
        print(f"[order] {how:12}  entity strict F1 {g('entity', 'f1'):.3f} (P {g('entity', 'precision'):.3f} "
              f"R {g('entity', 'recall'):.3f})  |  event_type strict F1 {g('event_type', 'f1'):.3f}", flush=True)


if __name__ == "__main__":
    main()
