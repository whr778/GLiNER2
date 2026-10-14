"""Does a model read LABELS or their POSITION? Same model, same records, same widened menu; only the order of one
dimension's labels changes (entities, event types, or relation types).

    uv run tools/train/probe_label_order.py --model out/x/best --config tools/train/config/ab/menusched-const20.yaml
    uv run tools/train/probe_label_order.py ... --dimension events

Structures cannot be probed: widened menus never offer an absent structure.

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
HEAD = {"entities": "entity", "events": "event_type", "relations": "relation"}


def gold_names(gold: dict, dim: str) -> set:
    """The labels of ``dim`` this record's gold actually has."""
    if dim == "entities":
        return set(gold.get("entities") or {})
    if dim == "events":
        return {e.get("event_type") for e in gold.get("events") or [] if isinstance(e, dict)}
    return {n for r in gold.get("relations") or [] if isinstance(r, dict) for n in r}


def names(menu: dict, dim: str) -> set:
    """The labels of ``dim`` a menu offers."""
    return {next(iter(r)) for r in menu[dim]} if dim == "relations" else set(menu[dim])


def reorder(menu: dict, gold: dict, how: str, seed: int, dim: str = "entities") -> dict:
    """``menu`` with ``dim``'s labels in one order: absent labels first, gold first, or shuffled. Entities and events
    are ``{label: ...}`` dicts; relations a list of ``{name: {head, tail}}``."""
    items = [(next(iter(r)), r) for r in menu[dim]] if dim == "relations" else list(menu[dim].items())
    have = gold_names(gold, dim)
    present = [it for it in items if it[0] in have]
    absent = [it for it in items if it[0] not in have]
    order = {"absent_first": absent + present, "gold_first": present + absent,
             "shuffled": random.Random(seed).sample(items, len(items))}[how]
    return dict(menu, **{dim: [r for _, r in order] if dim == "relations" else dict(order)})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="local checkpoint dir or Hub repo id")
    ap.add_argument("--config", required=True, help="the training config (its test split, labels and decode settings)")
    ap.add_argument("--dimension", default="entities", choices=sorted(HEAD))
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
    keep = [i for i, m in enumerate(menus) if m.get(args.dimension)][:args.limit]
    recs, menus = [recs[i] for i in keep], [menus[i] for i in keep]
    absent = sum(len(names(m, args.dimension) - gold_names(r["output"], args.dimension)) for r, m in zip(recs, menus))
    print(f"[order] {args.model}: {len(recs)} {args.split} records offering {args.dimension} under {args.menu} ({rep}), "
          f"{absent} absent {args.dimension} labels offered, threshold {args.threshold}")
    for how in ORDERS:
        ms = [reorder(m, r["output"], how, i, args.dimension) for i, (m, r) in enumerate(zip(menus, recs))]
        preds = model.batch_extract_long([r["input"] for r in recs], ms, batch_size=args.batch_size, threshold=args.threshold,
                                         chunk_size=ev["chunk_size"], chunk_overlap=ev["chunk_overlap"],
                                         global_decode=ev["global_decode"], global_decode_config=ev["global_decode_config"])
        s = score_predictions([r["output"] for r in recs], preds, report=False)
        g = lambda h, k: s.get(f"eval_{h}_strict_micro_{k}", 0.0)
        h, other = HEAD[args.dimension], ("event_type" if args.dimension == "entities" else "entity")
        print(f"[order] {how:12}  {h} strict F1 {g(h, 'f1'):.3f} (P {g(h, 'precision'):.3f} R {g(h, 'recall'):.3f})"
              f"  |  {other} strict F1 {g(other, 'f1'):.3f}", flush=True)


if __name__ == "__main__":
    main()
