"""Where does argument recall live? Score the same predictions WITH and WITHOUT the trigger key.

event_argument is scored on (event_type, role, entity, trigger_key), trigger_key being the event's
full sorted trigger set (eval_metrics._gold_event_argument_set). An argument found on the right
entity but attached to another mention (a sibling trigger, a boundary variant) is a miss. Scoring
again on (event_type, role, entity) only gives the recall that attaching arguments to the right
EVENT could recover: the upper bound for pooling / linking. A large gap -> recall lives in the
attachment; a small gap -> it lives in the scorer (arguments never scored on any mention).

Decodes at the checkpoint's own inference_defaults (or --decode), on argument-bearing val docs.

    uv run python tools/train/measure_argument_key_gap.py --checkpoint whr778/gliner2-eb18-balanced
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import train as T  # noqa: E402
from gliner2.training.eval_metrics import (  # noqa: E402
    _gold_event_argument_set, _pred_event_argument_set, _schema_from_gold, load_with_overrides)

CORPORA = {"casie": "data/casie.val.jsonl", "cmnee": "data/scaling_joint/cmnee.val.jsonl",
           "duee": "data/scaling_joint/duee.val.jsonl", "cc_news_events": "data/cc_news_events_haiku45.val.jsonl"}


def prf(gold: set, pred: set):
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb18-balanced")
    ap.add_argument("--per-corpus", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--decode", default=None, help="JSON decode settings overriding inference_defaults")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classify", action="store_true",
                    help="classify right-(type, role, entity)-wrong-trigger predictions: boundary / sibling / false trigger")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fns = T._category_fns(T.load_labels_cfg(cfg, args.config))
    model = load_with_overrides(args.checkpoint, map_location=args.device).to(args.device).eval()
    d = {**(model.config.inference_defaults or {}), **(json.loads(args.decode) if args.decode else {})}
    print(f"[gap] checkpoint {args.checkpoint} | decode {d}")
    totals = Counter()
    for name, path in CORPORA.items():
        recs = [T.transform_record(json.loads(l), fns) for l in open(path, encoding="utf-8") if l.strip()]
        recs = [r for r in recs if any(e.get("arguments") for e in r["output"].get("events") or [])]
        docs = random.Random(args.seed).sample(recs, min(args.per_corpus, len(recs)))
        gold_outs = [{"events": r["output"]["events"]} for r in docs]
        preds = model.batch_extract_long([r["input"] for r in docs], [_schema_from_gold(o) for o in gold_outs],
                                         batch_size=2, threshold=d["threshold"], chunk_size=d["chunk_size"],
                                         chunk_overlap=d["chunk_overlap"], global_decode=d["global_decode"])
        c = Counter()
        for i, (g, p) in enumerate(zip(gold_outs, preds)):
            gk = {(i,) + k for k in _gold_event_argument_set(g)}
            pk = {(i,) + k for k in _pred_event_argument_set(p)}
            if args.classify:
                classify(gk, pk, c)
            gu = {k[:4] for k in gk}
            pu = {k[:4] for k in pk}
            c["gold_k"] += len(gk); c["pred_k"] += len(pk); c["tp_k"] += len(gk & pk)
            c["gold_u"] += len(gu); c["pred_u"] += len(pu); c["tp_u"] += len(gu & pu)
        totals.update(c)
        report(name, c)
    report("ALL", totals)


def classify(gk: set, pk: set, c: Counter) -> None:
    """For predictions right on (type, role, entity) but wrong on trigger_key: which kind of wrong?

    boundary : the predicted trigger contains / is contained in a gold trigger of that type
    sibling  : the predicted trigger IS another gold event's trigger of that type
    false    : neither -- a non-gold trigger instance claimed the argument
    """
    gold_by_ure = {}
    for k in gk:
        gold_by_ure.setdefault(k[:4], set()).add(k)
    gold_triggers = {}
    for k in gk:
        gold_triggers.setdefault((k[0], k[1]), set()).update(k[4])
    gold_keys_by_type = {}
    for k in gk:
        gold_keys_by_type.setdefault((k[0], k[1]), set()).add(k[4])
    for k in pk - gk:
        if k[:4] not in gold_by_ure:
            continue
        c["wrong_trigger"] += 1
        pred = [t.lower() for t in k[4]]
        golds = [t.lower() for t in gold_triggers.get((k[0], k[1]), ())]
        if k[4] in gold_keys_by_type.get((k[0], k[1]), set()):
            c["sibling"] += 1
        elif any(a in b or b in a for a in pred for b in golds):
            c["boundary"] += 1
        else:
            c["false_trigger"] += 1


def report(name: str, c: Counter) -> None:
    def three(tp, pred, gold):
        p = tp / pred if pred else 0.0
        r = tp / gold if gold else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    pk, rk, fk = three(c["tp_k"], c["pred_k"], c["gold_k"])
    pu, ru, fu = three(c["tp_u"], c["pred_u"], c["gold_u"])
    print(f"[gap] {name:15s} gold {c['gold_k']:5d} | KEYED (today) P {pk:.4f} R {rk:.4f} F1 {fk:.4f} | "
          f"UNKEYED (type, role, entity) P {pu:.4f} R {ru:.4f} F1 {fu:.4f} | recall gap {ru - rk:+.4f}")
    if c["wrong_trigger"]:
        w = c["wrong_trigger"]
        print(f"[gap] {name:15s} wrong-trigger predictions {w}: boundary {c['boundary']} ({100 * c['boundary'] / w:.0f}%) | "
              f"sibling gold {c['sibling']} ({100 * c['sibling'] / w:.0f}%) | false trigger {c['false_trigger']} "
              f"({100 * c['false_trigger'] / w:.0f}%)")


if __name__ == "__main__":
    main()
