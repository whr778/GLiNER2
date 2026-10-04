"""Decode-time pooling: merge same-type instances whose triggers are boundary variants, then rescore.

A third of decoded arguments sit on the wrong trigger; ~14-26% of those on a boundary variant of
the gold trigger ("demanded" vs "demanded money"). Pooling merges same-type instances whose trigger
texts contain one another, unions their arguments, and keeps ONE trigger (a multi-trigger key would
never match single-trigger gold). Variants differ only in which trigger survives.

Decodes once (confidences on) and caches; variants are applied offline to the cache. GATE: the
'none' variant must reproduce measure_argument_key_gap.py's keyed numbers on the same docs.

    uv run python tools/train/eval_decode_pooling.py --checkpoint whr778/gliner2-eb18-balanced
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
from measure_argument_key_gap import CORPORA, classify  # noqa: E402
from gliner2.training.eval_metrics import (  # noqa: E402
    _gold_event_argument_set, _pred_event_argument_set, _schema_from_gold, load_with_overrides)


def text_of(t):
    return (t.get("text") if isinstance(t, dict) else t) or ""


def conf_of(t):
    return t.get("confidence", 0.0) if isinstance(t, dict) else 0.0


def pool(pred: dict, keep: str) -> dict:
    """Merge same-type instances whose (single) trigger texts contain one another."""
    if keep == "none":
        return pred
    out = dict(pred)
    block = {}
    for etype, insts in (pred.get("event_extraction") or {}).items():
        groups = []                                    # [triggers, arguments]
        for inst in insts:
            trig = list(inst.get("triggers") or [])
            t = text_of(trig[0]).lower() if trig else ""
            hit = next((g for g in groups if t and any((t in text_of(x).lower() or text_of(x).lower() in t)
                                                        for x in g[0])), None)
            if hit is None:
                groups.append([trig, list(inst.get("arguments") or [])])
            else:
                hit[0].extend(trig)
                hit[1].extend(inst.get("arguments") or [])
        merged = []
        for trig, args in groups:
            pick = max(trig, key=conf_of) if keep == "max_conf" else max(trig, key=lambda x: len(text_of(x)))
            seen, uniq = set(), []
            for a in args:
                k = (a.get("role") if isinstance(a, dict) else None, json.dumps(a, sort_keys=True, ensure_ascii=False))
                if k not in seen:
                    seen.add(k)
                    uniq.append(a)
            merged.append({"triggers": [pick] if trig else [], "arguments": uniq})
        block[etype] = merged
    out["event_extraction"] = block
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb18-balanced")
    ap.add_argument("--per-corpus", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--decode", default='{"threshold": 0.3, "chunk_size": 4096, "chunk_overlap": 0, "global_decode": true}')
    ap.add_argument("--cache", default=None, help="prediction cache (JSON); decoded and written if missing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fns = T._category_fns(T.load_labels_cfg(cfg, args.config))
    d = json.loads(args.decode)
    cache = Path(args.cache) if args.cache else None
    if cache and cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        model = load_with_overrides(args.checkpoint, map_location=args.device).to(args.device).eval()
        data = {}
        for name, path in CORPORA.items():
            recs = [T.transform_record(json.loads(l), fns) for l in open(path, encoding="utf-8") if l.strip()]
            recs = [r for r in recs if any(e.get("arguments") for e in r["output"].get("events") or [])]
            docs = random.Random(args.seed).sample(recs, min(args.per_corpus, len(recs)))
            golds = [{"events": r["output"]["events"]} for r in docs]
            preds = model.batch_extract_long([r["input"] for r in docs], [_schema_from_gold(g) for g in golds],
                                             batch_size=2, threshold=d["threshold"], chunk_size=d["chunk_size"],
                                             chunk_overlap=d["chunk_overlap"], global_decode=d["global_decode"],
                                             include_confidence=True)
            data[name] = {"gold": golds, "pred": preds}
        if cache:
            cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[pool] checkpoint {args.checkpoint} | decode {d}")
    for keep in ("none", "max_conf", "longest"):
        tot = Counter()
        for name, blob in data.items():
            c = Counter()
            for i, (g, p) in enumerate(zip(blob["gold"], blob["pred"])):
                gk = {(i,) + k for k in _gold_event_argument_set(g)}
                pk = {(i,) + k for k in _pred_event_argument_set(pool(p, keep))}
                c["gold"] += len(gk); c["pred"] += len(pk); c["tp"] += len(gk & pk)
                classify(gk, pk, c)
            tot.update(c)
        p_ = tot["tp"] / tot["pred"] if tot["pred"] else 0.0
        r_ = tot["tp"] / tot["gold"] if tot["gold"] else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        w = max(tot["wrong_trigger"], 1)
        print(f"[pool] {keep:9s} ALL keyed P {p_:.4f} R {r_:.4f} F1 {f_:.4f} (pred {tot['pred']}, tp {tot['tp']}) | wrong-trigger "
              f"{tot['wrong_trigger']}: boundary {tot['boundary']} ({100 * tot['boundary'] / w:.0f}%) sibling {tot['sibling']} "
              f"false {tot['false_trigger']}")


if __name__ == "__main__":
    main()
