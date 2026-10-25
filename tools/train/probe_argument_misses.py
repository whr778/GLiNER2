"""Where do eb22d's gold arguments go on its own validation sets?

Each gold argument is one of four buckets. Strict micro F1 uses
(event_type, role, entity, trigger_key). Relaxed micro F1 drops trigger_key.

  strict_hit           the full key is predicted
  wrong_trigger        (type, role, entity) is predicted under another trigger key
                       sibling            that key is another gold event of the same type
                       coreferent_trigger the predicted triggers are a non-empty proper
                                          subset of this gold event's triggers
                       boundary           a predicted trigger contains or is contained
                                          in a gold trigger string of that type
                       false_trigger      none of those
  coreferent_surface   the entity string differs, and both strings sit in the same
                       coref_clusters cluster, on a prediction of the same type and role.
                       Counted only for documents that carry coref_clusters.
  never_proposed       none of the above

A coreferent_surface of zero on a sample with no clusters is "the field is absent",
and the script says so. Counts that do not partition the gold are a refusal.

Gold menu, so a missing event type is not an absent-menu miss. Threshold 0.1 is the
epoch-2 in-run sweep pick. WikiEvents and the news coref corpora are train_only in
eb22d and are not in this val.

    uv run tools/train/probe_argument_misses.py \\
        --config tools/train/config/base/eb22d.yaml \\
        --checkpoint /Volumes/Development/tmp/eb22d_epochs/checkpoint-epoch-2 \\
        --per-corpus 20 --out /Volumes/Development/tmp/issue5a/misses.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from gliner2.training.eval_metrics import (  # noqa: E402
    _gold_event_argument_set, _pred_event_argument_set, _schema_from_gold, load_with_overrides)

VAL = {
    "casie": "data/scaling_joint/casie.val.jsonl",
    "cmnee": "data/scaling_joint/cmnee.val.jsonl",
    "duee": "data/scaling_joint/duee.val.jsonl",
}
BUCKETS = ("strict_hit", "wrong_trigger", "coreferent_surface", "never_proposed")
WRONG = ("sibling", "coreferent_trigger", "boundary", "false_trigger")


def clusters_of(record: dict) -> dict[str, int]:
    """Lowercased mention text -> cluster index. A text in two clusters keeps the first."""
    out = {}
    for i, cluster in enumerate(record.get("coref_clusters") or []):
        if not isinstance(cluster, list):
            continue
        for mention in cluster:
            text = mention.get("text") if isinstance(mention, dict) else None
            if isinstance(text, str) and text.strip() and text.strip().lower() not in out:
                out[text.strip().lower()] = i
    return out


def classify(gold: set, pred: set, cluster: dict[str, int]) -> dict[str, int]:
    """Partition `gold`. Raises if a gold tuple is not exactly one bucket."""
    counts = {k: 0 for k in BUCKETS + WRONG}
    gold_keys = {}
    gold_triggers = {}
    for et, _role, _ent, key in gold:
        gold_keys.setdefault(et, set()).add(key)
        gold_triggers.setdefault(et, set()).update(key)
    for item in gold:
        et, role, ent, key = item
        if item in pred:
            counts["strict_hit"] += 1
            continue
        same_entity = [p for p in pred if p[:3] == (et, role, ent)]
        if same_entity:
            counts["wrong_trigger"] += 1
            pred_keys = {p[3] for p in same_entity}
            pred_triggers = {t.lower() for p in same_entity for t in p[3]}
            gold_tr = {t.lower() for t in gold_triggers.get(et, ())}
            if pred_keys & (gold_keys.get(et, set()) - {key}):
                counts["sibling"] += 1
            elif any(set(p[3]) < set(key) and set(p[3]) for p in same_entity):
                counts["coreferent_trigger"] += 1
            elif any(a in b or b in a for a in pred_triggers for b in gold_tr):
                counts["boundary"] += 1
            else:
                counts["false_trigger"] += 1
            continue
        gid = cluster.get(ent.strip().lower())
        if gid is not None and any(
            p[0] == et and p[1] == role and cluster.get(p[2].strip().lower()) == gid and p[2].strip().lower() != ent.strip().lower()
            for p in pred
        ):
            counts["coreferent_surface"] += 1
            continue
        counts["never_proposed"] += 1
    placed = sum(counts[k] for k in BUCKETS)
    if placed != len(gold) or sum(counts[k] for k in WRONG) != counts["wrong_trigger"]:
        raise RuntimeError(f"partition failed: gold {len(gold)} placed {placed} counts {counts}")
    return counts


def three(tp: int, pred_n: int, gold_n: int) -> tuple[float, float, float]:
    p = tp / pred_n if pred_n else 0.0
    r = tp / gold_n if gold_n else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb22d.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--per-corpus", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    fns = T._category_fns(T.load_labels_cfg(cfg, args.config))
    model = load_with_overrides(args.checkpoint, map_location=args.device).to(args.device).eval()
    result = {"threshold": args.threshold, "metric": "strict micro F1 and relaxed micro F1",
              "per_corpus": args.per_corpus, "seed": args.seed, "corpora": {}}
    for name, path in VAL.items():
        rows = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            raw = json.loads(line)
            rec = T.transform_record(raw, fns)
            events = (rec.get("output") or {}).get("events") or []
            if any(isinstance(e, dict) and e.get("arguments") for e in events):
                rows.append(rec)
        docs = random.Random(args.seed).sample(rows, min(args.per_corpus, len(rows)))
        total = {k: 0 for k in BUCKETS + WRONG}
        total.update(gold=0, pred=0, gold_relaxed=0, pred_relaxed=0, tp_relaxed=0,
                     docs_with_clusters=0, docs_with_multi_trigger=0)
        for rec in docs:
            cluster = clusters_of(rec)
            if cluster:
                total["docs_with_clusters"] += 1
            events = (rec.get("output") or {}).get("events") or []
            if any(isinstance(e, dict) and len([t for t in (e.get("triggers") or []) if isinstance(t, str) and t.strip()]) > 1
                   for e in events):
                total["docs_with_multi_trigger"] += 1
            gold = _gold_event_argument_set(rec["output"])
            pred_out = model.batch_extract_long(
                [rec["input"]], [_schema_from_gold(rec["output"])], batch_size=1,
                threshold=args.threshold, chunk_size=4096, chunk_overlap=0, global_decode=True)[0]
            pred = _pred_event_argument_set(pred_out)
            counts = classify(gold, pred, cluster)
            for k, v in counts.items():
                total[k] += v
            total["gold"] += len(gold)
            total["pred"] += len(pred)
            gu = {g[:3] for g in gold}
            pu = {p[:3] for p in pred}
            total["gold_relaxed"] += len(gu)
            total["pred_relaxed"] += len(pu)
            total["tp_relaxed"] += len(gu & pu)
        sp, sr, sf = three(total["strict_hit"], total["pred"], total["gold"])
        rp, rr, rf = three(total["tp_relaxed"], total["pred_relaxed"], total["gold_relaxed"])
        gold_n = total["gold"]
        shares = {k: total[k] / gold_n if gold_n else 0.0 for k in BUCKETS}
        row = {
            "docs": len(docs), "docs_available": len(rows), **{k: total[k] for k in list(BUCKETS) + list(WRONG)},
            "gold": gold_n, "pred": total["pred"],
            "docs_with_clusters": total["docs_with_clusters"],
            "docs_with_multi_trigger": total["docs_with_multi_trigger"],
            "shares_of_gold": shares,
            "strict_micro_f1": {"P": sp, "R": sr, "F1": sf},
            "relaxed_micro_f1": {"P": rp, "R": rr, "F1": rf},
            "coreferent_surface_readable": total["docs_with_clusters"] > 0,
        }
        result["corpora"][name] = row
        print(f"[miss] {name:8s} docs {len(docs):3d} gold {gold_n:5d}  "
              f"strict F1 {sf:.4f}  relaxed F1 {rf:.4f}  "
              f"hit {shares['strict_hit']:.1%}  wrong-trigger {shares['wrong_trigger']:.1%}  "
              f"coref-surface {shares['coreferent_surface']:.1%}  never {shares['never_proposed']:.1%}  "
              f"clusters {total['docs_with_clusters']}/{len(docs)}", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[miss] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
