"""Do a trigger's ARGUMENTS vouch for it? Argument evidence of gold vs false trigger candidates, eb18.

THE ASYMMETRY ("the King of Sweden"). In natural-mode event records every argument is assigned
CONDITIONED ON its trigger (assign logits use the instance state the trigger seeded), but whether
the instance exists is decided by the trigger's span score ALONE, gated before any argument is
considered (records.py decode_group), and the natural-mode object loss is zero. The subjects
know the king; the king does not know his subjects. The probe found ~31% of gold triggers are
decoded and then cut by the gate. If gold triggers in that band carry clearly stronger argument
evidence than false triggers, a gate that consults the arguments recovers recall AND precision.

Decode-only on the real path: trigger and argument gates lowered to --floor, confidences on,
every type offered its FULL role set from the corpus taxonomy (not only the roles its gold
happens to use, which would give gold and false triggers different menus). Per decoded
instance: trigger confidence, its arguments' confidences, and whether (type, trigger) is gold.

    uv run python tools/train/trace_argument_evidence.py --per-corpus 100
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import train as T  # noqa: E402
from gliner2.training.eval_metrics import _gold_event_trigger_set, apply_boundary_overrides, load_with_overrides  # noqa: E402

# validation file -> the TRAIN corpus whose taxonomy pool holds its event roles
CORPORA = {"casie": ("data/casie.val.jsonl", "casie"),
           "cmnee": ("data/scaling_joint/cmnee.val.jsonl", "cmnee_ner"),
           "duee": ("data/scaling_joint/duee.val.jsonl", "duee_ner"),
           "cc_news_events": ("data/cc_news_events_haiku45.val.jsonl", "cc_news_events_haiku45")}
BANDS = ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01))


def confidences(node) -> list:
    """Every argument confidence in an instance's `arguments`, whatever its nesting."""
    if isinstance(node, dict):
        out = [node["confidence"]] if isinstance(node.get("confidence"), (int, float)) else []
        return out + [c for k, v in node.items() if k != "confidence" for c in confidences(v)]
    if isinstance(node, list):
        return [c for v in node for c in confidences(v)]
    return []


def auc(pos: list, neg: list) -> float:
    """P(gold evidence > false evidence), ties half: the separation the gate could exploit."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb18-balanced")
    ap.add_argument("--per-corpus", type=int, default=100)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fns = T._category_fns(T.load_labels_cfg(cfg, args.config))
    spec = importlib.util.spec_from_file_location("bnp", ROOT / "tools/data/build_negative_pools.py")
    bnp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bnp)
    pools = bnp.build_pools(cfg, Path(args.config), log=lambda *a: None)
    model = load_with_overrides(args.checkpoint, map_location=args.device).to(args.device).eval()
    f = args.floor
    apply_boundary_overrides(model, {"record_anchor_threshold": f, "record_anchor_proposal_threshold": f,
                                     "record_anchor_threshold_wins": True,
                                     "record_field_threshold": f, "record_field_threshold_wins": True})
    rows = []
    for name, (path, pool_name) in CORPORA.items():
        roles = (pools.get(pool_name) or {}).get("events") or {}
        recs = [T.transform_record(json.loads(l), fns) for l in open(path, encoding="utf-8") if l.strip()]
        recs = [r for r in recs if r["output"].get("events")]
        docs = random.Random(args.seed).sample(recs, min(args.per_corpus, len(recs)))
        schemas = [{"events": {e["event_type"]: list(roles.get(e["event_type"], [])) for e in r["output"]["events"]}}
                   for r in docs]
        preds = model.batch_extract_long([r["input"] for r in docs], schemas, batch_size=2, threshold=f,
                                         chunk_size=4096, chunk_overlap=0, global_decode=True, include_confidence=True)
        for r, pred in zip(docs, preds):
            gold = _gold_event_trigger_set({"events": r["output"]["events"]})
            for etype, insts in (pred.get("event_extraction") or {}).items():
                for inst in insts:
                    for trig in inst.get("triggers") or []:
                        arg_conf = sorted(confidences(inst.get("arguments")), reverse=True)
                        rows.append({"corpus": name, "type": etype, "trigger_conf": trig["confidence"],
                                     "gold": (etype, trig["text"].strip()) in gold, "arg_conf": arg_conf})
        print(f"[evidence] {name}: {len(docs)} docs, {sum(1 for x in rows if x['corpus'] == name)} trigger candidates "
              f"(roles offered for {len(roles)} types)", flush=True)
    report(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def report(rows: list) -> None:
    feats = {"max_arg": lambda r: r["arg_conf"][0] if r["arg_conf"] else 0.0,
             "n_args>=0.5": lambda r: sum(c >= 0.5 for c in r["arg_conf"]),
             "sum_top3": lambda r: sum(r["arg_conf"][:3])}
    for corpus in sorted({r["corpus"] for r in rows}) + ["ALL"]:
        sub = [r for r in rows if corpus in ("ALL", r["corpus"])]
        print(f"[evidence] --- {corpus}")
        print(f"[evidence] {'trigger band':14s} {'gold':>5s} {'false':>6s} | " +
              " | ".join(f"{k}: gold/false mean, AUC" for k in feats))
        for lo, hi in BANDS:
            band = [r for r in sub if lo <= r["trigger_conf"] < hi]
            g = [r for r in band if r["gold"]]
            n = [r for r in band if not r["gold"]]
            cells = []
            for k, fn in feats.items():
                gv, nv = [fn(r) for r in g], [fn(r) for r in n]
                mean = lambda v: sum(v) / len(v) if v else float("nan")
                cells.append(f"{mean(gv):5.2f}/{mean(nv):5.2f} {auc(gv, nv):4.2f}")
            gate = "below gate" if hi <= 0.3 else "above gate"
            print(f"[evidence] [{lo:.2f},{min(hi, 1):.2f}) {gate[:5]} {len(g):5d} {len(n):6d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
