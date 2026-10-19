"""Can the relation head fit a FEW training records? An overfit trace on the real training pipeline.

eb21's relation head scores the same on its own train docs as on val (strict F1 ~0.15-0.25 on the
two cleanest corpora, 0 on four others), and its val relation F1 sat at 0.000 -> 0.001 -> 0.085
over three epochs. This trains a checkpoint on N relation records of ONE corpus -- the trainer's
dataset, negatives, collator and loss -- and every --every steps decodes those same records
under their gold menu. A head that cannot fit 64 records has a structural defect; one that can
points at balance or training time in the full mix.

Logged per eval step: relation P/R/F1 (strict micro, exact head/tail text, No_Relation excluded)
at thresholds 0.3 and 0.05; and from the training forwards since the last eval: gold relation
pairs vs positive-labelled generated pairs (does the generator include the gold?), the mean logit
of positive and negative pairs, relation_loss and the total loss.

    uv run tools/train/overfit_relations.py --corpus sentence_rex --records 64 --steps 300 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
import gliner2.models.boundary.model as M  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.training.negatives import NegativeLabels  # noqa: E402
from gliner2.training.trainer import ExtractorCollator, ExtractorDataset  # noqa: E402

STATS = {}


def _hook_relation_loss():
    """Record gold pairs, positive-labelled pairs and pos/neg logits from each relation-loss call."""
    orig_loss, orig_bce = M.BoundaryExtractorModel._relation_loss, M.F.binary_cross_entropy_with_logits

    def rel_loss(self, batch, core, candidates, targets=None):
        et = getattr(targets, "edge_targets", None)
        STATS["gold"] = STATS.get("gold", 0) + (int(et[1].sum()) if isinstance(et, tuple) else 0)
        STATS["in_relation_loss"] = True
        try:
            return orig_loss(self, batch, core, candidates, targets)
        finally:
            STATS["in_relation_loss"] = False

    def bce(logits, labels, *a, **k):
        if STATS.get("in_relation_loss"):
            pos = labels > 0
            STATS["pos"] = STATS.get("pos", 0) + int(pos.sum())
            STATS["pairs"] = STATS.get("pairs", 0) + labels.numel()
            if pos.any():
                STATS.setdefault("pos_logit", []).append(float(logits[pos].detach().float().mean()))
            STATS.setdefault("neg_logit", []).append(float(logits[~pos].detach().float().mean()))
        return orig_bce(logits, labels, *a, **k)

    M.BoundaryExtractorModel._relation_loss = rel_loss
    M.F.binary_cross_entropy_with_logits = bce


def relation_prf(model, records, threshold):
    """Strict micro P/R/F1 on the records' own relation types (exact text, No_Relation excluded)."""
    tp = fp = fn = 0
    model.eval()
    with torch.no_grad():
        for r in records:
            gold = {(t, v["head"].strip(), v["tail"].strip())
                    for x in r["output"]["relations"] for t, v in x.items() if isinstance(v, dict) and t != "No_Relation"}
            types = sorted({g[0] for g in gold})
            if not types:
                continue
            out = model.batch_extract([r["input"]], {"relations": [{t: {"head": "", "tail": ""}} for t in types]},
                                      batch_size=1, threshold=threshold, include_spans=True)[0]
            pred = {(t, it["head"]["text"].strip(), it["tail"]["text"].strip())
                    for t, items in (out.get("relation_extraction") or {}).items() for it in items}
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    model.train()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9), tp + fp


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb21.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb21")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--records", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--task-lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--negative-pools", help="override the config's negative_pools (e.g. the committed eb21 JSON; `auto` reads every corpus)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tr, head = cfg["training"], cfg["model"]["boundary_head"]
    fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, args.config)), T.labels_passthrough(cfg)
    path = f"data/{args.corpus}.train.jsonl"
    T._fetch_if_missing(path)
    records = [r for r in T.read_transformed([path], fns, passthrough) if r["output"].get("relations")][:args.records]
    for r in records:
        r["_corpus"] = args.corpus
    print(f"[overfit] {args.corpus}: {len(records)} records, {sum(len(r['output']['relations']) for r in records)} gold relations "
          f"| steps {args.steps} batch {args.batch_size} lr enc {args.encoder_lr} task {args.task_lr} | device {args.device}", flush=True)

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu").to(args.device)
    negatives = None
    if tr.get("negative_pools") and tr.get("negative_labels_per_dim"):
        pools = args.negative_pools or tr["negative_pools"]
        if str(pools).strip().lower() == "auto":  # derived from this config's corpora, as train.py does
            pools = T._auto_negative_pools(cfg, args.config, str(Path(args.out).parent))
        negatives = NegativeLabels.load(pools, tr["negative_labels_per_dim"],
                                        seed=int(tr.get("negative_label_seed", 42)),
                                        partial=cfg["data"].get("partial_annotation"),
                                        gold_full_roles=bool(tr.get("negative_gold_full_roles", False)))
    ds = ExtractorDataset(data=records, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=negatives)
    coll = ExtractorCollator(model.processor, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)), on_capacity_exceeded="truncate_with_warning",
                             error_policy=tr.get("error_policy", "skip"), event_records=bool(head.get("event_records", False)))
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.encoder_lr}, {"params": rest, "lr": args.task_lr}])
    _hook_relation_loss()

    log, rng, t0 = [], random.Random(0), time.time()
    model.train()
    for step in range(args.steps + 1):
        if step % args.every == 0:
            row = {"step": step, "seconds": round(time.time() - t0)}
            for thr in (0.3, 0.05):
                p, r, f1, n = relation_prf(model, records, thr)
                row[f"@{thr}"] = {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4), "predicted": n}
            row["train"] = {"gold_pairs": STATS.get("gold", 0), "positive_pairs": STATS.get("pos", 0), "pairs": STATS.get("pairs", 0),
                            "pos_logit": round(sum(STATS.get("pos_logit", [0])) / max(len(STATS.get("pos_logit", [])), 1), 3),
                            "neg_logit": round(sum(STATS.get("neg_logit", [0])) / max(len(STATS.get("neg_logit", [])), 1), 3),
                            "relation_loss": round(sum(STATS.get("rel_loss", [0])) / max(len(STATS.get("rel_loss", [])), 1), 4),
                            "loss": round(sum(STATS.get("loss", [0])) / max(len(STATS.get("loss", [])), 1), 4)}
            log.append(row)
            print(f"[overfit] {json.dumps(row)}", flush=True)
            STATS.clear()
        if step == args.steps:
            break
        idx = rng.sample(range(len(ds)), min(args.batch_size, len(ds)))
        if negatives is not None:
            negatives.set_epoch(step)
        batch = coll([ds[i] for i in idx]).to(args.device)
        out = model(batch)
        STATS.setdefault("loss", []).append(float(out.loss.detach()))
        rl = out.losses.get("relation_loss")
        if rl is not None:
            STATS.setdefault("rel_loss", []).append(float(rl.detach()))
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("max_grad_norm", 1.0)))
        opt.step()
    Path(args.out).write_text(json.dumps({"args": vars(args), "log": log}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[overfit] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
