"""Can the classification head fit a FEW training records, and what does schema dropout cost it? Overfit trace on the
real training pipeline (dataset, collator with its sampling, model, loss), mirroring tools/train/overfit_relations.py.

Trains a checkpoint on N classification records of ONE corpus and every --every steps classifies those same records
under their gold menu (argmax over the menu; accuracy and macro F1). Logs, from the training collates since the last
eval: records whose true label was dropped from the menu (all-zero gold), synthetic-name records, and the menu size.

    uv run tools/train/overfit_classification.py --config tools/train/config/base/eb22c-ddp-trace.yaml \
        --checkpoint <dir> --corpus masakhanews_eng --records 64 --steps 400 --out out.json [--no-schema-dropout]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.training import ExtractorCollator  # noqa: E402
from gliner2.training.trainer import ExtractorDataset, apply_schema_dropout  # noqa: E402

STATS: dict = defaultdict(int)


def classification_metrics(model, records):
    """Accuracy and macro F1 of argmax-over-menu classification on the records' own menus (eval mode)."""
    model.eval(); correct = 0; tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    with torch.no_grad():
        for r in records:
            cls = r["output"]["classifications"][0]
            gold = cls["true_label"][0] if isinstance(cls["true_label"], list) else cls["true_label"]
            out = model.batch_extract([r["input"]], {"classifications": [{"task": cls["task"], "labels": list(cls["labels"])}]},
                                      batch_size=1, threshold=0.0)[0]
            # the runtime returns {task: label} (or {task: [labels]} for multi-label) at the top level
            p = out.get(cls["task"]) if isinstance(out, dict) else None
            p = p[0] if isinstance(p, list) and p else p
            correct += p == gold
            if p == gold: tp[gold] += 1
            else: fp[p] += 1; fn[gold] += 1
    model.train()
    labels = set(tp) | set(fp) | set(fn) - {None}
    f1s = [2 * tp[l] / max(2 * tp[l] + fp[l] + fn[l], 1) for l in labels]
    return correct / len(records), sum(f1s) / max(len(f1s), 1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tools/train/config/base/eb22c-ddp-trace.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--records", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--task-lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-schema-dropout", action="store_true", help="zero every SamplingConfig *_prob (training.schema_dropout: false)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8")); tr, head = cfg["training"], cfg["model"]["boundary_head"]
    path = f"data/{args.corpus}.train.jsonl"
    T._fetch_if_missing(path)        # a box only carries the config's corpora; these classification sets are not in eb22c's mix
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    records = [r for r in rows if r["output"].get("classifications")][:args.records]
    for r in records: r["_corpus"] = args.corpus
    print(f"[overfit-cls] {args.corpus}: {len(records)} records | steps {args.steps} batch {args.batch_size} | device {args.device}", flush=True)
    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu")
    sc = model.processor.sampling_config
    if args.no_schema_dropout:
        changed = apply_schema_dropout(sc, False); print(f"[overfit-cls] schema dropout OFF: zeroed {sorted(changed)}", flush=True)
    else:
        print(f"[overfit-cls] schema dropout ON: remove_classification_label_prob {sc.remove_classification_label_prob} synthetic_label_prob {sc.synthetic_label_prob} include_true_label_prob {sc.include_true_label_prob}", flush=True)
    model = model.to(args.device)
    ds = ExtractorDataset(data=records, max_samples=-1, shuffle=False, seed=42, validate=False)
    coll = ExtractorCollator(model.processor, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)), on_capacity_exceeded="truncate_with_warning",
                             error_policy=tr.get("error_policy", "skip"), event_records=bool(head.get("event_records", False)))
    tok = model.processor.tokenizer
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.encoder_lr}, {"params": rest, "lr": args.task_lr}])
    log, rng, t0 = [], random.Random(0), time.time()
    model.train()
    for step in range(args.steps + 1):
        if step % args.every == 0:
            acc, f1 = classification_metrics(model, records)
            n = max(STATS["records"], 1)
            row = {"step": step, "seconds": round(time.time() - t0), "acc": round(acc, 4), "macro_f1": round(f1, 4),
                   "train": {"records": STATS["records"], "true_absent": round(STATS["true_absent"] / n, 3), "synthetic": round(STATS["synthetic"] / n, 3),
                             "menu_size": round(STATS["menu"] / n, 2), "cls_loss": round(STATS["cls_loss"] / max(STATS["batches"], 1), 4)}}
            log.append(row); print(f"[overfit-cls] {json.dumps(row)}", flush=True); STATS.clear()
        if step == args.steps:
            break
        idx = rng.sample(range(len(ds)), min(args.batch_size, len(ds)))
        batch = coll([ds[i] for i in idx]).to(args.device)
        for i in range(len(idx)):
            vec = batch.structure_labels[i][0]
            prompt = tok.decode(batch.input_ids[i][:160]).split("[SEP_TEXT]")[0]
            STATS["records"] += 1; STATS["true_absent"] += (sum(vec) == 0); STATS["synthetic"] += ("label 1" in prompt or "label 2" in prompt); STATS["menu"] += len(vec)
        out = model(batch)
        cl = out.losses.get("classification_loss")
        if cl is not None: STATS["cls_loss"] += float(cl.detach()); STATS["batches"] += 1
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("max_grad_norm", 1.0))); opt.step()
    Path(args.out).write_text(json.dumps({"args": vars(args), "log": log}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[overfit-cls] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
