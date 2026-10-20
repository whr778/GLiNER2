"""Can the EVENT record head fit a FEW training records? An overfit trace on the real training pipeline.

The relation head did not fit its own training data (TODO #40, `overfit_relations.py`). Event arguments carry the
same inflation: each occurrence of an argument's text becomes a separate REQUIRED value of a list role (measured
on eb21's collator: wikievents x3.29, cc_news_events_sonnet55_v2 x2.95, casie x1.70, cmnee x1.43). This trains a
checkpoint on N event records of ONE corpus -- the trainer's dataset, negatives, collator and loss -- and every
--every steps scores those same records with the in-run evaluator (`compute_metrics`, gold menu, global decode).

Logged per eval step: event_type / trigger / argument strict micro F1 at thresholds 0.3 and 0.1, and the mean
record_field_loss and total loss of the training forwards since the last eval.

    uv run tools/train/overfit_events.py --corpus wikievents --records 64 --steps 400 --device cuda --out wiki.json
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
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.training.eval_metrics import compute_metrics  # noqa: E402
from gliner2.training.negatives import NegativeLabels  # noqa: E402
from gliner2.training.trainer import ExtractorCollator, ExtractorDataset  # noqa: E402


def event_scores(model, records, threshold: float) -> dict:
    """Strict micro F1 for event_type / trigger / argument on the records' own gold menu (the in-run evaluator)."""
    model.eval()
    with torch.no_grad():
        m = compute_metrics(model, ExtractorDataset(records, shuffle=False, validate=False), batch_size=2,
                            threshold=threshold, report=False, chunk_size=4096, chunk_overlap=0, global_decode=True) or {}
    model.train()
    return {h: round(float(m.get(f"eval_event_{h}_strict_micro_f1", float("nan"))), 4)
            for h in ("type", "trigger", "argument")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb21.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb21")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--records", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--task-lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--negative-pools", help="override the config's negative_pools (`auto` reads every corpus)")
    ap.add_argument("--labels-file", help="override the config's labels_file")
    ap.add_argument("--eval-corpus", help="score against THIS corpus's records for the same documents (e.g. wikievents for "
                                          "wikievents_off): the evaluator matches TEXT and cannot read {text, start, end} gold")
    ap.add_argument("--from-scratch", action="store_true",
                    help="build the model from the config's `model` section (fresh heads) instead of --checkpoint")
    ap.add_argument("--head-override", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override boundary_head settings, e.g. boundary_marginal_loss=bce (one arm of an A/B)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for kv in args.head_override:
        k, v = kv.split("=", 1)
        assert k in cfg["model"]["boundary_head"], f"{k} is not a boundary_head key of {args.config}"
        cfg["model"]["boundary_head"][k] = yaml.safe_load(v)
    tr, head = cfg["training"], cfg["model"]["boundary_head"]
    if args.labels_file:
        cfg["labels_file"] = args.labels_file
    fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, args.config)), T.labels_passthrough(cfg)
    path = f"data/{args.corpus}.train.jsonl"
    T._fetch_if_missing(path)
    records = [r for r in T.read_transformed([path], fns, passthrough) if r["output"].get("events")][:args.records]
    for r in records:
        r["_corpus"] = args.corpus
    eval_records = records
    if args.eval_corpus:
        epath = f"data/{args.eval_corpus}.train.jsonl"
        T._fetch_if_missing(epath)
        by_text = {r["input"]: r for r in T.read_transformed([epath], fns, passthrough)}
        eval_records = [by_text[r["input"]] for r in records]   # KeyError = not the same documents
        print(f"[overfit-events] scoring against {args.eval_corpus}: {len(eval_records)} records, same texts", flush=True)
    n_args = sum(len(e.get("arguments") or []) for r in records for e in r["output"]["events"])
    print(f"[overfit-events] {args.corpus}: {len(records)} records, {sum(len(r['output']['events']) for r in records)} gold events, "
          f"{n_args} arguments | steps {args.steps} batch {args.batch_size} | device {args.device} | labels {cfg.get('labels_file')}", flush=True)

    if args.from_scratch:
        model = T._build_model(cfg["model"]).to(args.device)
    else:
        model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu").to(args.device)
    bh = getattr(model, "boundary_head", None)
    print(f"[overfit-events] model {'FROM SCRATCH (' + args.config + ')' if args.from_scratch else args.checkpoint} | "
          f"marginal loss {getattr(getattr(bh, 'settings', None), 'boundary_marginal_loss', '?')} | "
          f"overrides {args.head_override or 'none'}", flush=True)
    negatives = None
    if tr.get("negative_pools") and tr.get("negative_labels_per_dim"):
        negatives = NegativeLabels.load(args.negative_pools or tr["negative_pools"], tr["negative_labels_per_dim"],
                                        seed=int(tr.get("negative_label_seed", 42)), partial=cfg["data"].get("partial_annotation"),
                                        gold_full_roles=bool(tr.get("negative_gold_full_roles", False)))
    ds = ExtractorDataset(data=records, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=negatives)
    coll = ExtractorCollator(model.processor, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)),
                             on_capacity_exceeded="truncate_with_warning", error_policy=tr.get("error_policy", "skip"),
                             event_records=bool(head.get("event_records", False)))
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.encoder_lr}, {"params": rest, "lr": args.task_lr}])

    log, rng, t0, losses = [], random.Random(0), time.time(), {"record_field_loss": [], "loss": []}
    model.train()
    for step in range(args.steps + 1):
        if step % args.every == 0:
            row = {"step": step, "seconds": round(time.time() - t0)}
            for thr in (0.3, 0.1):
                row[f"@{thr}"] = event_scores(model, eval_records, thr)
            row["train"] = {k: round(sum(v) / max(len(v), 1), 4) for k, v in losses.items()}
            log.append(row)
            print(f"[overfit-events] {json.dumps(row)}", flush=True)
            losses = {k: [] for k in losses}
        if step == args.steps:
            break
        idx = rng.sample(range(len(ds)), min(args.batch_size, len(ds)))
        if negatives is not None:
            negatives.set_epoch(step)
        batch = coll([ds[i] for i in idx]).to(args.device)
        out = model(batch)
        losses["loss"].append(float(out.loss.detach()))
        if out.losses.get("record_field_loss") is not None:
            losses["record_field_loss"].append(float(out.losses["record_field_loss"].detach()))
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("max_grad_norm", 1.0)))
        opt.step()
    Path(args.out).write_text(json.dumps({"args": vars(args), "log": log}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[overfit-events] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
