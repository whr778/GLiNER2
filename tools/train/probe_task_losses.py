"""Where does the boundary loss actually go? Per-task shares of start/end/pair.

The phase-3 flat-weight sweep (`task_loss_weights`) was null on every metric: doses
of 0.5/1.0/2.0/4.0 on the event queries moved nothing past the run-to-run floor. The
next lever is a per-task `pos_weight`, but choosing its dose blind repeats the same
mistake -- so measure first.

`reduce_by_task` splits the query-conditioned terms (start, end, pair) into per-task
CONTRIBUTIONS that sum back to the unweighted term. Two numbers come out of that:

    share       how much of the gradient events already hold. If events are 3% of
                the loss, a 2x flat weight was never going to be visible.
    pos/neg     `pos_weight` scales ONLY the positive part, so a dose k multiplies a
                task's contribution by (k*pos + neg) / (pos + neg). With boundary
                targets this sparse, neg dominates and k=4 may buy almost nothing --
                which is exactly what has to be known before spending a GPU on it.

Reported honestly: only start/end/pair are bucketed. The single-task terms
(relation, record) are attributed by construction, and the remainder is printed as
unattributed rather than folded into a task.

Runs the TRAINING path -- train mode, the training collator, the scheduled gold
injection -- because that is the loss being characterised, not the eval loss.

    uv run python tools/train/probe_task_losses.py \
        --config tools/train/config/warmstart-natural.yaml \
        --checkpoint out/event-loss-sweep/warmstart-natural-seed43/final \
        --batches 60 --gold-injection 0.25
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from gliner2.models.boundary.model import TASK_TYPES
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from tools.train.train import _build_model, _event_split, _split_files

TERMS = ("start", "end", "pair", "inside")

# The terms `reduce_by_task` cannot split: they are listwise over candidates, or
# per-query scalars, not elementwise over (query, position). Printed by name with
# their weights rather than lumped into "unattributed", because they are what a
# `pos_weight` on the BCE terms would NOT touch.
UNBUCKETED = (
    ("soft_iou_loss", "soft_iou_aux_weight"),
    ("rerank_listwise_loss", "rerank_listwise_weight"),
    ("proposal_loss", "proposal_loss_weight"),
    ("consistency_loss", "consistency_loss_weight"),
    ("abstention_loss", "abstention_loss_weight"),
    ("count_loss", "count_loss_weight"),
)


def build(args):
    """Model + dataloader on the training path, with the buckets switched on."""
    cfg = yaml.safe_load(Path(args.config).read_text())

    model_cfg = dict(cfg["model"])
    if args.checkpoint:
        model_cfg["pretrained"] = args.checkpoint
    model_cfg["map_location"] = args.device
    head = dict(model_cfg.get("boundary_head") or {})
    head["report_task_losses"] = True
    model_cfg["boundary_head"] = head
    model = _build_model(model_cfg)

    train_cfg = dict(cfg["training"])
    train_cfg.update(
        output_dir=args.output_dir,
        max_train_samples=args.records,
        num_workers=0,
        bf16=False,
        fp16=False,
        batch_size=args.batch_size,
    )
    config = TrainingConfig(**train_cfg)

    trainer = GLiNER2Trainer(model, config)
    trainer.device = torch.device(args.device)
    model.to(args.device)

    data = cfg.get("data") or {}
    files = _split_files(data.get("corpora") or [], "train") + _event_split(
        data.get("event_files") or {}, "train"
    )
    dataset = trainer._prepare_data(files, is_train=True)
    loader = trainer._create_dataloader(
        dataset, args.batch_size, shuffle=False, is_training=True
    )
    return model, loader


def accumulate(model, loader, args):
    """Sum every loss term over N batches, checking each batch reconciles."""
    model.train()
    totals = defaultdict(float)
    batches = 0
    drift = 0.0

    for batch in loader:
        if batches >= args.batches:
            break
        with torch.no_grad():
            out = model(batch, gold_injection_prob=args.gold_injection)
        losses = dict(out.losses or {})
        if not losses:
            continue
        # Two different totals share the name `total_loss`: the boundary head's
        # (start..count) inside `losses`, and the model's combined total (that plus
        # classification/record/relation) on the output. Summing both into one bucket
        # doubles the denominator and halves every share -- keep them apart.
        boundary_total = losses.pop("total_loss", None)
        for key, value in losses.items():
            totals[key] += float(value)
        if boundary_total is not None:
            totals["boundary_total_loss"] += float(boundary_total)
        if out.total_loss is not None:
            totals["total_loss"] += float(out.total_loss)

        # The buckets are decoration unless they reconcile against the scalar the
        # optimizer sees. Check every batch, report the worst.
        for term in TERMS:
            if f"{term}_loss" not in losses:
                continue
            bucketed = sum(float(losses[f"{t}_{term}_loss"]) for t in TASK_TYPES)
            whole = float(losses[f"{term}_loss"])
            drift = max(drift, abs(bucketed - whole) / max(abs(whole), 1e-9))
        batches += 1

    return totals, batches, drift


def report(model, totals, batches, drift, args):
    weights = model.boundary_head.loss_weights
    n = max(batches, 1)
    mean = {k: v / n for k, v in totals.items()}

    print(f"\ncheckpoint: {args.checkpoint or 'from config'}")
    print(f"batches: {batches} x {args.batch_size}   gold_injection: {args.gold_injection}")
    print(f"bucket reconciliation: worst relative drift {drift:.2e}")

    print("\n=== per-task contribution to each bucketed term (mean per batch)")
    header = f"{'term':<10}{'weight':>8}" + "".join(f"{t:>16}" for t in TASK_TYPES) + f"{'term total':>14}"
    print(header)
    weighted = defaultdict(float)
    bucketed_total = 0.0
    for term in TERMS:
        if f"{term}_loss" not in mean:
            continue
        w = float(weights.get(term, 0.5 if term == "inside" else 1.0))
        row = f"{term:<10}{w:>8.2f}"
        for task in TASK_TYPES:
            share = mean[f"{task}_{term}_loss"]
            weighted[task] += w * share
            row += f"{share:>16.5f}"
        row += f"{mean[f'{term}_loss']:>14.5f}"
        bucketed_total += w * mean[f"{term}_loss"]
        print(row)

    print("\n=== positive vs negative mass inside each term (mean per batch)")
    print(f"{'term':<10}{'task':<16}{'positive':>12}{'negative':>12}{'pos frac':>10}")
    for term in TERMS:
        if f"{term}_loss" not in mean:
            continue
        for task in TASK_TYPES:
            total = mean[f"{task}_{term}_loss"]
            pos = mean[f"{task}_{term}_pos_loss"]
            if total <= 0:
                continue
            print(f"{term:<10}{task:<16}{pos:>12.5f}{total - pos:>12.5f}{pos / total:>10.3f}")

    print("\n=== terms the buckets cannot split (weighted contribution)")
    settings = model.boundary_head.settings
    unbucketed = 0.0
    for key, weight_attr in UNBUCKETED:
        if key not in mean:
            continue
        w = float(getattr(settings, weight_attr, 0.0))
        contribution = w * mean[key]
        unbucketed += contribution
        print(f"  {key:<24}{'w=' + format(w, '.3f'):>12}{mean[key]:>12.5f}"
              f" -> {contribution:>10.5f}")
    print(f"  {'classification_loss':<24}{'':>12}{mean.get('classification_loss', 0.0):>12.5f}")

    print("\n=== gradient share")
    # The record head supervises EVENT records too when `event_records` is on
    # (records.py: is_event = task_type == "events"), in which case this line folds
    # event supervision into the wrong task. Off by default -- assert rather than assume.
    event_records = bool(getattr(settings, "event_records", False))
    single = {
        "relations": mean.get("relation_loss", 0.0),
        "json_structures": (
            mean.get("record_object_loss", 0.0) + mean.get("record_field_loss", 0.0)
        ),
    }
    if event_records:
        print("  NOTE: event_records=True -- record loss below mixes events into "
              "json_structures and the event share is a LOWER BOUND.")
    grand = mean.get("total_loss", 0.0)
    attributed = sum(weighted.values()) + sum(single.values())
    print(f"{'task':<18}{'bucketed':>12}{'single-task':>14}{'attributed':>12}{'of total':>10}")
    for task in TASK_TYPES:
        got = weighted[task] + single.get(task, 0.0)
        print(f"{task:<18}{weighted[task]:>12.5f}{single.get(task, 0.0):>14.5f}"
              f"{got:>12.5f}{got / max(grand, 1e-9):>10.1%}")
    task_blind = unbucketed + mean.get("classification_loss", 0.0)
    print(f"{'task-blind terms':<18}{'':>12}{'':>14}{task_blind:>12.5f}"
          f"{task_blind / max(grand, 1e-9):>10.1%}")
    print(f"{'TOTAL':<18}{'':>12}{'':>14}{grand:>12.5f}")
    print(f"  residual (should be ~0): {grand - attributed - task_blind:+.6f}")
    print(f"  output_dir: {args.output_dir}")

    print("\n=== queries per batch")
    for task in TASK_TYPES:
        print(f"  {task:<18}{mean.get(f'{task}_query_count', 0.0):>10.2f}")

    print("\n=== pos_weight dose response on events: (k*pos + neg) / (pos + neg)")
    pos = sum(
        float(weights.get(t, 0.5 if t == "inside" else 1.0))
        * mean.get(f"events_{t}_pos_loss", 0.0)
        for t in TERMS
    )
    neg = weighted["events"] - pos
    print(f"  event bucketed loss {weighted['events']:.5f} = pos {pos:.5f} + neg {neg:.5f}")
    for k in (2.0, 4.0, 8.0, 16.0, 32.0):
        factor = (k * pos + neg) / max(pos + neg, 1e-9)
        print(f"  pos_weight {k:>5.1f} -> event term x{factor:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="tools/train/config/warmstart-natural.yaml")
    ap.add_argument("--checkpoint", default=None, help="overrides model.pretrained")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--records", type=int, default=2000, help="corpus records to load")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--gold-injection", type=float, default=1.0,
        help="mirror the trainer's schedule: 1.0 at init, 0.25 at the end of training",
    )
    ap.add_argument("--output-dir", default="out/probe_task_losses")
    args = ap.parse_args()

    # Both generators: the processor samples schemas with Python's `random`, and the
    # negative-query sampler draws with torch. Seeding only one leaves the probe
    # nondeterministic (the same defect the gate-1 harness had).
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, loader = build(args)
    totals, batches, drift = accumulate(model, loader, args)
    if not batches:
        raise SystemExit("no batches produced a loss -- is the corpus present?")
    report(model, totals, batches, drift, args)


if __name__ == "__main__":
    main()
