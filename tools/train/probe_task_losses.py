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
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from gliner2.models.boundary.model import TASK_TYPES
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from tools.train.train import _build_model, _event_split, _split_files

# Every query-typed term, with where its weight comes from and whether
# `task_loss_weights` currently reaches it. `reached` is the measurement that
# explains the null flat sweep: the weight only ever touched start/end/pair.
TERMS = (
    # name        scalar-weight source            loss key            reached
    ("start",     None,                           "start_loss",         True),
    ("end",       None,                           "end_loss",           True),
    ("pair",      None,                           "pair_loss",          True),
    ("inside",    None,                           "inside_loss",        False),
    ("soft_iou",  "soft_iou_aux_weight",          "soft_iou_loss",      False),
    ("rerank",    "rerank_listwise_weight",       "rerank_listwise_loss", False),
    ("proposal",  "proposal_loss_weight",         "proposal_loss",      False),
    ("abstention", "abstention_loss_weight",      "abstention_loss",    False),
    # CAVEAT: count is Poisson NLL with full=False, i.e. exp(x) - t*x, which is
    # unbounded below. A per-task contribution here can legitimately be NEGATIVE,
    # and upweighting a task whose count term is negative LOWERS the reported loss
    # while its gradient still scales correctly. The contributions still sum to the
    # term (the reconciliation holds), but read this row as "signed contribution",
    # not "share of gradient".
    ("count",     "count_loss_weight",            "count_loss",         False),
)

# Not query-typed, so no per-task split is definable. Reported so the total closes.
NOT_QUERY_TYPED = (
    ("consistency_loss", "consistency_loss_weight"),
)

# `w.get(term, ...)` defaults inside _compute_losses for the four mechanism terms.
MECHANISM_DEFAULTS = {"start": 1.0, "end": 1.0, "pair": 1.0, "inside": 0.5}

# Where task_pos_weights is actually threaded. NOT soft_iou (fractional targets) and
# NOT abstention (its positive is "query is absent", which a dose would invert rather
# than sharpen) -- both HAVE a positive term, so including them would silently
# overstate the dose.
POS_WEIGHT_TERMS = frozenset({"start", "end", "pair", "inside"})


def term_weight(term, source, model):
    """The scalar multiplying this term inside the head's `total`."""
    if source is None:
        return float(model.boundary_head.loss_weights.get(term, MECHANISM_DEFAULTS[term]))
    return float(getattr(model.boundary_head.settings, source, 0.0))


def sample_records(files, args):
    """A seeded RANDOM sample of the training records, not the first N.

    The trainer caps with ``max_train_samples`` *before* its post-chunk shuffle, so
    head-N follows file order. On a pre-shuffled mix that is harmless; on a config
    whose ``corpora``/``event_files`` are separate files it is not -- head-200 of
    the cold-start mix is 100% relation records and 0% events, which would make the
    measured composition an artefact of file ordering.

    Lines are sampled as raw strings and only the survivors are parsed, so the peak
    cost is the corpus bytes rather than the parsed objects.
    """
    lines = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            lines.extend(line for line in handle if line.strip())
    rng = random.Random(args.seed)
    rng.shuffle(lines)
    if args.records and args.records > 0:
        lines = lines[: args.records]
    print(f"[probe] sampled {len(lines)} records from {len(files)} file(s), seed {args.seed}")
    return [json.loads(line) for line in lines]


def build(args):
    """Model + dataloader on the training path, with the buckets switched on."""
    cfg = yaml.safe_load(Path(args.config).read_text())

    model_cfg = dict(cfg["model"])
    if args.checkpoint:
        model_cfg["pretrained"] = args.checkpoint
    if "pretrained" in model_cfg:
        model_cfg["map_location"] = args.device
    else:
        # Cold-start config (`encoder:`): from_encoder takes no map_location, and
        # flash_attention_2 has no CPU kernel. Fall back to sdpa -- the documented
        # bf16 non-finite hazard with sdpa does not apply here because the probe
        # runs fp32 and never steps the optimizer.
        model_cfg.pop("map_location", None)
        if model_cfg.get("attn_implementation") == "flash_attention_2":
            model_cfg["attn_implementation"] = "sdpa"
            print("[probe] cold-start path: flash_attention_2 -> sdpa (fp32, no steps)")
    head = dict(model_cfg.get("boundary_head") or {})
    head["report_task_losses"] = True
    model_cfg["boundary_head"] = head
    model = _build_model(model_cfg)

    train_cfg = dict(cfg["training"])
    train_cfg.update(
        output_dir=args.output_dir,
        max_train_samples=-1,   # sample_records already capped, and by seed not by file order
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
    dataset = trainer._prepare_data(sample_records(files, args), is_train=True)
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
        for term, _, loss_key, _ in TERMS:
            if f"{TASK_TYPES[0]}_{term}_loss" not in losses:
                continue
            bucketed = sum(float(losses[f"{t}_{term}_loss"]) for t in TASK_TYPES)
            whole = float(losses[loss_key])
            drift = max(drift, abs(bucketed - whole) / max(abs(whole), 1e-9))
        batches += 1

    return totals, batches, drift


def report(model, totals, batches, drift, args):
    n = max(batches, 1)
    mean = {k: v / n for k, v in totals.items()}

    print(f"\ncheckpoint: {args.checkpoint or 'from config'}")
    print(f"batches: {batches} x {args.batch_size}   gold_injection: {args.gold_injection}")
    print(f"bucket reconciliation: worst relative drift {drift:.2e}")

    print("\n=== per-task contribution to each query-typed term (weighted, mean per batch)")
    header = (f"{'term':<11}{'weight':>7}{'reached':>9}"
              + "".join(f"{t:>16}" for t in TASK_TYPES) + f"{'term total':>13}")
    print(header)
    weighted = defaultdict(float)
    reachable_now = reachable_all = 0.0
    event_reach_now = event_reach_all = 0.0
    for term, source, loss_key, reached in TERMS:
        if f"{TASK_TYPES[0]}_{term}_loss" not in mean:
            continue
        w = term_weight(term, source, model)
        row = f"{term:<11}{w:>7.2f}{('yes' if reached else '-'):>9}"
        for task in TASK_TYPES:
            share = w * mean[f"{task}_{term}_loss"]
            weighted[task] += share
            row += f"{share:>16.5f}"
        whole = w * mean[loss_key]
        reachable_all += whole
        event_reach_all += w * mean[f"events_{term}_loss"]
        if reached:
            reachable_now += whole
            event_reach_now += w * mean[f"events_{term}_loss"]
        print(row + f"{whole:>13.5f}")

    print("\n=== positive vs negative mass (only where a positive term is defined)")
    print(f"{'term':<11}{'task':<16}{'positive':>12}{'negative':>12}{'pos frac':>10}")
    for term, _, _, _ in TERMS:
        for task in TASK_TYPES:
            total = mean.get(f"{task}_{term}_loss", 0.0)
            key = f"{task}_{term}_pos_loss"
            if total <= 0 or key not in mean:
                continue
            pos = mean[key]
            print(f"{term:<11}{task:<16}{pos:>12.5f}{total - pos:>12.5f}{pos / total:>10.3f}")

    settings = model.boundary_head.settings
    other = 0.0
    print("\n=== not query-typed (no per-task split is definable)")
    for key, weight_attr in NOT_QUERY_TYPED:
        w = float(getattr(settings, weight_attr, 0.0))
        other += w * mean.get(key, 0.0)
        print(f"  {key:<24}{'w=' + format(w, '.3f'):>12}{w * mean.get(key, 0.0):>12.5f}")
    for key in ("classification_loss", "relation_loss", "record_object_loss",
                "record_field_loss"):
        other += mean.get(key, 0.0)
        print(f"  {key:<24}{'':>12}{mean.get(key, 0.0):>12.5f}")

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
    print(f"{'task':<18}{'query-typed':>13}{'own head':>12}{'total':>12}{'of loss':>10}")
    for task in TASK_TYPES:
        got = weighted[task] + single.get(task, 0.0)
        print(f"{task:<18}{weighted[task]:>13.5f}{single.get(task, 0.0):>12.5f}"
              f"{got:>12.5f}{got / max(grand, 1e-9):>10.1%}")
    print(f"{'TOTAL':<18}{'':>13}{'':>12}{grand:>12.5f}")
    print(f"  residual (should be ~0): {grand - attributed - other + sum(single.values()):+.6f}")

    print("\n=== REACH: how much of the loss a per-task weight can touch")
    print(f"  today (start/end/pair)      {reachable_now:>10.5f}  "
          f"{reachable_now / max(grand, 1e-9):>6.1%} of loss")
    print(f"  if extended to all 9 terms  {reachable_all:>10.5f}  "
          f"{reachable_all / max(grand, 1e-9):>6.1%} of loss"
          f"   ({reachable_all / max(reachable_now, 1e-9):.1f}x)")
    print(f"  EVENT mass reachable today  {event_reach_now:>10.5f}  "
          f"{event_reach_now / max(grand, 1e-9):>6.2%} of loss")
    print(f"  EVENT mass if extended      {event_reach_all:>10.5f}  "
          f"{event_reach_all / max(grand, 1e-9):>6.2%} of loss"
          f"   ({event_reach_all / max(event_reach_now, 1e-9):.1f}x)")
    print("  -> a flat event weight w moves events to about "
          f"w x {event_reach_now / max(grand, 1e-9):.2%} today, "
          f"w x {event_reach_all / max(grand, 1e-9):.2%} extended")

    print("\n=== queries per batch")
    for task in TASK_TYPES:
        print(f"  {task:<18}{mean.get(f'{task}_query_count', 0.0):>10.2f}")

    # Candidate treatments, on ONE comparable axis: what fraction of the whole
    # training gradient ends up being events. A term multiplier is not comparable --
    # a x5 on 1.6% of the loss and a x5 on 6.6% are different experiments.
    event_now = weighted["events"]
    pos = sum(
        term_weight(term, source, model) * mean.get(f"events_{term}_pos_loss", 0.0)
        for term, source, _, _ in TERMS if term in POS_WEIGHT_TERMS
    )
    pw_mass = sum(
        term_weight(term, source, model) * mean.get(f"events_{term}_loss", 0.0)
        for term, source, _, _ in TERMS if term in POS_WEIGHT_TERMS
    )
    neg = pw_mass - pos

    def share_after(delta):
        return (event_now + delta) / max(grand + delta, 1e-9)

    print(f"\n=== candidate treatments -> events' share of the WHOLE gradient")
    print(f"    control: events {event_now / max(grand, 1e-9):.2%}")
    print(f"    pos_weight reaches {pw_mass:.5f} ({pw_mass / max(grand, 1e-9):.2%} of loss), "
          f"pos {pos:.5f} / neg {neg:.5f}, pos frac {pos / max(pw_mass, 1e-9):.3f}")
    print(f"    flat weight reaches {event_reach_now:.5f} "
          f"({event_reach_now / max(grand, 1e-9):.2%}); extended would reach "
          f"{event_reach_all:.5f} ({event_reach_all / max(grand, 1e-9):.2%})")
    print(f"\n{'treatment':<34}{'event mass':>12}{'events % of gradient':>22}")
    rows = [("control", 0.0)]
    rows += [(f"flat w={w} (start/end/pair) [TESTED]", (w - 1) * event_reach_now)
             for w in (2.0, 4.0)]
    rows += [(f"pos_weight k={k} (+inside)", (k - 1) * pos) for k in (4.0, 8.0, 16.0, 32.0)]
    rows += [(f"flat w={w}, EXTENDED to 9 terms", (w - 1) * event_reach_all)
             for w in (2.0, 4.0, 8.0)]
    for label, delta in rows:
        print(f"{label:<34}{event_now + delta:>12.5f}{share_after(delta):>21.1%}")
    print(f"\n  output_dir: {args.output_dir}")


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
