"""Which loss terms drive the update? Gradient share per term on real training micro-batches.

share_k = <g_k, g> / |g|^2, where g_k is the gradient of term k's WEIGHTED contribution and g
the gradient of the total; shares sum to 1 by linearity, for ANY weights. Computed PER
PARAMETER GROUP, because the optimizer is AdamW: it divides each coordinate by its own RMS,
so a module touched by one term alone (the record decoder) steps at full size whatever its
global share. Within a group the terms compete for the same coordinates, which is where a
share means "how much of this module's direction the term sets". Batches are drawn like
training: `batch_size` chunks from the natural mix (corpus weight = train records), real
transform, chunking, sampling, negatives and collator, eb18 in train mode. CPU, no step.

    uv run python tools/train/measure_gradient_shares.py --batches 8 [--absent-reduction separate]
    # late-training state, where eb18 spent most of its steps:
    uv run python tools/train/measure_gradient_shares.py --gold-injection 0.25 --soft-iou-scale 0 --proposal-gold identity
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import measure_absent_dilution as D  # noqa: E402
import train as T  # noqa: E402

SPAN = ("start_loss", "end_loss", "pair_loss", "inside_loss", "soft_iou_loss", "rerank_listwise_loss",
        "proposal_loss", "consistency_loss", "abstention_loss", "count_loss")
OUTER = ("classification_loss", "record_object_loss", "record_field_loss", "relation_loss")


def span_weights(model) -> dict:
    s, h, lw = model.boundary_head.settings, model.boundary_head, model.boundary_head.loss_weights
    return {"start_loss": lw.get("start", 1.0), "end_loss": lw.get("end", 1.0), "pair_loss": lw.get("pair", 1.0),
            "inside_loss": lw.get("inside", 0.5), "soft_iou_loss": s.soft_iou_aux_weight * h._soft_iou_scale,
            "rerank_listwise_loss": s.rerank_listwise_weight, "proposal_loss": s.proposal_loss_weight,
            "consistency_loss": s.consistency_loss_weight * h._consistency_scale,
            "abstention_loss": s.abstention_loss_weight, "count_loss": s.count_loss_weight}


GROUPS = (("encoder", "encoder."), ("boundary_head", "boundary_head."), ("record_decoder", "record_decoder."),
          ("relations", "relation_"), ("classifier", "classifier."))


def group_of(name: str) -> str:
    return next((g for g, prefix in GROUPS if name.startswith(prefix)), "other")


def shares(model, batch, seed):
    """({group: {term: share}}, {group: grad norm}, residual of the weighted sum vs the model's loss)."""
    model.train()
    torch.manual_seed(seed)
    out = model(batch)
    L = {k: v for k, v in out.losses.items() if torch.is_tensor(v) and v.dim() == 0}
    w = span_weights(model)
    terms = {**{k: w[k] * L[k] for k in SPAN}, **{k: L[k] for k in OUTER if k in L}}
    residual = float(out.loss) - float(sum(terms.values()))
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    groups = [group_of(n) for n, _ in named]
    g = torch.autograd.grad(out.loss, params, retain_graph=True, allow_unused=True)
    sq = lambda xs, ys: {grp: float(sum((a.float() * b.float()).sum() for a, b, gg in zip(xs, ys, groups)
                                        if gg == grp and a is not None and b is not None)) for grp in set(groups)}
    g2 = sq(g, g)
    out_shares = {grp: {} for grp in g2}
    for k, t in terms.items():
        dots = sq(torch.autograd.grad(t, params, retain_graph=True, allow_unused=True), g) if t.requires_grad else {}
        for grp in g2:
            out_shares[grp][k] = dots.get(grp, 0.0) / g2[grp] if g2[grp] > 0 else 0.0
    return out_shares, {grp: v ** 0.5 for grp, v in g2.items()}, residual


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb18-balanced")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--absent-reduction", default="pooled")
    ap.add_argument("--proposal-gold", default="injected")
    ap.add_argument("--gold-injection", type=float, default=1.0, help="trainer schedule: 1.0 early, 0.25 at the end")
    ap.add_argument("--soft-iou-scale", type=float, default=1.0, help="trainer schedule: 1.0 at step 0, 0 after 20k steps")
    ap.add_argument("--consistency-scale", type=float, default=1.0, help="trainer schedule: 0 at step 0, 1 after 2k steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tr, head = cfg["training"], cfg["model"]["boundary_head"]
    from gliner2 import AutoExtractor
    from gliner2.training.chunking import chunk_records
    from gliner2.training.negatives import NegativeLabels
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu")
    T._apply_boundary_head_overrides(model, {"absent_reduction": args.absent_reduction, "absent_loss_weight": 1.0,
                                             "present_loss_scale": 1.0, "proposal_gold": args.proposal_gold})
    head_ = model.boundary_head
    head_.set_gold_injection_prob(args.gold_injection)
    head_.set_soft_iou_scale(args.soft_iou_scale)
    head_.set_consistency_scale(args.consistency_scale)
    proc = model.processor
    spec = importlib.util.spec_from_file_location("bnp", ROOT / "tools/data/build_negative_pools.py")
    bnp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bnp)
    neg = NegativeLabels(bnp.build_pools(cfg, Path(args.config), log=lambda *a: None), tr["negative_labels_per_dim"],
                         seed=int(tr.get("negative_label_seed", 42)), partial=cfg["data"].get("partial_annotation"))
    sampled, totals = D.load_records(cfg, args.config, 40, args.seed)
    names = sorted(sampled)
    coll = ExtractorCollator(proc, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)),
                             on_capacity_exceeded=tr.get("on_capacity_exceeded", "raise"), error_policy="skip",
                             event_records=bool(head.get("event_records", False)))
    rng = random.Random(args.seed)
    per_term = defaultdict(list)
    print(f"[shares] absent_reduction={head_.settings.absent_reduction} proposal_gold={head_.settings.proposal_gold} "
          f"p_inj={head_._gold_injection_prob} soft_iou_scale={head_._soft_iou_scale} "
          f"consistency_scale={head_._consistency_scale} batch_size={tr['batch_size']}")
    for b in range(args.batches):
        corpora = rng.choices(names, [totals[n] for n in names], k=int(tr["batch_size"]))
        recs = [rng.choice(sampled[c]) for c in corpora]
        chunks = chunk_records(recs, tokenizer=proc.tokenizer, window_size=int(tr["max_len"]),
                               stride=int(tr["window_stride"]), show_progress=False)
        ds = ExtractorDataset(data=chunks, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=neg)
        random.seed(args.seed + b)
        batch = coll([ds[i] for i in rng.sample(range(len(ds)), min(len(ds), int(tr["batch_size"])))])
        sh, gn, residual = shares(model, batch, args.seed + b)
        for grp, terms in sh.items():
            for k, v in terms.items():
                per_term[(grp, k)].append(v)
        print(f"[shares] batch {b} corpora {corpora} | residual {residual:.1e} | "
              + " | ".join(f"{grp} |g| {gn[grp]:.2f} sum {sum(sh[grp].values()):.4f}" for grp in sorted(sh)), flush=True)
    for grp in sorted({g for g, _ in per_term}):
        print(f"[shares] --- {grp}: {'term':22s} {'mean':>7s} {'min':>7s} {'max':>7s}  over {args.batches} batches")
        keys = [k for g, k in per_term if g == grp]
        for k in sorted(keys, key=lambda k: -statistics.mean(per_term[(grp, k)])):
            v = per_term[(grp, k)]
            print(f"[shares] {grp:14s} {k:22s} {statistics.mean(v):7.4f} {min(v):7.4f} {max(v):7.4f}")


if __name__ == "__main__":
    main()
