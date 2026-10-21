"""Do entity and event supervision push the SHARED weights in opposite directions? Per-group gradient cosines.

Entity, trigger and role queries are rows of one pair-score tensor, scored by one SparseBoundaryPairScorer on one
encoder. In eb22c's autopsy run (2026-10-21) the entity and trigger score LEVELS moved in opposite phase across the
warmup peak (3.5k -> 4k -> 4.5k), within the same CASIE documents. This measures the cause directly.

One forward per batch on the TRAINING path (real trainer data preparation, negatives, collator, train mode with
dropout off, the gold-injection / consistency / soft-IoU schedules at the checkpoint's step), then three gradients:
  entity   sum over terms of weight x the entities share of that span loss   (reduce_by_task contributions)
  event    the same for the events share (trigger AND role queries)
  bind     the record head's loss WITHOUT the trigger link: trigger <-> argument assignment, junction, negatives
  tlink    the trigger x trigger coreference link (TriggerLink), captured per group from records._coref_link_loss
  relation the relation scorer's loss;  elink  the weighted entity-mention coreference link
  rest     total minus all of these: classification, consistency, the relation/json shares of the span losses
The shares must sum to each term, or the tool refuses to report. Gradients are summed over --batches, then cosines
between pairs of components are reported per parameter group. Negative = they pull the shared weights apart.

    uv run tools/train/gradient_conflict.py --config tools/train/config/base/eb22c-ddp-trace.yaml \\
        --checkpoints whr778/gliner2-eb22c-collapse-trace:checkpoint-3500 whr778/gliner2-eb22c-collapse-trace:checkpoint-4000 \\
        --records 400 --batches 40 --out conflict.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import resource
import sys
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # probe_task_losses imports tools.train.train
import probe_event_collapse as P  # noqa: E402
import train as T  # noqa: E402
from probe_task_losses import TERMS, term_weight  # noqa: E402
from gliner2.models.boundary import records as R  # noqa: E402
from gliner2.models.boundary.model import TASK_TYPES  # noqa: E402
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig  # noqa: E402


def group_of(name: str) -> str:
    """Parameter group: one per encoder layer, one per head module."""
    parts = name.split(".")
    if parts[0] == "encoder":
        return f"encoder.layer{int(parts[2]):02d}" if parts[1] == "layers" else "encoder.other"
    return ".".join(parts[:2])


def training_records(cfg: dict, cfg_path: str, n: int, seed: int, only=None) -> list[dict]:
    """A seeded sample of the config's real train mix, label-mapped and corpus-tagged as train.py does
    (restricted to the corpora in `only` when given)."""
    d = cfg["data"]
    paths = T._dedupe_paths(T._split_files(d["corpora"], "train", set(d.get("train_only") or []))
                            + T._event_split(d.get("event_files") or {}, "train"), "train")
    fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, cfg_path)), T.labels_passthrough(cfg)
    records = [r for r in T.read_transformed(paths, fns, passthrough) if not only or r.get("_corpus") in only]
    return random.Random(seed).sample(records, min(n, len(records)))


def build(cfg: dict, cfg_path: str, checkpoint: str, records: list[dict], batch_size: int, device: str = "cpu"):
    """Model with per-task losses on, and the trainer's own training dataloader over `records`."""
    model_cfg = dict(cfg["model"], pretrained=P.local_checkpoint(checkpoint), map_location="cpu")
    model_cfg["boundary_head"] = dict(model_cfg.get("boundary_head") or {}, report_task_losses=True)
    model = T._build_model(model_cfg).to(device)
    train_cfg = dict(cfg["training"], output_dir=tempfile.mkdtemp(), max_train_samples=-1, num_workers=0, bf16=False,
                     fp16=False, batch_size=batch_size)
    if train_cfg.get("negative_pools") == "auto":
        train_cfg["negative_pools"] = T._auto_negative_pools(cfg, cfg_path, tempfile.mkdtemp())
    known = TrainingConfig.__dataclass_fields__
    trainer = GLiNER2Trainer(model, TrainingConfig(**{k: v for k, v in train_cfg.items() if k in known}))
    trainer.device = torch.device(device)
    dataset = trainer._prepare_data(records, is_train=True)
    return model, trainer, trainer._create_dataloader(dataset, batch_size, shuffle=False, is_training=True)


def set_schedules(trainer, model, step: int, planned: int) -> None:
    """The per-step settings the trainer applies in _backward_one, at this checkpoint's step."""
    head, c = model.boundary_head, trainer.config
    head.set_gold_injection_prob(trainer._gold_injection_probability(
        step / planned, c.gold_injection_start, c.gold_injection_end, c.gold_injection_hold_frac))
    warm = head.settings.consistency_warmup_steps
    head.set_consistency_scale(1.0 if warm <= 0 else min(step / warm, 1.0))
    head.set_soft_iou_scale(trainer._soft_iou_anneal_scale(step, head.settings.soft_iou_anneal_steps))


def task_loss(model, losses: dict, task: str) -> torch.Tensor:
    """Weighted sum of one task's share of every query-typed span term, as it enters the head's total."""
    total = 0.0
    for term, source, key, _ in TERMS:
        share = losses.get(f"{task}_{term}_loss")
        if share is None:
            continue
        scale = {"soft_iou": float(model.boundary_head._soft_iou_scale)}.get(term, 1.0)
        total = total + term_weight(term, source, model) * scale * share
    return total


def check_shares(losses: dict) -> None:
    """Every term's per-task shares must sum to the term itself."""
    for term, _, key, _ in TERMS:
        shares = [losses[f"{t}_{term}_loss"] for t in TASK_TYPES if f"{t}_{term}_loss" in losses]
        if shares and key in losses:
            s, whole = float(sum(shares)), float(losses[key])
            assert abs(s - whole) <= 1e-4 * max(1.0, abs(whole)), f"{term}: shares sum {s:.6f} != {key} {whole:.6f}"


def check_weights(model, losses: dict) -> None:
    """The weighted task losses plus the (not query-typed) consistency term must rebuild the head's span total.
    A batch with no span supervision (e.g. classification only) has no span total and nothing to rebuild."""
    if "total_loss" not in losses:
        return
    head = model.boundary_head
    rebuilt = sum(float(task_loss(model, losses, t)) for t in TASK_TYPES) + float(
        head.settings.consistency_loss_weight * head._consistency_scale * losses.get("consistency_loss", 0.0))
    whole = float(losses["total_loss"])
    assert abs(rebuilt - whole) <= 1e-3 * max(1.0, abs(whole)), f"weighted task losses {rebuilt:.6f} != span total {whole:.6f}"


PARTS = ("entity", "event", "bind", "tlink", "relation", "elink", "rest")
PAIRS = (("entity", "event"), ("entity", "bind"), ("event", "bind"), ("entity", "tlink"), ("event", "tlink"),
         ("entity", "elink"), ("entity", "relation"), ("entity", "rest"), ("event", "rest"))


def trigger_link_term(model, captured: dict):
    """The trigger x trigger coreference link's share of the record loss, as _record_loss weights it:
    record_loss_weight * record_coref_link_weight * (summed BCE over groups) / (pairs over groups)."""
    pairs = sum(n for _, n in captured.get("tlink", []))
    if not pairs:
        return None
    s = model.boundary_settings
    return s.record_loss_weight * s.record_coref_link_weight * sum(l for l, _ in captured["tlink"]) / pairs


def components(model, out, captured: dict) -> dict:
    """The additive pieces of out.loss that have their own gradient; `rest` is computed as total minus these.
    bind = the record loss WITHOUT the trigger link (trigger <-> argument assignment, junction column, negatives)."""
    losses = out.losses
    elink, tlink, record = losses.get("entity_coref_loss"), trigger_link_term(model, captured), captured.get("record")
    return {"entity": task_loss(model, losses, "entities"), "event": task_loss(model, losses, "events"),
            "bind": record if tlink is None or record is None else record - tlink, "tlink": tlink,
            "relation": losses.get("relation_loss"),
            "elink": None if elink is None else model.boundary_settings.entity_coref_link_weight * elink}


def _device(model) -> torch.device:
    return next(model.parameters()).device


def verify_link(model, batch, captured: dict, seed: int, bf16: bool) -> None:
    """Replay the batch with record_coref_link_weight 0 under the same seed: the record loss must drop by exactly
    the captured trigger-link term, or the link was mis-attributed."""
    term = trigger_link_term(model, captured)
    with_link, base, off = captured["record"], model.boundary_settings, {}
    zero = dataclasses.replace(base, record_coref_link_weight=0.0)
    orig = model._record_loss

    def hook(*a, **k):
        out = orig(*a, **k)
        off["record"] = out["total"]
        return out

    model.boundary_settings = model.boundary_head.settings = zero
    model._record_loss = hook
    try:
        torch.manual_seed(seed)
        random.seed(seed)
        with torch.no_grad(), torch.autocast(_device(model).type, dtype=torch.bfloat16, enabled=bf16):
            model(batch)
    finally:
        model.boundary_settings = model.boundary_head.settings = base
        model._record_loss = orig
    diff = float(with_link) - float(off["record"])
    assert abs(diff - float(term)) <= 1e-3 + 0.02 * abs(float(term)), \
        f"trigger link mis-attributed: record(w) - record(0) = {diff:.6f}, captured term {float(term):.6f}"
    print(f"[conflict] verify-link: record(w) - record(0) = {diff:+.6f}, captured term {float(term):+.6f} -- OK", flush=True)


def measure(model, trainer, loader, step: int, planned: int, batches: int, verify: int = 0, bf16: bool = False) -> dict:
    """Summed gradient of each component per parameter, and the per-batch global cos(entity, event)."""
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.embeddings")]
    ps = [p for _, p in params]
    sums = {k: [torch.zeros_like(p) for p in ps] for k in PARTS}
    per_batch, used, captured = [], 0, {}
    orig = model._record_loss

    def record_hook(*a, **k):
        out = orig(*a, **k)
        captured["record"] = out["total"]
        return out

    orig_link = R._coref_link_loss

    def link_hook(*a, **k):
        loss, n_pairs, wins, pairs = orig_link(*a, **k)
        if n_pairs:
            captured.setdefault("tlink", []).append((loss, n_pairs))
        return loss, n_pairs, wins, pairs

    model._record_loss = record_hook
    R._coref_link_loss = link_hook
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    set_schedules(trainer, model, step, planned)
    for batch in loader:
        if used >= batches:
            break
        captured.clear()
        batch = batch.to(_device(model))
        torch.manual_seed(used)
        random.seed(used)
        with torch.autocast(_device(model).type, dtype=torch.bfloat16, enabled=bf16):
            out = model(batch)
        check_shares(out.losses)
        check_weights(model, out.losses)
        comp = components(model, out, captured)
        if verify > 0 and comp["tlink"] is not None:
            verify_link(model, batch, dict(captured), used, bf16)
            verify -= 1
        zero = lambda: [None] * len(ps)
        g = {k: (torch.autograd.grad(v, ps, retain_graph=True, allow_unused=True)
                 if torch.is_tensor(v) and v.requires_grad else zero()) for k, v in comp.items()}
        total = torch.autograd.grad(out.loss, ps, allow_unused=True)
        g["rest"] = [t - sum(g[k][i] for k in comp if g[k][i] is not None) if t is not None else None
                     for i, t in enumerate(total)]
        for k in PARTS:
            for acc, v in zip(sums[k], g[k]):
                if v is not None:
                    acc += v
        flat = {k: torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for v, p in zip(g[k], ps)])
                for k in ("entity", "event")}
        if flat["entity"].norm() > 0 and flat["event"].norm() > 0:
            per_batch.append(float(torch.nn.functional.cosine_similarity(flat["entity"], flat["event"], dim=0)))
        used += 1
        peak = (f"cuda peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if _device(model).type == "cuda"
                else f"peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB (excludes compressed)")
        print(f"[conflict] batch {used}: input {tuple(batch.input_ids.shape)} | {peak}", flush=True)
        del out, g, flat, total
    model._record_loss = orig
    R._coref_link_loss = orig_link
    return {"params": params, "sums": sums, "per_batch_cosine": per_batch, "batches": used}


def cos(a: torch.Tensor, b: torch.Tensor):
    na, nb = a.norm(), b.norm()
    return float(a.flatten() @ b.flatten() / (na * nb)) if na > 0 and nb > 0 else None


def report(name: str, m: dict) -> dict:
    groups = {}
    for i, (n, _) in enumerate(m["params"]):
        groups.setdefault(group_of(n), []).append(i)
    rows = {}
    for g, idx in sorted(groups.items()):
        v = {k: torch.cat([m["sums"][k][i].flatten() for i in idx]) for k in PARTS}
        rows[g] = {"cos": {f"{a}~{b}": cos(v[a], v[b]) for a, b in PAIRS}, "norm": {k: float(v[k].norm()) for k in PARTS}}
    pb = sorted(m["per_batch_cosine"])
    print(f"[conflict] {name}: {m['batches']} batches, {len(pb)} with both tasks | per-batch cos(entity, event) "
          f"median {pb[len(pb) // 2] if pb else float('nan'):+.3f}, negative in {sum(c < 0 for c in pb)}/{len(pb)}", flush=True)
    fmt = lambda x: "  n/a " if x is None else f"{x:+.3f}"
    print(f"[conflict]   {'group':34s} " + " ".join(f"{a[:3]}~{b[:4]:5s}" for a, b in PAIRS) + "   |g| " + " ".join(PARTS), flush=True)
    for g, r in rows.items():
        if not any(r["norm"][k] > 0 for k in ("entity", "event")):
            continue
        print(f"[conflict]   {g:34s} " + " ".join(f"{fmt(r['cos'][f'{a}~{b}']):>10s}" for a, b in PAIRS) + "   "
              + " ".join(f"{r['norm'][k]:.3g}" for k in PARTS), flush=True)
    return {"groups": rows, "per_batch_cosine": m["per_batch_cosine"], "batches": m["batches"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="local dirs or repo:checkpoint-<step>")
    ap.add_argument("--records", type=int, default=400, help="seeded sample of the real train mix")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--planned-steps", type=int, default=82930, help="the run's planned total (schedules)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only-corpora", nargs="*", help="restrict the sample to these corpora (traces only)")
    ap.add_argument("--verify-link", type=int, default=0, help="replay the first N link-carrying batches at weight 0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    records = training_records(cfg, args.config, args.records, args.seed, set(args.only_corpora or []))
    result = {"args": vars(args), "checkpoints": {}}
    for ref in args.checkpoints:
        step = int(re.search(r"checkpoint-(\d+)", ref).group(1))
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        model, trainer, loader = build(cfg, args.config, ref, [dict(r) for r in records], args.batch_size, args.device)
        bf16 = args.device.startswith("cuda") and bool(cfg["training"].get("bf16"))
        print(f"[conflict] {ref}: step {step}, device {args.device}, bf16 autocast {bf16}, batch {args.batch_size}", flush=True)
        result["checkpoints"][ref] = report(ref, measure(model, trainer, loader, step, args.planned_steps, args.batches,
                                                         args.verify_link, bf16))
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        del model, trainer, loader
    print(f"[conflict] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
