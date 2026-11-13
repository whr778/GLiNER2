"""Can the EVENT record head fit a FEW training records? An overfit trace on the real training pipeline.

The relation head did not fit its own training data (TODO #40, `overfit_relations.py`). Event arguments carry the
same inflation: each occurrence of an argument's text becomes a separate REQUIRED value of a list role (measured
on eb21's collator: wikievents x3.29, cc_news_events_sonnet55_v2 x2.95, casie x1.70, cmnee x1.43). This trains a
checkpoint on N event records of ONE corpus -- the trainer's dataset, negatives, collator and loss -- and every
--every steps scores those same records with the in-run evaluator (`compute_metrics`, gold menu, global decode).

Logged per eval step: event_type / trigger / argument strict micro F1 at thresholds 0.5, 0.3 and 0.1, the median/mean
gold role assignment probability on training forwards since the last eval, and the mean record_field_loss and total loss.

    uv run tools/train/overfit_events.py --corpus casie_off --records 64 --steps 400 --device cuda --out casie.json
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
import gliner2.models.boundary.engine as E  # noqa: E402
import gliner2.models.boundary.records as R  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.training.eval_metrics import compute_metrics  # noqa: E402
from gliner2.training.negatives import NegativeLabels  # noqa: E402
from gliner2.training.trainer import ExtractorCollator, ExtractorDataset  # noqa: E402

GOLD_PROBS: list[float] = []


def _strip_event_offsets(records: list[dict]) -> tuple[list[dict], int, int]:
    """Convert {text, start, end} trigger/argument dicts to surface strings for compute_metrics."""
    out = copy.deepcopy(records)
    n_trig = n_arg = 0
    for rec in out:
        for ev in (rec.get("output") or {}).get("events") or []:
            if not isinstance(ev, dict):
                continue
            trigs = []
            for t in ev.get("triggers") or []:
                if isinstance(t, dict):
                    n_trig += 1
                    trigs.append(t.get("text", ""))
                else:
                    trigs.append(t)
            ev["triggers"] = trigs
            for a in ev.get("arguments") or []:
                if isinstance(a, dict) and isinstance(a.get("entity"), dict):
                    n_arg += 1
                    a["entity"] = a["entity"].get("text", "")
    return out, n_trig, n_arg


def _install_role_loss_probe(legacy: bool) -> None:
    """Record gold candidate probabilities and optionally restore the pre-fix role loss (control arm)."""
    real_bce = R._list_field_bce

    def legacy_bce(logits_row: torch.Tensor, positive_cols, hard_k: int = 0, ignore_cols=()) -> torch.Tensor:
        cand_logits = logits_row[1:]
        if cand_logits.numel() == 0:
            return logits_row.new_zeros(())
        target = torch.zeros_like(cand_logits)
        n = cand_logits.shape[0]
        for col in positive_cols:
            idx = col - 1
            if 0 <= idx < n:
                target[idx] = 1.0
        pos = target > 0
        if bool(pos.any()):
            GOLD_PROBS.extend(torch.sigmoid(cand_logits[pos].detach().float()).tolist())
        if hard_k > 0:
            n_neg = int((target == 0).sum())
            if n_neg > hard_k:
                wrong = cand_logits.detach().masked_fill(pos, float("-inf"))
                keep = pos.clone()
                keep.index_fill_(0, torch.topk(wrong, hard_k).indices, True)
                return F.binary_cross_entropy_with_logits(cand_logits[keep], target[keep], reduction="mean")
        return F.binary_cross_entropy_with_logits(cand_logits, target, reduction="mean")

    def legacy_instance_field_loss(group, inst: int, record, span_indices, hard_k: int = 0, drop_anchor: bool = False):
        total = group.object_logits.new_zeros(())
        n_fields = 0
        for f_idx, fspec in enumerate(group.field_specs):
            cols, is_scalar = R._field_target_cols(fspec, record, span_indices[f_idx])
            row = group.assign_logits[f_idx][inst]
            total = total + (
                R._scalar_field_nll(row, cols)
                if is_scalar else R._list_field_bce(row, cols, hard_k)
            )
            n_fields += 1
        return total / max(n_fields, 1)

    def tracked_bce(logits_row: torch.Tensor, positive_cols, hard_k: int = 0, ignore_cols=()) -> torch.Tensor:
        cand_logits = logits_row[1:]
        n = cand_logits.shape[0]
        for col in positive_cols:
            idx = col - 1
            if 0 <= idx < n:
                GOLD_PROBS.append(float(torch.sigmoid(cand_logits[idx].detach().float())))
        return real_bce(logits_row, positive_cols, hard_k, ignore_cols)

    if legacy:
        R._list_field_bce = legacy_bce
        R._instance_field_loss = legacy_instance_field_loss
        E.record_field_decision_score = (
            lambda assign_p, cand_p: min(float(cand_p), float(assign_p))
            if assign_p is not None else float(cand_p)
        )
    else:
        R._list_field_bce = tracked_bce


def event_scores(model, records, threshold: float) -> dict:
    """Strict micro F1 for event_type / trigger / argument on the records' own gold menu (the in-run evaluator)."""
    model.eval()
    with torch.no_grad():
        m = compute_metrics(model, ExtractorDataset(records, shuffle=False, validate=False), batch_size=2,
                            threshold=threshold, report=False, chunk_size=4096, chunk_overlap=0, global_decode=True) or {}
    model.train()
    return {h: round(float(m.get(f"eval_event_{h}_strict_micro_f1", float("nan"))), 4)
            for h in ("type", "trigger", "argument")}


def summarize(paths) -> None:
    """One block per run file: the config and overrides, then trigger/argument F1 at every scored step."""
    for path in paths:
        d = json.load(open(path, encoding="utf-8"))
        a = d["args"]
        print(f"== {path}: {a.get('config')} legacy={a.get('legacy_role_loss')} overrides={a.get('head_override') or 'none'}")
        for r in d["log"]:
            p50 = r["train"].get("gold_role_p_median")
            print(f"   step {r['step']:>4}  @0.1 trig {r['@0.1']['trigger']:.3f} arg {r['@0.1']['argument']:.3f}"
                  f"  |  @0.3 trig {r['@0.3']['trigger']:.3f} arg {r['@0.3']['argument']:.3f}"
                  f"  |  @0.5 trig {r.get('@0.5', {}).get('trigger', 0):.3f} arg {r.get('@0.5', {}).get('argument', 0):.3f}"
                  f"  |  gold_P_med {p50}  field_loss {r['train'].get('record_field_loss', 0):.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb22e.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb22e-epoch")
    ap.add_argument("--summarize", nargs="+", metavar="JSON", help="print the per-step table of written runs and exit")
    ap.add_argument("--corpus")
    ap.add_argument("--records", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--encoder-lr", type=float)
    ap.add_argument("--task-lr", type=float)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--negative-pools", help="override the config's negative_pools (`auto` reads every corpus)")
    ap.add_argument("--labels-file", help="override the config's labels_file")
    ap.add_argument("--eval-corpus", help="score against THIS corpus's records for the same documents")
    ap.add_argument("--from-scratch", action="store_true",
                    help="build the model from the config's `model` section (fresh heads) instead of --checkpoint")
    ap.add_argument("--legacy-role-loss", action="store_true",
                    help="control arm: combined (1+K) mean in _list_field_bce, unresolved roles trained as empty, "
                         "anchor row kept, and format gate reading min(pair, assignment)")
    ap.add_argument("--remove-events-prob", type=float,
                    help="override processor.sampling_config.remove_events_prob")
    ap.add_argument("--strip-bare-space-subword", type=int, choices=[0, 1],
                    help="override model.strip_bare_space_subword (0=False, 1=True)")
    ap.add_argument("--head-override", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override boundary_head settings, e.g. boundary_marginal_loss=bce (one arm of an A/B)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.summarize:
        return summarize(args.summarize)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.strip_bare_space_subword is not None:
        cfg["model"]["strip_bare_space_subword"] = bool(args.strip_bare_space_subword)
    for kv in args.head_override:
        k, v = kv.split("=", 1)
        cfg["model"]["boundary_head"][k] = yaml.safe_load(v)
    tr, head = cfg["training"], dict(cfg["model"]["boundary_head"])
    if args.legacy_role_loss:
        head["record_drop_anchor_field_loss"] = False
    if args.labels_file:
        cfg["labels_file"] = args.labels_file
    enc_lr = args.encoder_lr if args.encoder_lr is not None else float(tr.get("encoder_lr", 2e-5))
    task_lr = args.task_lr if args.task_lr is not None else float(tr.get("task_lr", 5e-4))

    batch_size, grad_accum = args.batch_size, args.grad_accum
    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb < 32.0 and batch_size > 2:
            eff = batch_size * grad_accum
            batch_size = 2
            grad_accum = max(1, eff // batch_size)
            print(f"[overfit-events] GPU {vram_gb:.1f} GB (<32 GB): adjusted batch_size={batch_size} grad_accum={grad_accum} (effective {batch_size * grad_accum})", flush=True)

    fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, args.config)), T.labels_passthrough(cfg)
    path = f"data/{args.corpus}.train.jsonl"
    T._fetch_if_missing(path)
    records = [r for r in T.read_transformed([path], fns, passthrough) if r["output"].get("events")][:args.records]
    for r in records:
        r["_corpus"] = args.corpus
    if args.eval_corpus:
        epath = f"data/{args.eval_corpus}.train.jsonl"
        T._fetch_if_missing(epath)
        by_text = {r["input"]: r for r in T.read_transformed([epath], fns, passthrough)}
        eval_records = [by_text[r["input"]] for r in records]
        print(f"[overfit-events] scoring against {args.eval_corpus}: {len(eval_records)} records, same texts", flush=True)
    else:
        eval_records, n_trig_off, n_arg_off = _strip_event_offsets(records)
        if n_trig_off or n_arg_off:
            print(f"[overfit-events] stripped offsets for eval: {n_trig_off} triggers, {n_arg_off} arguments", flush=True)
    n_args = sum(len(e.get("arguments") or []) for r in records for e in r["output"]["events"])
    print(f"[overfit-events] {args.corpus}: {len(records)} records, {sum(len(r['output']['events']) for r in records)} gold events, "
          f"{n_args} arguments | steps {args.steps} batch {batch_size}x{grad_accum} lr enc {enc_lr} task {task_lr} | "
          f"device {args.device} | labels {cfg.get('labels_file')}", flush=True)

    torch.manual_seed(42)
    random.seed(42)
    if args.from_scratch:
        model = T._build_model(cfg["model"])
    else:
        model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu")
        if "strip_bare_space_subword" in cfg["model"]:
            strip_bare = bool(cfg["model"]["strip_bare_space_subword"])
            model.config.strip_bare_space_subword = strip_bare
            if hasattr(model, "processor"):
                model.processor.strip_bare_space_subword = strip_bare
        T._apply_boundary_head_overrides(model, head)
    if args.device == "cuda" and bool(tr.get("gradient_checkpointing", False)) and hasattr(model, "encoder"):
        model.encoder.gradient_checkpointing_enable()
        if hasattr(model.encoder, "config"):
            model.encoder.config.use_cache = False
    model = model.to(args.device)
    from gliner2.training.trainer import apply_schema_dropout
    sd_setting = tr.get("schema_dropout", True)
    if sd_setting is not True:
        apply_schema_dropout(model.processor.sampling_config, sd_setting)
    if args.remove_events_prob is not None:
        model.processor.sampling_config.remove_events_prob = float(args.remove_events_prob)
    _install_role_loss_probe(args.legacy_role_loss)
    bh_settings = getattr(getattr(model, "boundary_head", None), "settings", None)
    print(f"[overfit-events] PROOF role_loss_mode={'control(legacy)' if args.legacy_role_loss else 'treatment(split+withhold)'} "
          f"hard_k={getattr(bh_settings, 'record_role_hard_negatives', '?')} "
          f"drop_anchor={getattr(bh_settings, 'record_drop_anchor_field_loss', '?')} "
          f"coref_ownership={getattr(bh_settings, 'record_coreferent_ownership', '?')} "
          f"pair_split={getattr(bh_settings, 'pair_split_reduction', '?')} "
          f"keep_all_absent={getattr(bh_settings, 'hard_negative_keep_all_when_absent', '?')} "
          f"max_field_tokens={getattr(bh_settings, 'record_field_max_tokens', '?')} "
          f"min_field_pair_logit={getattr(bh_settings, 'record_field_min_pair_logit', '?')} "
          f"strip_bare_space={getattr(model.processor, 'strip_bare_space_subword', False)} "
          f"entity_coref_thr={getattr(bh_settings, 'entity_coref_threshold', None)} "
          f"arg_coref_match={getattr(bh_settings, 'record_argument_coref_match', '?')} "
          f"arg_coref_copy={getattr(bh_settings, 'record_argument_coref_copy', '?')} "
          f"arg_coref_boost={getattr(bh_settings, 'record_argument_coref_boost', '?')} "
          f"remove_events_prob={model.processor.sampling_config.remove_events_prob}", flush=True)

    negatives = None
    if tr.get("negative_pools") and tr.get("negative_labels_per_dim"):
        pools = args.negative_pools or tr["negative_pools"]
        if str(pools).strip().lower() == "auto":
            cached = Path(args.out).parent / "negative_pools.auto.json"
            pools = str(cached) if cached.is_file() else T._auto_negative_pools(cfg, args.config, str(Path(args.out).parent))
        negatives = NegativeLabels.load(pools, tr["negative_labels_per_dim"],
                                        seed=int(tr.get("negative_label_seed", 42)), partial=cfg["data"].get("partial_annotation"),
                                        gold_full_roles=bool(tr.get("negative_gold_full_roles", False)))
    ds = ExtractorDataset(data=records, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=negatives)
    coll = ExtractorCollator(model.processor, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)),
                             on_capacity_exceeded="truncate_with_warning", error_policy=tr.get("error_policy", "skip"),
                             event_records=bool(head.get("event_records", False)))
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc, "lr": enc_lr}, {"params": rest, "lr": task_lr}])
    use_bf16 = args.device == "cuda" and bool(tr.get("bf16", False))

    # Re-seed after _auto_negative_pools so cached and uncached runs start from the exact same RNG state.
    torch.manual_seed(42)
    random.seed(42)
    R.reset_anchor_gate()
    log, rng, t0, losses = [], random.Random(0), time.time(), {"record_field_loss": [], "pair_loss": [], "loss": []}
    model.train()
    model.processor.change_mode(is_training=True)
    for step in range(args.steps + 1):
        if step % args.every == 0 and not (args.from_scratch and step == 0):
            model.processor.change_mode(is_training=False)
            row = {"step": step, "seconds": round(time.time() - t0)}
            for thr in (0.5, 0.3, 0.1):
                row[f"@{thr}"] = event_scores(model, eval_records, thr)
            model.processor.change_mode(is_training=True)
            train_stats = {k: round(sum(v) / max(len(v), 1), 4) for k, v in losses.items()}
            if GOLD_PROBS:
                sp = sorted(GOLD_PROBS)
                train_stats["gold_role_n"] = len(sp)
                train_stats["gold_role_p_median"] = round(sp[len(sp) // 2], 6)
                train_stats["gold_role_p_mean"] = round(sum(sp) / len(sp), 6)
                train_stats["gold_role_p_ge_0.5"] = round(sum(p >= 0.5 for p in sp) / len(sp), 4)
            gate_st = R.anchor_gate_stats()
            train_stats["arg_exceeds_max_tokens_n"] = int(gate_st.get("arg_exceeds_max_tokens_n", 0))
            row["train"] = train_stats
            log.append(row)
            print(f"[overfit-events] {json.dumps(row)}", flush=True)
            losses = {k: [] for k in losses}
            GOLD_PROBS.clear()
        if step == args.steps:
            break
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_rfl = []
        step_pl = []
        for micro in range(grad_accum):
            idx = rng.sample(range(len(ds)), min(batch_size, len(ds)))
            if negatives is not None:
                negatives.set_epoch(step * grad_accum + micro)
            batch = coll([ds[i] for i in idx]).to(args.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(batch)
                loss = out.loss / grad_accum
            step_loss += float(loss.detach())
            if out.losses.get("record_field_loss") is not None:
                step_rfl.append(float(out.losses["record_field_loss"].detach()))
            if out.losses.get("pair_loss") is not None:
                step_pl.append(float(out.losses["pair_loss"].detach()))
            loss.backward()
        losses["loss"].append(step_loss)
        if step_rfl:
            losses["record_field_loss"].append(sum(step_rfl) / len(step_rfl))
        if step_pl:
            losses["pair_loss"].append(sum(step_pl) / len(step_pl))
        if step % 10 == 0:
            print(f"[overfit-events] step {step} loss {step_loss:.4f} | {time.time() - t0:.0f}s", flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("max_grad_norm", 1.0)))
        opt.step()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"args": vars(args), "log": log}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[overfit-events] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
