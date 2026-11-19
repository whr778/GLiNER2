"""One-shot training-time estimate.

Prints a wall-clock projection ahead of ``trainer.train`` by:

1. Counting records (and estimating sliding-window chunk expansion on a
   strided sample so multi-corpus / long-document runs are representative).
2. Timing ``n_warmup`` real training micro-batches using the trainer's live
   model (LoRA + gradient checkpointing), negative-label injector, and
   ``ExtractorCollator`` (so ``RecordHead``, ``SparseRelationScorer``, and
   ``MentionLink`` losses execute during timing).
3. Timing validation loss and decode passes when ``eval_data`` is active,
   including per-epoch ``metric_sweep`` and optional post-train passes.
4. Restoring RNG state, gradients, and module-level once-only diagnostic
   counters so the estimate has zero side effect on training gates or seeds.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

from gliner2.training.data import InputExample


@contextmanager
def _preserve_run_state():
    """Save and restore RNGs and module-level in-run gate counters."""
    py_rng = random.getstate()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    saved_attrs: List[tuple] = []
    saved_dicts: List[tuple] = []
    saved_levels: List[tuple] = []

    from gliner2.models.boundary import (
        engine as b_eng,
        mention_link as b_ml,
        model as b_mod,
        records as b_rec,
        relations as b_rel,
    )

    scalar_targets = (
        (b_mod, ("_NEGATIVE_QUERY_SEEN", "_NEGATIVE_QUERY_SELECTED", "_NEGATIVE_QUERY_CALLS",
                 "_ABSENT_CELLS", "_KEPT_CELLS", "_ABSENT_CELL_CALLS",
                 "_PROPOSAL_INJECTED", "_PROPOSAL_IDENTITY", "_PROPOSAL_GOLD_CALLS")),
        (b_rec, ("_ANCHOR_FIELD_DROPS", "_ANCHOR_GATE_CALLS", "_ARG_MAX_TOKENS_DROPS",
                 "_NEGATIVE_INSTANCES", "_NEGATIVE_CALLS",
                 "_COLUMN_TERMS", "_COLUMN_WINS", "_COLUMN_PAIRS",
                 "_COLUMN_SIB_WINS", "_COLUMN_SIB_PAIRS", "_COLUMN_CALLS")),
        (b_rel, ("_REL_GOLD_TOTAL", "_REL_GOLD_KEPT", "_REL_GOLD_APPENDED", "_REL_GOLD_CALLS")),
        (b_eng, ("_TYPED_ROLE_CALLS",)),
    )
    for mod, names in scalar_targets:
        for name in names:
            if hasattr(mod, name):
                saved_attrs.append((mod, name, getattr(mod, name)))

    for mod, name in ((b_rec, "_ANCHOR_GATE"), (b_rec, "_ANCHOR_GATE_BY_TASK"),
                      (b_rec, "_CPAIR"), (b_rec, "_COREF"), (b_ml, "_NOTE")):
        d = getattr(mod, name, None)
        if isinstance(d, dict):
            saved_dicts.append((d, dict(d)))

    for lname in ("gliner2.models.boundary", "gliner2.processing"):
        lg = logging.getLogger(lname)
        saved_levels.append((lg, lg.level))
        lg.setLevel(logging.ERROR)

    try:
        yield
    finally:
        for lg, lvl in saved_levels:
            lg.setLevel(lvl)
        for mod, name, val in saved_attrs:
            setattr(mod, name, val)
        for d, snap in saved_dicts:
            d.clear()
            d.update(snap)
        random.setstate(py_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _build_collator(model, config, processor, *, is_training: bool):
    from gliner2.training.trainer import ExtractorCollator

    base = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    base = base.module if hasattr(base, "module") else base
    model_cfg = getattr(base, "config", None)
    architecture = getattr(base, "architecture", getattr(model_cfg, "architecture", "span"))
    bhead = getattr(model_cfg, "boundary_head", None) or {}
    max_len = getattr(config, "max_len", None) or getattr(model_cfg, "max_len", None)
    max_gold = bhead.get("max_gold_per_query", 32)
    return ExtractorCollator(
        processor,
        is_training=is_training,
        max_len=max_len,
        architecture=architecture,
        max_gold_per_query=max_gold,
        build_targets=None if is_training else True,
        on_capacity_exceeded=getattr(config, "on_capacity_exceeded", "raise"),
        error_policy=getattr(config, "error_policy", "raise"),
        event_records=bool(bhead.get("event_records", False)),
        allow_invalid_samples=bool(getattr(config, "allow_invalid_samples", False)),
        ignore_missing_entities=bool(getattr(config, "ignore_missing_entities", False)),
        suppress_orphaned_queries=(
            is_training and bool(getattr(config, "suppress_orphaned_queries", False))
        ),
        dedupe_contained_entity_spans=bool(getattr(config, "dedupe_contained_entity_spans", True)),
        relation_gold_pairs=getattr(config, "relation_gold_pairs", "all"),
    )


def _resolve_negatives(trainer, config):
    from gliner2.training.negatives import NegativeLabels

    if trainer is not None and hasattr(trainer, "_negative_labels"):
        neg = trainer._negative_labels()
        if neg is None:
            return None
        return NegativeLabels(
            neg.pools,
            neg.per_dim,
            seed=neg.seed,
            max_per_record=neg.max_per_record,
            partial=neg.partial,
            gold_free_dims=neg.gold_free_dims,
            k_sampling=neg.k_sampling,
            schedule=neg.schedule,
            gold_full_roles=neg.gold_full_roles,
        )
    pools = getattr(config, "negative_pools", None)
    per_dim = getattr(config, "negative_labels_per_dim", None)
    if not (pools and per_dim):
        return None
    return NegativeLabels.load(
        pools,
        per_dim,
        seed=getattr(config, "negative_label_seed", 42),
        max_per_record=getattr(config, "negative_max_per_record", None),
        partial=getattr(config, "partial_annotation", None),
        gold_free_dims=bool(getattr(config, "negative_gold_free_dims", False)),
        k_sampling=getattr(config, "negative_k_sampling", "fixed"),
        schedule=getattr(config, "negative_labels_schedule", None),
        gold_full_roles=bool(getattr(config, "negative_gold_full_roles", False)),
    )


def _maybe_chunk(records: List[dict], processor, config) -> tuple[List[dict], float]:
    if not records or not getattr(config, "sliding_window", False):
        return records, 1.0
    max_len = getattr(config, "max_len", None)
    if not max_len or max_len <= 0:
        return records, 1.0
    from gliner2.training.chunking import chunk_records

    stride = getattr(config, "window_stride", None) or max(1, (max_len * 3) // 4)
    use_coref = getattr(config, "coref_clusters", "enable") == "enable"
    chunked = chunk_records(
        records,
        processor.tokenizer,
        window_size=max_len,
        stride=stride,
        show_progress=False,
        use_coref_clusters=use_coref,
    )
    if not chunked:
        return records, 1.0
    return chunked, len(chunked) / len(records)


def _time_eval_seconds(
    base_model,
    live_model,
    processor,
    config,
    eval_data: Any,
    *,
    total_opt_steps: int,
    eval_batch_size: Optional[int],
    eval_decode_passes: int,
    post_eval_passes: float,
    global_decode: bool,
    chunk_size: Optional[int],
    chunk_overlap: int,
    use_amp: bool,
    amp_dtype,
    device_type: str,
) -> tuple[float, str]:
    n_eval_raw = _count_records(eval_data)
    if n_eval_raw == 0:
        return 0.0, ""

    eval_strategy = getattr(config, "eval_strategy", "no")
    if eval_strategy == "epoch":
        n_evals = int(config.num_epochs)
    elif eval_strategy == "steps" and getattr(config, "eval_steps", 0) > 0:
        n_evals = total_opt_steps // int(config.eval_steps)
    else:
        n_evals = 0

    if n_evals == 0 and post_eval_passes <= 0:
        return 0.0, ""

    ebs = max(1, int(eval_batch_size or getattr(config, "eval_batch_size", 0) or config.batch_size))
    sample = _warmup_records(eval_data, n=min(n_eval_raw, max(4, 2 * ebs)), total_hint=n_eval_raw)
    if not sample:
        return 0.0, ""

    from gliner2.training.eval_metrics import _schema_from_gold
    from gliner2.training.trainer import ExtractorDataset

    chunked_eval, eval_ratio = _maybe_chunk(sample, processor, config)
    n_eval_chunks = max(1, round(n_eval_raw * eval_ratio))

    # 1) Time eval loss forward (1 pass per in-loop eval).
    loss_per_rec = 0.0
    if n_evals > 0:
        eval_ds = ExtractorDataset(
            chunked_eval, max_samples=-1, shuffle=False, validate=False, expand_argument_coref=False
        )
        eval_coll = _build_collator(base_model, config, processor, is_training=False)
        eval_loader = DataLoader(eval_ds, batch_size=ebs, collate_fn=eval_coll, shuffle=False)
        live_model.eval()
        processor.change_mode(is_training=False)
        t0 = None
        seen = 0
        with torch.no_grad():
            for idx, batch in enumerate(eval_loader):
                if idx == 0 and len(eval_loader) > 1:
                    with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
                        live_model(batch)
                    t0 = time.perf_counter()
                    continue
                if t0 is None:
                    t0 = time.perf_counter()
                with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
                    live_model(batch)
                seen += len(batch)
                if idx >= 2:
                    break
        if t0 is not None and seen > 0:
            loss_per_rec = (time.perf_counter() - t0) / seen

    # 2) Time decode pass per record using the real inference path.
    total_decode_passes = n_evals * max(0, int(eval_decode_passes)) + max(0.0, float(post_eval_passes))
    decode_per_rec = 0.0
    if total_decode_passes > 0 and hasattr(base_model, "batch_extract"):
        texts, schemas = [], []
        for rec in sample[: max(2, ebs)]:
            txt = rec.get("input") or rec.get("text") or ""
            out = rec.get("output") or rec.get("schema") or {}
            sc = _schema_from_gold(out)
            if txt and any(sc.values()):
                texts.append(txt)
                schemas.append(sc)
        if texts:
            t_dec = time.perf_counter()
            if global_decode or chunk_size is not None:
                cs = chunk_size or getattr(config, "max_len", None) or 384
                base_model.batch_extract_long(
                    texts, schemas, batch_size=ebs, threshold=0.5,
                    chunk_size=cs, chunk_overlap=chunk_overlap, global_decode=global_decode,
                )
            else:
                base_model.batch_extract(texts, schemas, batch_size=ebs, threshold=0.5)
            decode_per_rec = (time.perf_counter() - t_dec) / len(texts)

    eval_secs = (
        n_evals * n_eval_chunks * loss_per_rec
        + total_decode_passes * n_eval_raw * decode_per_rec
    )
    detail_parts = []
    if n_evals > 0:
        detail_parts.append(f"{n_evals * max(1, int(eval_decode_passes))} in-loop")
    if post_eval_passes > 0:
        detail_parts.append(f"~{post_eval_passes:g} post-train")
    note = f"({', '.join(detail_parts)} decode passes)" if detail_parts else ""
    return eval_secs, note


def estimate_eta(
    model,
    train_data: Any,
    config,
    n_warmup: int = 6,
    *,
    trainer=None,
    eval_data: Any = None,
    eval_batch_size: Optional[int] = None,
    eval_decode_passes: int = 0,
    post_eval_passes: float = 0.0,
    global_decode: bool = False,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 128,
) -> Optional[Dict[str, float]]:
    """Print a one-shot ETA estimate for the upcoming ``trainer.train`` call."""
    n_raw = _count_records(train_data)
    if n_raw == 0:
        print("[ETA] No training records found; skipping estimate.")
        return None

    live_model = trainer.model if (trainer is not None and getattr(trainer, "model", None) is not None) else model
    base_model = live_model.module if hasattr(live_model, "module") else live_model
    unwrap = base_model.get_base_model() if callable(getattr(base_model, "get_base_model", None)) else base_model
    processor = getattr(trainer, "processor", None) or getattr(unwrap, "processor", None)
    if processor is None:
        print("[ETA] Model has no processor; skipping estimate.")
        return None

    batch_size = max(1, int(config.batch_size))
    grad_accum = max(1, int(config.gradient_accumulation_steps))
    effective_batch = batch_size * grad_accum

    want_records = max((n_warmup + 2) * batch_size, 16)
    sample_records = _warmup_records(train_data, n=want_records, total_hint=n_raw)
    if not sample_records:
        print("[ETA] Could not build warmup sample; skipping estimate.")
        return None

    with _preserve_run_state():
        chunked_sample, chunk_ratio = _maybe_chunk(sample_records, processor, config)
        n_total = max(1, round(n_raw * chunk_ratio))

        batches_per_epoch = (
            n_total // batch_size if n_total > batch_size else math.ceil(n_total / batch_size)
        )
        steps_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum))
        if getattr(config, "max_steps", -1) and int(config.max_steps) > 0:
            total_opt_steps = int(config.max_steps)
            total_micro_batches = total_opt_steps * grad_accum
        else:
            total_opt_steps = steps_per_epoch * int(config.num_epochs)
            total_micro_batches = max(1, batches_per_epoch) * int(config.num_epochs)

        from gliner2.training.trainer import ExtractorDataset

        negatives = _resolve_negatives(trainer, config)
        use_coref = getattr(config, "coref_clusters", "enable") == "enable"
        sample_ds = ExtractorDataset(
            chunked_sample,
            max_samples=-1,
            shuffle=False,
            validate=False,
            negatives=negatives,
            expand_argument_coref=use_coref and bool(getattr(config, "expand_argument_coref", True)),
            use_coref_clusters=use_coref,
        )
        if len(sample_ds) == 0:
            print("[ETA] Empty warmup dataset; skipping estimate.")
            return None

        live_model.train()
        processor.change_mode(is_training=True)
        collator = _build_collator(unwrap, config, processor, is_training=True)

        loader = DataLoader(
            sample_ds,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        device_type = next(live_model.parameters()).device.type
        amp_dtype = torch.bfloat16 if getattr(config, "bf16", False) else torch.float16
        use_amp = bool(getattr(config, "bf16", False) or getattr(config, "fp16", False))
        gold_inj = float(getattr(config, "gold_injection_start", 1.0))

        timed = 0
        t_start = None
        for i, batch in enumerate(loader):
            if len(batch) == 0:
                continue
            if device_type == "cuda":
                torch.cuda.synchronize()
            elif device_type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
            if i == 1:
                t_start = time.perf_counter()
            try:
                with torch.amp.autocast(
                    device_type=device_type, enabled=use_amp, dtype=amp_dtype
                ):
                    try:
                        outputs = live_model(batch, gold_injection_prob=gold_inj)
                    except TypeError:
                        outputs = live_model(batch)
                loss = outputs.get("total_loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
                if loss is None and hasattr(outputs, "get"):
                    loss = outputs.get("loss")
                if loss is not None and loss.requires_grad:
                    loss.backward()
                if device_type == "cuda":
                    torch.cuda.synchronize()
                elif device_type == "mps" and hasattr(torch, "mps"):
                    torch.mps.synchronize()
            except Exception as e:
                print(f"[ETA] Warmup step {i} raised {type(e).__name__}: {e}")
                break
            finally:
                live_model.zero_grad(set_to_none=True)

            if i == 0 and len(loader) == 1:
                # Single-batch dataset: use that one batch rather than skipping.
                pass
            elif i >= 1:
                timed += 1
            if timed >= n_warmup:
                break

        if timed == 0:
            # Dataset had only 1 batch total; time one pass explicitly.
            for batch in loader:
                if len(batch) == 0:
                    break
                t0 = time.perf_counter()
                try:
                    with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
                        outputs = live_model(batch)
                    loss = outputs.get("total_loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
                    if loss is not None and loss.requires_grad:
                        loss.backward()
                    if device_type == "cuda":
                        torch.cuda.synchronize()
                    elif device_type == "mps" and hasattr(torch, "mps"):
                        torch.mps.synchronize()
                    t_start = t0
                    timed = 1
                except Exception:
                    pass
                finally:
                    live_model.zero_grad(set_to_none=True)
                break

        if t_start is None or timed == 0:
            print("[ETA] Too few warmup batches to time; skipping estimate.")
            return None

        per_micro = (time.perf_counter() - t_start) / timed
        per_opt_step = per_micro * grad_accum
        train_seconds = per_micro * total_micro_batches

        src_eval = eval_data if eval_data is not None else getattr(trainer, "eval_data", None)
        eval_seconds, eval_note = _time_eval_seconds(
            unwrap,
            live_model,
            processor,
            config,
            src_eval,
            total_opt_steps=total_opt_steps,
            eval_batch_size=eval_batch_size,
            eval_decode_passes=eval_decode_passes,
            post_eval_passes=post_eval_passes,
            global_decode=global_decode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            device_type=device_type,
        )
        live_model.train()
        processor.change_mode(is_training=True)

    eta_seconds = train_seconds + eval_seconds
    ex_str = (
        f"{n_raw:,} (est. {n_total:,} chunks)"
        if n_total != n_raw
        else f"{n_total:,}"
    )

    print("=" * 60)
    print("Training Time Estimate")
    print("=" * 60)
    print(f"Examples          : {ex_str}")
    print(f"Epochs            : {int(config.num_epochs)}")
    print(f"Effective batch   : {effective_batch}")
    print(f"Optimisation steps: {total_opt_steps:,} ({steps_per_epoch:,}/epoch)")
    print(f"Per micro-batch   : {per_micro * 1000:.0f}ms")
    print(f"Per optim. step   : {per_opt_step * 1000:.0f}ms")
    if eval_seconds > 0:
        print(f"Train steps ETA   : {_fmt_seconds(train_seconds)}")
        print(f"Eval / sweep ETA  : {_fmt_seconds(eval_seconds)} {eval_note}".rstrip())
    print(f"Projected ETA     : {_fmt_seconds(eta_seconds)}")
    print("=" * 60)
    return {
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "eta_seconds": eta_seconds,
        "total_opt_steps": float(total_opt_steps),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_jsonl_path(item: Any) -> bool:
    return isinstance(item, (str, Path)) and str(item).endswith(".jsonl")


def _iter_data_items(train_data: Any) -> Iterable[Any]:
    if isinstance(train_data, (str, Path)) or isinstance(train_data, InputExample):
        yield train_data
        return
    if isinstance(train_data, list):
        for item in train_data:
            yield item
        return
    try:
        for item in train_data:
            yield item
    except TypeError:
        return


def _count_records(train_data: Any) -> int:
    """Cheaply count records across JSONL files / in-memory lists."""
    if train_data is None:
        return 0
    total = 0
    for item in _iter_data_items(train_data):
        if _is_jsonl_path(item):
            p = Path(item)
            if not p.is_file():
                continue
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        total += 1
        else:
            total += 1
    return total


def _warmup_records(train_data: Any, n: int, total_hint: int = 0) -> List[dict]:
    """Pull ``n`` records strided across all training sources so multi-corpus
    and multi-genre mixes get a representative sample rather than only the
    first file's opening lines.
    """
    if n <= 0:
        return []
    total = total_hint or _count_records(train_data)
    stride = max(1, total // n) if total > n else 1

    out: List[dict] = []
    seen = 0
    for item in _iter_data_items(train_data):
        if len(out) >= n:
            break
        if _is_jsonl_path(item):
            p = Path(item)
            if not p.is_file():
                continue
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if seen % stride == 0:
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                        if len(out) >= n:
                            break
                    seen += 1
        elif isinstance(item, InputExample):
            if seen % stride == 0:
                out.append(item.to_dict())
            seen += 1
        elif isinstance(item, dict):
            if seen % stride == 0:
                out.append(item)
            seen += 1
    return out


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d ({s / 3600:.0f}h)"
