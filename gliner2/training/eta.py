"""One-shot training-time estimate.

Prints a wall-clock projection at the start of a training run by:

1. Counting total records cheaply via JSONL line counts.
2. Loading a small warmup slice from the first training source.
3. Running ``n_warmup`` forward+backward passes through the live model
   to measure the per-micro-batch time.
4. Multiplying out to total micro-batches and printing the projection.

The warmup runs on the same device and dtype the trainer will use, so
the estimate reflects real per-step cost (data loading, encoding, span
representation, loss). Model state is restored — ``zero_grad`` is called
after the warmup loop and the optimizer hasn't stepped.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, List

from torch.utils.data import DataLoader

from gliner2.training.data import InputExample


def estimate_eta(
    model,
    train_data: Any,
    config,
    n_warmup: int = 8,
) -> None:
    """Print a one-shot ETA estimate for the upcoming ``trainer.train`` call.

    Args:
        model: A loaded ``Extractor`` / ``GLiNER2`` instance.
        train_data: Same value that will be passed as ``train_data`` to
            ``trainer.train`` — a JSONL path, a list of JSONL paths, a
            list of dicts, or a list of ``InputExample``.
        config: The same ``TrainingConfig`` passed to ``GLiNER2Trainer``.
            Read for ``batch_size``, ``gradient_accumulation_steps``,
            ``num_epochs``, ``num_workers``, ``pin_memory``.
        n_warmup: Number of micro-batches to time after a single
            untimed warmup batch.
    """
    # 1) Total record count — stream-counted so we don't load 1M-row corpora.
    n_total = _count_records(train_data)
    if n_total == 0:
        print("[ETA] No training records found; skipping estimate.")
        return

    effective_batch = max(1, int(config.batch_size) * max(1, int(config.gradient_accumulation_steps)))
    total_opt_steps = (n_total * int(config.num_epochs) + effective_batch - 1) // effective_batch
    total_micro_batches = total_opt_steps * max(1, int(config.gradient_accumulation_steps))

    # 2) Build a small in-memory sample for timing from the first source.
    sample_records = _warmup_records(train_data, n=(n_warmup + 2) * int(config.batch_size))
    if not sample_records:
        print("[ETA] Could not build warmup sample; skipping estimate.")
        return

    # 3) Time n_warmup micro-batches through the live model.
    from gliner2.training.trainer import ExtractorDataset

    sample_ds = ExtractorDataset(
        sample_records, max_samples=-1, shuffle=False, validate=False,
    )
    if len(sample_ds) == 0:
        print("[ETA] Empty warmup dataset; skipping estimate.")
        return

    model.train()
    model.processor.change_mode(is_training=True)
    collator = model.processor.collate_fn_train

    loader = DataLoader(
        sample_ds,
        batch_size=int(config.batch_size),
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    timed = 0
    t_start = None
    for i, batch in enumerate(loader):
        try:
            outputs = model(batch)
            loss = outputs.get("total_loss")
            if loss is not None and loss.requires_grad:
                loss.backward()
        except Exception as e:
            print(f"[ETA] Warmup step {i} raised {type(e).__name__}: {e}")
            break
        finally:
            model.zero_grad(set_to_none=True)

        if i == 0:
            t_start = time.perf_counter()
        else:
            timed = i
        if i >= n_warmup:
            break

    if t_start is None or timed == 0:
        print("[ETA] Too few warmup batches to time; skipping estimate.")
        return

    per_micro = (time.perf_counter() - t_start) / timed
    per_opt_step = per_micro * max(1, int(config.gradient_accumulation_steps))
    eta_seconds = per_micro * total_micro_batches

    print("=" * 60)
    print("📊 Training Time Estimate")
    print("=" * 60)
    print(f"Examples         : {n_total:,}")
    print(f"Epochs           : {int(config.num_epochs)}")
    print(f"Effective batch  : {effective_batch}")
    print(f"Optimisation steps: {total_opt_steps:,}")
    print(f"Per micro-batch  : {per_micro * 1000:.0f}ms")
    print(f"Per optim. step  : {per_opt_step * 1000:.0f}ms")
    print(f"Projected ETA    : {_fmt_seconds(eta_seconds)}")
    print("=" * 60)


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
    # Anything else (TrainingDataset, ExtractorDataset, etc.) — count len() once.
    try:
        for item in train_data:
            yield item
    except TypeError:
        return


def _count_records(train_data: Any) -> int:
    """Cheaply count records across JSONL files / in-memory lists."""
    total = 0
    for item in _iter_data_items(train_data):
        if _is_jsonl_path(item):
            p = Path(item)
            if not p.is_file():
                continue
            with p.open() as fh:
                for line in fh:
                    if line.strip():
                        total += 1
        else:
            total += 1
    return total


def _warmup_records(train_data: Any, n: int) -> List[dict]:
    """Pull the first ``n`` records as raw dicts for timing.

    Skips empty/malformed lines silently. Reads at most one JSONL file —
    we just want a representative slice, not a balanced mix.
    """
    import json

    out: List[dict] = []
    for item in _iter_data_items(train_data):
        if len(out) >= n:
            break
        if _is_jsonl_path(item):
            p = Path(item)
            if not p.is_file():
                continue
            with p.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    out.append(rec)
                    if len(out) >= n:
                        break
        elif isinstance(item, InputExample):
            out.append(item.to_dict())
        elif isinstance(item, dict):
            out.append(item)
    return out


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d ({s / 3600:.0f}h)"
