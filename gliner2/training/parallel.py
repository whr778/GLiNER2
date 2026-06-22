"""Multi-GPU training via ``nn.DataParallel`` for the GLiNER2 trainer.

``nn.DataParallel`` replicates the model on each GPU, splits the input batch
along dim 0, runs the replicas in parallel, and gathers their outputs on the
primary device. Two things make the stock wrapper insufficient here:

1. The model is called as ``model(batch)`` where ``batch`` is a custom
   :class:`~gliner2.processor.PreprocessedBatch` (padded tensors plus
   per-sample Python lists). DataParallel's default scatter cannot split a
   custom object, so ``scatter`` is overridden to slice the batch along dim 0.
2. ``forward`` returns a dict of *summed* per-sample losses (plus an int
   ``batch_size``). The default gather would ``cat`` scalar tensors and crash
   on the int. ``gather`` is overridden to **sum** the per-replica loss tensors
   -- which reproduces single-GPU semantics exactly, since the model already
   sums per-sample losses (``Σ_g Σ_{i∈chunk_g} l_i == Σ_i l_i``) -- and to sum
   the int ``batch_size``. The reduced dict has the same shape as the
   single-GPU output, so the trainer's loss/backward/logging code is unchanged.

The split/slice/reduce logic lives in module-level functions so it can be unit
tested on CPU; the subclass methods are thin wrappers. ``nn.DataParallel``
itself is CUDA-only and only exercises scatter/gather with >=2 CUDA devices.

Known caveat: ``torch.autocast`` is thread-local and DataParallel runs replicas
in worker threads, so AMP (fp16/bf16) does not apply on the replicas -- they run
in fp32. Correctness is unaffected, but per-GPU memory is higher than a
single-GPU AMP run. For full AMP multi-GPU, use DistributedDataParallel instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from gliner2.processor import PreprocessedBatch


def _chunk_sizes(total: int, n: int) -> List[int]:
    """Split ``total`` items into ``n`` near-even contiguous chunks (larger first)."""
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _slice_batch(batch: PreprocessedBatch, start: int, end: int,
                 device: torch.device) -> PreprocessedBatch:
    """Return the ``[start:end]`` sub-batch with tensors moved to ``device``."""
    def t(x):
        return None if x is None else x[start:end].to(device)

    def l(x):
        return None if x is None else x[start:end]

    return PreprocessedBatch(
        input_ids=t(batch.input_ids),
        attention_mask=t(batch.attention_mask),
        mapped_indices=l(batch.mapped_indices),
        schema_counts=l(batch.schema_counts),
        original_lengths=l(batch.original_lengths),
        structure_labels=l(batch.structure_labels),
        task_types=l(batch.task_types),
        text_tokens=l(batch.text_tokens),
        schema_tokens_list=l(batch.schema_tokens_list),
        start_mappings=l(batch.start_mappings),
        end_mappings=l(batch.end_mappings),
        original_texts=l(batch.original_texts),
        original_schemas=l(batch.original_schemas),
        text_word_indices=t(batch.text_word_indices),
        text_word_counts=l(batch.text_word_counts),
        schema_special_indices=l(batch.schema_special_indices),
    )


def split_batch(batch: PreprocessedBatch,
                devices: Sequence[torch.device]) -> List[PreprocessedBatch]:
    """Split ``batch`` along dim 0 into one sub-batch per device.

    Uses at most ``len(batch)`` devices (smaller batches use fewer replicas).
    """
    n = min(len(devices), len(batch))
    sizes = _chunk_sizes(len(batch), n)
    subs: List[PreprocessedBatch] = []
    start = 0
    for size, device in zip(sizes, devices):
        subs.append(_slice_batch(batch, start, start + size, device))
        start += size
    return subs


def reduce_loss_dicts(outputs: List[Dict[str, Any]],
                      output_device: Any) -> Dict[str, Any]:
    """Combine per-replica forward dicts into one matching the single-GPU shape.

    Loss tensors are summed on ``output_device`` (cross-device ``.to`` keeps the
    autograd graph), int/float values (e.g. ``batch_size``) are summed, and any
    other value falls back to the first replica's.
    """
    combined: Dict[str, Any] = {}
    for key in outputs[0]:
        values = [out[key] for out in outputs]
        first = values[0]
        if isinstance(first, torch.Tensor):
            acc = values[0].to(output_device)
            for v in values[1:]:
                acc = acc + v.to(output_device)
            combined[key] = acc
        elif isinstance(first, (int, float)):
            combined[key] = sum(values)
        else:
            combined[key] = first
    return combined


class BatchDataParallel(nn.DataParallel):
    """``nn.DataParallel`` that scatters a ``PreprocessedBatch`` and sum-reduces
    the model's loss dict. See module docstring for the reduction rationale."""

    def forward(self, *inputs, **kwargs):
        batch = inputs[0]
        # No CUDA device ids, or an empty batch (which has nothing to scatter):
        # run the underlying module directly to match single-GPU behaviour.
        if not self.device_ids or len(batch) == 0:
            return self.module(*inputs, **kwargs)
        return super().forward(*inputs, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        batch = inputs[0]
        devices = [torch.device("cuda", i) for i in device_ids]
        subs = split_batch(batch, devices)
        scattered_inputs = [(sub,) for sub in subs]
        scattered_kwargs = [dict(kwargs) for _ in subs]
        return scattered_inputs, scattered_kwargs

    def gather(self, outputs, output_device):
        return reduce_loss_dicts(outputs, output_device)
