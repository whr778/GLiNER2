"""CPU unit tests for the DataParallel split/reduce logic.

nn.DataParallel itself is CUDA-only (its scatter/gather only run with >=2 CUDA
devices), so these exercise the pure helpers that the subclass delegates to.
"""

import torch

from gliner2.processor import PreprocessedBatch
from gliner2.training.parallel import (
    _chunk_sizes,
    split_batch,
    reduce_loss_dicts,
)


def _make_batch(b: int) -> PreprocessedBatch:
    """A minimal length-b batch: two tensors + the per-sample lists used here."""
    return PreprocessedBatch(
        input_ids=torch.arange(b * 3).reshape(b, 3),
        attention_mask=torch.ones(b, 3),
        mapped_indices=[[(f"m{i}",)] for i in range(b)],
        schema_counts=list(range(b)),
        original_lengths=[3] * b,
        structure_labels=[[i] for i in range(b)],
        task_types=[["entities"] for _ in range(b)],
        text_tokens=[[f"t{i}"] for i in range(b)],
        schema_tokens_list=[[[f"s{i}"]] for i in range(b)],
        start_mappings=[[0] for _ in range(b)],
        end_mappings=[[1] for _ in range(b)],
        original_texts=[f"text{i}" for i in range(b)],
        original_schemas=[{"i": i} for i in range(b)],
        text_word_indices=torch.arange(b * 2).reshape(b, 2),
        text_word_counts=[2] * b,
        schema_special_indices=[[[i]] for i in range(b)],
    )


def test_chunk_sizes_partition_evenly_with_remainder_first():
    assert _chunk_sizes(5, 2) == [3, 2]
    assert _chunk_sizes(8, 4) == [2, 2, 2, 2]
    assert _chunk_sizes(3, 4) == [1, 1, 1, 0]


def test_split_batch_partitions_tensors_and_lists():
    batch = _make_batch(5)
    cpu = torch.device("cpu")
    subs = split_batch(batch, [cpu, cpu])

    assert [len(s) for s in subs] == [3, 2]
    # tensors split along dim 0 and concatenate back to the original
    assert torch.equal(torch.cat([s.input_ids for s in subs]), batch.input_ids)
    assert torch.equal(
        torch.cat([s.text_word_indices for s in subs]), batch.text_word_indices
    )
    # per-sample lists sliced to match the same chunk boundaries
    assert subs[0].original_texts == ["text0", "text1", "text2"]
    assert subs[1].original_texts == ["text3", "text4"]
    assert subs[1].structure_labels == [[3], [4]]


def test_split_batch_uses_fewer_replicas_than_devices_when_batch_is_small():
    batch = _make_batch(1)
    cpu = torch.device("cpu")
    subs = split_batch(batch, [cpu, cpu, cpu])
    assert len(subs) == 1
    assert len(subs[0]) == 1


def test_reduce_sums_loss_tensors_and_preserves_grad():
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    out_a = {"total_loss": a * 1, "classification_loss": a * 1, "batch_size": 2}
    out_b = {"total_loss": b * 1, "classification_loss": b * 1, "batch_size": 3}

    combined = reduce_loss_dicts([out_a, out_b], torch.device("cpu"))

    assert combined["total_loss"].item() == 5.0          # summed, not meaned
    assert combined["batch_size"] == 5                    # ints summed
    assert combined["total_loss"].requires_grad
    combined["total_loss"].backward()
    assert a.grad.item() == 1.0 and b.grad.item() == 1.0  # dL/da == dL/db == 1


def test_reduce_handles_empty_shard():
    """A replica whose samples all failed returns _empty_loss_dict() (same keys,
    all-zero). Reducing it with a normal dict must add the zeros, not crash."""
    a = torch.tensor(4.0, requires_grad=True)
    normal = {
        "total_loss": a * 1, "classification_loss": a * 1,
        "structure_loss": a * 1, "count_loss": a * 1, "batch_size": 3,
    }
    empty = {  # mirrors Extractor._empty_loss_dict()
        "total_loss": torch.tensor(0.0, requires_grad=True),
        "classification_loss": torch.tensor(0.0),
        "structure_loss": torch.tensor(0.0),
        "count_loss": torch.tensor(0.0),
        "batch_size": 0,
    }
    combined = reduce_loss_dicts([normal, empty], torch.device("cpu"))
    assert combined["total_loss"].item() == 4.0   # normal + 0
    assert combined["batch_size"] == 3            # 3 + 0
    combined["total_loss"].backward()
    assert a.grad.item() == 1.0
