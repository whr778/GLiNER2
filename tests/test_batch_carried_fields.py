"""A collated batch keeps its per-sample side fields through the DataLoader's pin and the model's to().

`pin_memory()` and `to()` rebuild the batch field by field, and torch's DataLoader calls the batch's
OWN `pin_memory()` when `pin_memory: true` (every CUDA run). `typed_spans` used to be a loose
attribute: pinned, it vanished and the typed-role margin silently received no inputs.
"""
import torch
from torch.utils.data._utils import pin_memory as PM

from gliner2.processor import PreprocessedBatch


def _batch():
    t = torch.zeros(1, 4, dtype=torch.long)
    return PreprocessedBatch(input_ids=t, attention_mask=t, mapped_indices=[[]], schema_counts=[0],
                             original_lengths=[4], structure_labels=[[]], task_types=[[]], text_tokens=[["a"]],
                             schema_tokens_list=[[]], start_mappings=[[0]], end_mappings=[[1]],
                             original_texts=["a."], original_schemas=[{}],
                             coref_mentions=([(0, 0, 0), (0, 0, 1)],), typed_spans=({(0, 1): ("Person",)},))


def test_to_keeps_side_fields():
    b = _batch().to(torch.device("cpu"))
    assert b.typed_spans == ({(0, 1): ("Person",)},) and b.coref_mentions == ([(0, 0, 0), (0, 0, 1)],)


def test_dataloader_pin_keeps_side_fields(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)   # no CUDA needed
    b = PM.pin_memory(_batch())
    assert b.typed_spans == ({(0, 1): ("Person",)},) and b.coref_mentions == ([(0, 0, 0), (0, 0, 1)],)
