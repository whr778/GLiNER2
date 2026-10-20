"""DDP correctness guards: the forward output is visible to pytree, and the trainer defaults to plain DDP.

DistributedDataParallel(static_graph=True) attaches its backward sink to the tensors torch.utils._pytree finds in
the forward output. An unregistered dataclass is one opaque leaf, so the sink never entered the graph and NO gradient
was averaged across 2 GPUs (2026-10-20). Plain DDP with every parameter touched averages correctly.
"""
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from gliner2.models.outputs import ExtractorOutput
from gliner2.training.trainer import TrainingConfig


def test_extractor_output_flattens_to_its_tensors_and_round_trips():
    out = ExtractorOutput(loss=torch.tensor(1.0, requires_grad=True), total_loss=torch.tensor(2.0), batch_size=3)
    leaves, spec = tree_flatten(out)
    assert any(isinstance(x, torch.Tensor) and x.requires_grad for x in leaves)   # the loss is a visible leaf
    back = tree_unflatten(leaves, spec)
    assert type(back) is ExtractorOutput and back["total_loss"].item() == 2.0 and back.batch_size == 3


def test_the_trainer_defaults_to_plain_ddp():
    assert TrainingConfig.ddp_static_graph is False
