"""Under DDP every trainable parameter must get a gradient every step, or the ranks drift.

On 2x H100 (ddp_sync_probe, 2026-10-20) the default static_graph=True let 25 of 33 module groups drift by step 100,
and both alternative DDP settings crashed, because which parameters get a gradient varies by batch. The trainer now
adds an exactly-zero term touching every trainable parameter when distributed. Real model, real trainer step; only
the process group is absent (gloo cannot run on macOS), so `is_distributed` is forced and the consensus all_reduce off.
"""
import tempfile

import torch

from gliner2.training.data import create_entity_example
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def _grads_after_one_step(monkeypatch, distributed: bool):
    cfg = TrainingConfig(output_dir=tempfile.mkdtemp(), bf16=False, fp16=False, validate_data=False, batch_size=1,
                         num_workers=0, ddp_consensus_check=False)
    tr = ExtractorTrainer(build_tiny_boundary_model(), cfg)
    example = create_entity_example("apple released iphone .", {"company": ["apple"]}, {"company": "a company"})
    ds = tr._prepare_data([example], True)
    batch = next(iter(tr._create_dataloader(ds, 1, shuffle=False, is_training=True))).to(tr.device)
    if distributed:   # after the dataloader: a distributed one would need a real process group for its sampler
        monkeypatch.setattr(type(tr), "is_distributed", property(lambda self: True), raising=False)
    tr.model.train()
    tr._backward_one(batch, 0, False, None)
    return {n: p.grad for n, p in tr.model.named_parameters() if p.requires_grad}


def test_without_ddp_an_entity_batch_leaves_some_parameters_without_gradient(monkeypatch):
    grads = _grads_after_one_step(monkeypatch, distributed=False)
    assert any(g is None for g in grads.values())          # the situation DDP cannot handle


def test_under_ddp_every_parameter_gets_a_gradient(monkeypatch):
    grads = _grads_after_one_step(monkeypatch, distributed=True)
    missing = [n for n, g in grads.items() if g is None]
    assert not missing, f"{len(missing)} parameters without gradient, e.g. {missing[:5]}"
