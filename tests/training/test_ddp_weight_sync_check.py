"""_check_ddp_weight_sync names the module groups whose weights differ across ranks, and passes identical ones.

Multi-process gloo cannot run on macOS (tests/test_trainer_distributed_integration.py), so all_gather is replaced by
a function returning this rank's checksums plus a second rank's; the method, the model and the comparison are real.
On 2 GPUs, rank 0 and rank 1 scored the SAME 1,500 eval records "wildly" differently (2026-10-20).
"""
import logging
import tempfile

import torch

import gliner2.training.trainer as trainer_module
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def _run(monkeypatch, caplog, perturb_group=None):
    tr = ExtractorTrainer(build_tiny_boundary_model(), TrainingConfig(output_dir=tempfile.mkdtemp(), bf16=False,
                                                                      fp16=False, validate_data=False))
    names = sorted({".".join(n.split(".")[:2]) for n, _ in tr.model.named_parameters()})

    def fake_all_gather(out, local):
        out[0].copy_(local)
        other = local.clone()
        if perturb_group is not None:
            other[names.index(perturb_group), 0] += 1.0
        out[1].copy_(other)

    monkeypatch.setattr(type(tr), "is_distributed", property(lambda self: True), raising=False)
    monkeypatch.setattr(trainer_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(trainer_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module.dist, "all_gather", fake_all_gather)
    with caplog.at_level(logging.INFO):
        tr._check_ddp_weight_sync()
    return [r.getMessage() for r in caplog.records if "[ddp-sync]" in r.getMessage()], names


def test_identical_ranks_pass(monkeypatch, caplog):
    lines, names = _run(monkeypatch, caplog)
    assert lines == [f"[ddp-sync] step 0 weights: all {len(names)} module groups identical across 2 ranks"]


def test_a_drifted_group_is_named(monkeypatch, caplog):
    lines, _ = _run(monkeypatch, caplog, perturb_group="record_decoder.field_proj")
    assert len(lines) == 1 and "1 of" in lines[0] and "DIFFER" in lines[0] and "record_decoder.field_proj=" in lines[0]


def test_gradient_mode_names_a_group_whose_gradients_differ(monkeypatch, caplog):
    tr = ExtractorTrainer(build_tiny_boundary_model(), TrainingConfig(output_dir=tempfile.mkdtemp(), bf16=False,
                                                                      fp16=False, validate_data=False))
    for p in tr.model.parameters():
        p.grad = torch.ones_like(p)
    names = sorted({".".join(n.split(".")[:2]) for n, _ in tr.model.named_parameters()})

    def fake_all_gather(out, local):
        out[0].copy_(local)
        other = local.clone()
        other[names.index("classifier.0"), 0] += 1.0
        out[1].copy_(other)

    monkeypatch.setattr(type(tr), "is_distributed", property(lambda self: True), raising=False)
    monkeypatch.setattr(trainer_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(trainer_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module.dist, "all_gather", fake_all_gather)
    with caplog.at_level(logging.INFO):
        tr._check_ddp_sync("grads")
    lines = [r.getMessage() for r in caplog.records if "[ddp-sync]" in r.getMessage()]
    assert len(lines) == 1 and "grads: 1 of" in lines[0] and "classifier.0=" in lines[0]
