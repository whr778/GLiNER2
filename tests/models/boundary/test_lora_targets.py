"""Architecture-aware LoRA target resolution for the boundary model."""

from __future__ import annotations

from gliner2.training.lora import _resolve_targets, _task_module_names
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def test_boundary_task_module_names():
    model = build_tiny_boundary_model()
    names = _task_module_names(model)
    assert "boundary_head" in names
    assert "classifier" in names


def test_extractive_head_alias_targets_boundary_head_not_encoder():
    model = build_tiny_boundary_model()
    resolved = _resolve_targets(model, ["extractive_head"])
    assert resolved, "expected boundary_head Linear layers"
    assert all(name.startswith("boundary_head") for name in resolved)
    assert not any(name.startswith("encoder.") for name in resolved)


def test_classification_head_alias_targets_classifier():
    model = build_tiny_boundary_model()
    resolved = _resolve_targets(model, ["classification_head"])
    assert resolved
    assert all(name.startswith("classifier") for name in resolved)


def test_all_task_heads_alias_covers_boundary_and_classifier():
    model = build_tiny_boundary_model()
    resolved = _resolve_targets(model, ["all_task_heads"])
    assert any(n.startswith("boundary_head") for n in resolved)
    assert any(n.startswith("classifier") for n in resolved)
    assert not any(n.startswith("encoder.") for n in resolved)


def test_encoder_alias_still_resolves_encoder_only():
    model = build_tiny_boundary_model()
    resolved = _resolve_targets(model, ["encoder"])
    assert resolved
    assert all(name.startswith("encoder.") for name in resolved)


def test_mention_link_target_resolves_when_enabled():
    model = build_tiny_boundary_model()
    assert "mention_link" not in _task_module_names(model)
    model.enable_mention_link()
    assert "mention_link" in _task_module_names(model)
    resolved = _resolve_targets(model, ["mention_link"])
    assert resolved
    assert all(name.startswith("mention_link.") for name in resolved)
    all_heads = _resolve_targets(model, ["all_task_heads"])
    assert any(name.startswith("mention_link.") for name in all_heads)


def test_modernbert_and_llama_style_encoder_linear_names_resolve():
    import torch.nn as nn

    class FakeModernBertLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.Wqkv = nn.Linear(16, 48)
            self.Wo = nn.Linear(16, 16)
            self.Wi = nn.Linear(16, 32)
            self.q_proj = nn.Linear(16, 16)
            self.up_proj = nn.Linear(16, 32)

    class FakeExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FakeModernBertLayer()
            self.boundary_head = nn.Linear(16, 16)

    resolved = _resolve_targets(FakeExtractor(), ["encoder"])
    assert set(resolved) == {
        "encoder.Wqkv", "encoder.Wo", "encoder.Wi",
        "encoder.q_proj", "encoder.up_proj",
    }


def test_lora_optimizer_splits_encoder_lr_and_task_lr(tmp_path):
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    model = build_tiny_boundary_model()
    cfg = TrainingConfig(
        output_dir=str(tmp_path),
        use_lora=True,
        lora_r=4,
        lora_alpha=8.0,
        lora_target_modules=["encoder", "boundary_head", "relation_scorer"],
        encoder_lr=1.0e-4,
        task_lr=3.0e-4,
        encoder_lr_backbone_only=True,
        bf16=False,
        fp16=False,
    )
    trainer = GLiNER2Trainer(model, cfg)
    opt = trainer._create_optimizer()
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1.0e-4
    assert opt.param_groups[1]["lr"] == 3.0e-4
    assert len(opt.param_groups[0]["params"]) > 0
    assert len(opt.param_groups[1]["params"]) > 0
