"""Trainer generalization: alias, strict config, disjoint optimizer groups."""

from __future__ import annotations

import pytest
import torch

from gliner2.training.trainer import (
    ExtractorTrainer,
    GLiNER2Trainer,
    TrainingConfig,
)
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model


def test_gliner2trainer_is_extractor_trainer_alias():
    assert GLiNER2Trainer is ExtractorTrainer


def test_strict_training_defaults():
    config = TrainingConfig()
    assert config.strict_training is True
    assert config.allow_invalid_samples is False
    assert config.log_proposal_metrics is True


def test_optimizer_groups_are_disjoint_and_complete(tmp_path):
    model = build_tiny_span_model()
    config = TrainingConfig(output_dir=str(tmp_path / "out"), num_workers=0, fp16=False)
    trainer = ExtractorTrainer(model=model, config=config)

    optimizer = trainer._create_optimizer()  # asserts disjoint/complete internally
    grouped = sum(len(g["params"]) for g in optimizer.param_groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert grouped == trainable

    # Encoder params land in the encoder LR group, task params in the task group.
    enc_group, task_group = optimizer.param_groups
    assert enc_group["lr"] == config.encoder_lr
    assert task_group["lr"] == config.task_lr


def _trainer_selecting_on(tmp_path, metric_for_best):
    config = TrainingConfig(
        output_dir=str(tmp_path / "out"),
        num_workers=0,
        fp16=False,
        metric_for_best=metric_for_best,
        greater_is_better=True,
    )
    return ExtractorTrainer(model=build_tiny_span_model(), config=config)


def test_absent_metric_for_best_raises_instead_of_falling_back(tmp_path):
    """A missing metric must not silently become eval_loss.

    The MAVEN A/B selected its worst checkpoint this way: compute_metrics
    returned {}, selection fell back to eval_loss, and greater_is_better=True
    then maximized the loss.
    """
    trainer = _trainer_selecting_on(tmp_path, "eval_event_strict_micro_f1")
    metrics = {"eval_loss": 0.88, "step": 91, "epoch": 1}

    with pytest.raises(ValueError, match="eval_event_strict_micro_f1"):
        trainer._selection_metric(metrics)
    with pytest.raises(ValueError, match="eval_event_strict_micro_f1"):
        trainer._check_early_stopping(metrics)


def test_present_metric_for_best_is_used(tmp_path):
    trainer = _trainer_selecting_on(tmp_path, "eval_event_strict_micro_f1")
    metrics = {"eval_loss": 0.88, "eval_event_strict_micro_f1": 0.7327}

    assert trainer._selection_metric(metrics) == 0.7327


def test_default_eval_loss_selection_still_works(tmp_path):
    trainer = _trainer_selecting_on(tmp_path, "eval_loss")

    assert trainer._selection_metric({"eval_loss": 0.42}) == 0.42
