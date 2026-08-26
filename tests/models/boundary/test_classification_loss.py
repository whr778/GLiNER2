"""Boundary classification loss: normalization, weighting, loud mismatch (3.4)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import logging

import torch

from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


CLS_EXAMPLE = (
    "apple released iphone .",
    {"classifications": [{"task": "sentiment", "labels": ["positive", "negative"], "true_label": ["positive"]}]},
)


def _batch_and_core(model):
    model.eval()
    collator = ExtractorCollator(model.processor, is_training=True, architecture="boundary")
    batch = collator([CLS_EXAMPLE])
    with torch.no_grad():
        core = model._encode_core(batch)
    return batch, core


def test_classification_loss_scales_with_weight():
    model = build_tiny_boundary_model()
    batch, core = _batch_and_core(model)

    with torch.no_grad():
        base = model._classification_loss(batch, core)
        assert torch.isfinite(base) and float(base) > 0.0

        model.boundary_settings = replace(model.boundary_settings, classification_loss_weight=3.0)
        scaled = model._classification_loss(batch, core)
    assert float(scaled) == pytest.approx(3.0 * float(base), rel=1e-5)


def test_classification_loss_is_normalized_mean_scale():
    # Normalized by supervised label count -> a per-label mean, i.e. bounded on
    # the same scale as the boundary marginal losses (not an unbounded sum).
    model = build_tiny_boundary_model()
    batch, core = _batch_and_core(model)
    with torch.no_grad():
        loss = model._classification_loss(batch, core)
    # 2 labels here; a mean BCE is comfortably < 100 (a raw sum could exceed it
    # for many labels). Sanity bound that the value is a mean, not a sum.
    assert 0.0 < float(loss) < 100.0


def test_classification_loss_skips_shape_mismatch():
    model = build_tiny_boundary_model()
    batch, core = _batch_and_core(model)
    group_index = core["cls_specs"][0][0]["group_index"]
    # Corrupt the label vector length so it disagrees with the logits.
    labels = list(batch.structure_labels[0][group_index])
    batch.structure_labels[0][group_index] = labels + [0.0]

    with torch.no_grad():
        loss = model._classification_loss(batch, core)
    # Shape mismatch groups are gracefully skipped, yielding zero loss.
    assert float(loss) == 0.0


def test_classification_loss_shape_mismatch_is_never_silent(caplog):
    """Our stricter contract, adapted: upstream skips instead of raising, but a group
    that contributes NO supervision must still be visible. A head that quietly receives
    nothing is indistinguishable from a head that cannot learn."""
    model = build_tiny_boundary_model()
    batch, core = _batch_and_core(model)
    group_index = core["cls_specs"][0][0]["group_index"]
    labels = list(batch.structure_labels[0][group_index])
    batch.structure_labels[0][group_index] = labels + [0.0]

    with caplog.at_level(logging.WARNING, logger="gliner2.models.boundary.model"):
        with torch.no_grad():
            loss = model._classification_loss(batch, core)

    assert float(loss) == 0.0
    assert any("shape mismatch" in r.getMessage() for r in caplog.records), \
        "the skipped group must be logged, not silent"
