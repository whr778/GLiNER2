"""``model.boundary_head`` overrides must reach the model on the ``pretrained`` path.

``_build_model`` applies leftover config keys with ``setattr(model.config, key,
value)``. That works for flat keys, but ``boundary_settings`` is built from
``config.boundary_head`` inside the model's ``__init__``, which ``from_pretrained``
has already run -- so a ``boundary_head`` override landed on the config and was
never read. Every such override in every ``pretrained:`` config was decorative.

It surfaced as a treatment arm that was a silent duplicate of its control: a
config setting ``boundary_head.task_loss_weights`` built a model reporting
``task_loss_weights=None``. The arm trained for ten minutes before the check
caught it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))

from train import _apply_boundary_head_overrides  # noqa: E402

from gliner2.configuration import BoundaryHeadSettings  # noqa: E402


class FakeConfig:
    def __init__(self, boundary_head):
        self.boundary_head = boundary_head


class FakeModel:
    """Stands in for a loaded checkpoint: a config carrying the CHECKPOINT's
    boundary_head, plus the settings object built from it at construction."""

    class FakeHead:
        """The head keeps its OWN settings reference, and copies two values out
        of it at construction -- both true of the real BoundaryHead."""

        def __init__(self, settings):
            self.settings = settings
            self.hard_negatives_per_positive = settings.hard_negatives_per_positive
            self.minimum_hard_negatives = settings.minimum_hard_negatives

    def __init__(self, boundary_head):
        self.config = FakeConfig(dict(boundary_head))
        self.boundary_settings = BoundaryHeadSettings(**boundary_head)
        self.boundary_head = FakeModel.FakeHead(self.boundary_settings)


CHECKPOINT = {"enable_relations": True, "enable_records": True}


def test_loss_override_rebuilds_the_settings_object():
    model = FakeModel(CHECKPOINT)
    assert model.boundary_settings.task_loss_weights is None  # precondition

    _apply_boundary_head_overrides(
        model, {**CHECKPOINT, "task_loss_weights": {"events": 2.0}}
    )

    assert model.boundary_settings.task_loss_weights == {"events": 2.0}


def test_override_is_merged_not_replaced():
    """The checkpoint's own boundary_head keys must survive an override that
    does not mention them."""
    model = FakeModel({**CHECKPOINT, "record_loss_weight": 3.0})

    _apply_boundary_head_overrides(model, {"task_loss_weights": {"events": 2.0}})

    assert model.boundary_settings.record_loss_weight == 3.0
    assert model.boundary_settings.enable_records is True
    assert model.boundary_settings.task_loss_weights == {"events": 2.0}


def test_structural_override_raises_rather_than_half_applying():
    """enable_records sizes modules built during __init__. Applying it after the
    fact would leave the settings and the modules disagreeing."""
    model = FakeModel(CHECKPOINT)

    with pytest.raises(SystemExit, match="structural"):
        _apply_boundary_head_overrides(model, {"enable_records": False})


def test_structural_key_matching_the_checkpoint_is_not_a_conflict():
    """Configs restate enable_relations/enable_records to document intent. That
    must stay legal when it agrees with the checkpoint."""
    model = FakeModel(CHECKPOINT)

    _apply_boundary_head_overrides(
        model, {**CHECKPOINT, "task_loss_weights": {"events": 0.5}}
    )

    assert model.boundary_settings.task_loss_weights == {"events": 0.5}


def test_overrides_reach_the_HEAD_settings_not_just_the_model():
    """dfaaa2a rebuilt `model.boundary_settings`, but the head holds its OWN
    reference built in __init__. Every knob the head reads through `self.settings`
    -- the soft_iou/rerank/proposal/count weights, boundary_negative_weight,
    negative_query_ratio, task_loss_weight_scope -- stayed at the checkpoint value,
    so a config setting them produced a treatment arm inert in exactly the way that
    commit was meant to end. Measured before the fix: scope="all" on the model and
    "span" on the head."""
    model = FakeModel(CHECKPOINT)
    assert model.boundary_head.settings.task_loss_weight_scope == "span"

    _apply_boundary_head_overrides(
        model,
        {"task_loss_weight_scope": "all", "rerank_listwise_weight": 0.77,
         "minimum_hard_negatives": 9},
    )

    assert model.boundary_settings.task_loss_weight_scope == "all"
    assert model.boundary_head.settings.task_loss_weight_scope == "all"
    assert model.boundary_head.settings is model.boundary_settings
    assert model.boundary_head.settings.rerank_listwise_weight == 0.77
    # Copied at construction rather than read live, so assigning settings alone
    # would not move it.
    assert model.boundary_head.minimum_hard_negatives == 9
