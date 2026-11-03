"""A record group that exceeds instance capacity is logged and skipped.

ValueError and IndexError from the role loss propagate. TargetCapacityError is a
ValueError subclass and is the one capacity miss that stays non-fatal.
"""
import logging
import random
import sys
from pathlib import Path

import pytest
import torch

import gliner2.models.boundary.records as R
from gliner2.processing.targets import TargetCapacityError
from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "train"))
import train as T  # noqa: E402

TEXT = "hackers attacked the bank and stole money from customers yesterday ."
SCHEMA = {"events": [{"event_type": "attack", "triggers": ["attacked"],
                      "arguments": [{"role": "attacker", "entity": "hackers"},
                                    {"role": "target", "entity": "bank"}]}]}


def _forward(monkeypatch, exc):
    def boom(*_args, **_kwargs):
        raise exc

    monkeypatch.setattr(R, "compute_group_loss", boom)
    random.seed(0)
    torch.manual_seed(0)
    model = build_tiny_boundary_model()
    T._apply_boundary_head_overrides(model, {})
    model.train()
    batch = ExtractorCollator(
        model.processor, is_training=True, architecture="boundary", event_records=True,
    )([(TEXT, SCHEMA)])
    return model(batch)


@pytest.mark.parametrize("exc", [ValueError("role loss failed"), IndexError("role column")])
def test_role_loss_errors_propagate(monkeypatch, exc):
    with pytest.raises(type(exc), match=str(exc)):
        _forward(monkeypatch, exc)


def test_capacity_miss_is_logged_and_the_step_continues(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="gliner2.models.boundary.model"):
        out = _forward(monkeypatch, TargetCapacityError("more gold instances than hypotheses"))
    assert torch.isfinite(out.loss)
    assert "record loss skipped" in caplog.text
    assert "more gold instances than hypotheses" in caplog.text
