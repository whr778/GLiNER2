"""A `_used` gate must reach logged metrics without iterating ExtractorOutput.

Regression for 2026-09-19: a first attempt did `for k in outputs`, but ExtractorOutput has
.get(), a STRING-ONLY __getitem__ and __contains__, and NO __iter__ -- so iteration falls
back to integer indexing and dies with KeyError: 0 on the first logging step. It killed both
restarted A/B arms at step 20. `hasattr(outputs, "get")` passed and proved nothing, and a
hand-written simulation of the dict also passed; only the real object shows it.
"""

import pytest
import torch

from gliner2.models.outputs import ExtractorOutput


def _lift(outputs) -> dict:
    """The trainer's gate-lifting step, isolated."""
    logged = {}
    losses = getattr(outputs, "losses", None)
    if isinstance(losses, dict):
        for k, v in losses.items():
            if str(k).endswith("_used"):
                try:
                    logged[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
    return logged


def test_gate_reaches_logged_metrics():
    o = ExtractorOutput(losses={"total_loss": torch.tensor(1.0),
                                "absent_negatives_used": torch.tensor(7.0)})
    assert _lift(o) == {"absent_negatives_used": 7.0}


def test_iterating_the_output_object_still_raises():
    """The trap this regression guards. If ExtractorOutput ever gains __iter__ this test
    fails, and the comment in trainer.py explaining WHY we read .losses must be revisited."""
    o = ExtractorOutput(losses={"absent_negatives_used": torch.tensor(1.0)})
    with pytest.raises(KeyError):
        for _ in o:
            pass


def test_no_gate_key_yields_nothing():
    o = ExtractorOutput(losses={"total_loss": torch.tensor(1.0)})
    assert _lift(o) == {}


def test_missing_losses_is_safe():
    assert _lift(ExtractorOutput()) == {}


def test_non_numeric_gate_is_skipped_not_fatal():
    o = ExtractorOutput(losses={"weird_used": "not a number"})
    assert _lift(o) == {}
