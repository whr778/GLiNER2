"""absent_reduction: the start/end boundary loss with absent queries averaged over their OWN cells.

`pooled` (default) divides every cell by N_P + N_A, so gold-bearing cells are scaled by
pi_P = N_P / (N_P + N_A). `separate` computes present_loss_scale * mean_P + absent_loss_weight
* mean_A. The identity pooled == separate(scale=pi_P, weight=pi_P * N_A / N_P) pins the
implementation to the equation; the other tests pin that only start/end move.
"""
import random
import sys
from pathlib import Path

import pytest
import torch

import gliner2.models.boundary.model as M
from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head
from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "train"))
import train as T  # noqa: E402

TEXT = "apple released iphone ."
WITH_ABSENT = {"entities": {"company": ["apple"], "product": ["iphone"], "venue": [], "person": []}}
NO_ABSENT = {"entities": {"company": ["apple"], "product": ["iphone"]}}


def _losses(schema, monkeypatch, **bh):
    """One seeded training forward with EVERY reduction key set (overrides merge)."""
    cells = {}
    monkeypatch.setattr(M, "_note_absent_cells", lambda mode, kept, absent: cells.update(
        mode=mode, kept=int(kept), absent=int(absent)))
    random.seed(0)
    torch.manual_seed(0)
    model = build_tiny_boundary_model()
    T._apply_boundary_head_overrides(model, {"absent_reduction": "pooled", "absent_loss_weight": 1.0,
                                             "present_loss_scale": 1.0, **bh})
    model.train()
    random.seed(0)
    batch = ExtractorCollator(model.processor, is_training=True, architecture="boundary")([(TEXT, schema)])
    torch.manual_seed(0)
    out = model(batch)
    return {k: float(v.detach()) for k, v in out.losses.items() if torch.is_tensor(v) and v.dim() == 0}, cells


def test_default_is_pooled_and_bad_values_are_refused():
    assert BoundaryHeadSettings(**validate_boundary_head({})).absent_reduction == "pooled"
    for bad in ({"absent_reduction": "split"}, {"absent_loss_weight": -0.1}, {"present_loss_scale": 0.0}):
        with pytest.raises(ValueError):
            validate_boundary_head(bad)


def test_separate_reproduces_pooled_under_the_identity(monkeypatch):
    pooled, cells = _losses(WITH_ABSENT, monkeypatch)
    n_a = cells["absent"]
    n_p = cells["kept"] - n_a
    assert n_a > 0 and n_p > 0, "the fixture must carry both present and absent cells"
    pi = n_p / (n_p + n_a)
    ident, again = _losses(WITH_ABSENT, monkeypatch, absent_reduction="separate",
                           present_loss_scale=pi, absent_loss_weight=pi * n_a / n_p)
    assert again["mode"] == "separate"
    assert (again["kept"], again["absent"]) == (cells["kept"], cells["absent"]), "the two forwards saw different batches"
    for key in ("start_loss", "end_loss", "total_loss"):
        assert ident[key] == pytest.approx(pooled[key], rel=1e-5), key


def test_separate_moves_only_the_boundary_terms(monkeypatch):
    pooled, _ = _losses(WITH_ABSENT, monkeypatch)
    separate, cells = _losses(WITH_ABSENT, monkeypatch, absent_reduction="separate")
    assert cells["mode"] == "separate"
    assert separate["start_loss"] != pytest.approx(pooled["start_loss"], rel=1e-4)
    assert separate["end_loss"] != pytest.approx(pooled["end_loss"], rel=1e-4)
    for key in ("pair_loss", "abstention_loss", "count_loss"):
        assert separate[key] == pytest.approx(pooled[key], rel=1e-6), key


def test_without_absent_queries_separate_equals_pooled(monkeypatch):
    pooled, cells = _losses(NO_ABSENT, monkeypatch)
    assert cells["absent"] == 0
    separate, _ = _losses(NO_ABSENT, monkeypatch, absent_reduction="separate", absent_loss_weight=5.0)
    assert separate["start_loss"] == pytest.approx(pooled["start_loss"], rel=1e-6)


def test_reduction_keys_are_training_time_not_structural():
    for key in ("absent_reduction", "absent_loss_weight", "present_loss_scale"):
        assert key not in T._STRUCTURAL_BOUNDARY_KEYS
