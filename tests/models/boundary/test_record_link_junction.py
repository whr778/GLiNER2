"""record_link_mode: junction -- the trigger x argument link and its column loss (JUNCTION_LAYER_SPEC.md).

Pinned: a junction added to a built model is exactly zero at init (scores and loss unchanged);
the column loss is invariant to anything constant down a column (the role term R cancels, so only
the link can lower it); a junction checkpoint saves and strictly reloads; bad settings are refused.
"""
import dataclasses
import random
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import gliner2.models.boundary.records as R
from gliner2.configuration import validate_boundary_head
from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "train"))
import train as T  # noqa: E402

TEXT = "hackers attacked the bank and stole money from customers ; police attacked the hackers later ."
SCHEMA = {"events": [{"event_type": "attack", "triggers": ["attacked"],
                      "arguments": [{"role": "attacker", "entity": "hackers"}, {"role": "target", "entity": "bank"}]}]}


def _forward(model, monkeypatch=None, capture=None):
    random.seed(0)
    batch = ExtractorCollator(model.processor, is_training=True, architecture="boundary",
                              event_records=True)([(TEXT, SCHEMA)])
    if capture is not None:
        real = R.compute_group_loss
        monkeypatch.setattr(R, "compute_group_loss",
                            lambda g, rs, *a, **k: (capture.append((g, rs)), real(g, rs, *a, **k))[1])
    model.train()
    torch.manual_seed(0)
    return model(batch)


def _model(**bh):
    torch.manual_seed(0)
    model = build_tiny_boundary_model()
    T._apply_boundary_head_overrides(model, bh)
    return model


def test_junction_is_exactly_zero_when_added(monkeypatch):
    seen_a, seen_j = [], []
    out_a = _forward(_model(record_link_mode="additive"), monkeypatch, seen_a)
    out_j = _forward(_model(record_link_mode="junction"), monkeypatch, seen_j)
    assert seen_j and all(g.spec.mode == "natural" for g, _ in seen_j)
    for (ga, _), (gj, _) in zip(seen_a, seen_j):
        for a, j in zip(ga.assign_logits, gj.assign_logits):
            assert torch.equal(a, j)
    assert float(out_a.loss) == pytest.approx(float(out_j.loss), abs=0)


def test_column_loss_ignores_anything_constant_down_a_column(monkeypatch):
    seen = []
    _forward(_model(record_link_mode="junction", record_role_hard_negatives=8), monkeypatch, seen)
    g, rs = next((g, rs) for g, rs in seen if g.spec.mode == "natural" and rs)
    aq = g.spec.anchor_query_id
    spans = [R._span_index(s) for s in g.field_spans]
    base, terms, _, _ = R._column_loss(g, rs, aq, spans, 4)
    assert terms > 0, "the fixture must produce at least one gold-argument column"
    shifted = dataclasses.replace(g, assign_logits=[a + torch.randn(1, a.shape[1]) * 5 for a in g.assign_logits])
    moved, _, _, _ = R._column_loss(shifted, rs, aq, spans, 4)
    assert float(moved) == pytest.approx(float(base), rel=1e-5), "a per-column constant (the role term) moved the loss"


def test_column_loss_is_reported_and_weighted(monkeypatch):
    seen = []
    out = _forward(_model(record_link_mode="junction", record_link_column_weight=1.0), monkeypatch, seen)
    off = _forward(_model(record_link_mode="junction", record_link_column_weight=0.0))
    assert float(out.losses["record_field_loss"]) > float(off.losses["record_field_loss"])


def test_junction_checkpoint_saves_and_strictly_reloads():
    model = _model(record_link_mode="junction")
    with torch.no_grad():
        model.record_decoder.link.v.weight.normal_()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        reloaded = type(model).from_pretrained(d, map_location="cpu")
    assert reloaded.record_decoder.link is not None
    assert torch.equal(reloaded.record_decoder.link.v.weight, model.record_decoder.link.v.weight)


def test_settings_are_validated():
    assert validate_boundary_head({})["record_link_mode"] == "additive"
    for bad in ({"record_link_mode": "dense"}, {"record_link_column_weight": 1.0},
                {"record_link_mode": "junction", "candidate_pool": "shared"},
                {"record_link_mode": "junction", "record_link_column_negatives": 0}):
        with pytest.raises(ValueError):
            validate_boundary_head(bad)
