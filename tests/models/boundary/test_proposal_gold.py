"""proposal_gold: which candidates proposal_loss treats as gold.

`injected` flags only this step's injected copies, so below injection 1.0 a gold span the
proposer found itself is a NEGATIVE in proposal_loss while pair/rerank label it positive.
`identity` uses the pair labels. Pinned: the modes agree at injection 1.0; at 0.0 `injected`
supervises nothing and `identity` keeps every proposed gold positive.
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

TEXT = "apple released iphone in cupertino ."
SCHEMA = {"entities": {"company": ["apple"], "product": ["iphone"], "city": ["cupertino"]}}


def _run(mode, injection, monkeypatch):
    """(proposal_loss, gold mask proposal_loss received, pair-label gold) for one seeded forward."""
    seen = {}
    real_prop, real_pair = M.proposal_listwise_loss, M.candidate_pair_loss

    def prop(logits, gold, valid, *a, **k):
        seen["gold"] = gold & valid
        return real_prop(logits, gold, valid, *a, **k)

    def pair(logits, labels, valid, *a, **k):
        seen.setdefault("identity", (labels > 0.5) & valid)
        return real_pair(logits, labels, valid, *a, **k)

    monkeypatch.setattr(M, "proposal_listwise_loss", prop)
    monkeypatch.setattr(M, "candidate_pair_loss", pair)
    random.seed(0)
    torch.manual_seed(0)
    model = build_tiny_boundary_model()
    T._apply_boundary_head_overrides(model, {"proposal_gold": mode})
    model.train()
    random.seed(0)
    batch = ExtractorCollator(model.processor, is_training=True, architecture="boundary")([(TEXT, SCHEMA)])
    torch.manual_seed(0)
    out = model(batch, gold_injection_prob=injection)
    return float(out.losses["proposal_loss"].detach()), seen["gold"], seen["identity"]


def test_default_is_injected_and_bad_values_are_refused():
    assert BoundaryHeadSettings(**validate_boundary_head({})).proposal_gold == "injected"
    with pytest.raises(ValueError):
        validate_boundary_head({"proposal_gold": "natural"})
    assert "proposal_gold" not in T._STRUCTURAL_BOUNDARY_KEYS


def test_modes_agree_when_every_gold_is_injected(monkeypatch):
    injected, g_inj, ident = _run("injected", 1.0, monkeypatch)
    identity, g_id, _ = _run("identity", 1.0, monkeypatch)
    assert int(ident.sum()) > 0
    assert torch.equal(g_inj, g_id)
    assert injected == pytest.approx(identity, rel=1e-6)


def test_without_injection_injected_trains_gold_as_negatives(monkeypatch):
    loss, gold, identity = _run("injected", 0.0, monkeypatch)
    assert int(identity.sum()) > 0, "the fixture must propose some gold naturally"
    assert int(gold.sum()) == 0 and loss == 0.0


def test_identity_keeps_proposed_gold_positive_without_injection(monkeypatch):
    loss, gold, identity = _run("identity", 0.0, monkeypatch)
    assert torch.equal(gold, identity)
    assert loss > 0.0
