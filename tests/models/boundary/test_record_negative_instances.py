"""record_negative_instances: false trigger instances trained on their ROLE fields, own mean.

Pinned: off by default (no negative keys, gold loss untouched); on, the picked instances are
false (none overlaps a gold trigger), the negative term is returned separately and never enters
the gold denominator, and the shared pool refuses the setting.
"""
import random
import sys
from pathlib import Path

import pytest
import torch

import gliner2.models.boundary.records as R
from gliner2.configuration import validate_boundary_head
from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "train"))
import train as T  # noqa: E402

TEXT = "hackers attacked the bank and stole money from customers yesterday ."
SCHEMA = {"events": [{"event_type": "attack", "triggers": ["attacked"],
                      "arguments": [{"role": "attacker", "entity": "hackers"}, {"role": "target", "entity": "bank"}]}]}


def _groups(k, monkeypatch):
    """[(group, records, losses)] for every natural group of one seeded training forward."""
    seen = []
    real = R.compute_group_loss

    def wrap(group, records, *a, **kw):
        out = real(group, records, *a, **kw)
        seen.append((group, records, out))
        return out

    monkeypatch.setattr(R, "compute_group_loss", wrap)
    random.seed(0)
    torch.manual_seed(0)
    model = build_tiny_boundary_model()
    T._apply_boundary_head_overrides(model, {"record_negative_instances": k, "record_negative_weight": 1.0})
    model.train()
    random.seed(0)
    batch = ExtractorCollator(model.processor, is_training=True, architecture="boundary",
                              event_records=True)([(TEXT, SCHEMA)])
    torch.manual_seed(0)
    model(batch)
    return [(g, r, o) for g, r, o in seen if g.spec.mode == "natural"]


def test_off_by_default_returns_no_negative_term(monkeypatch):
    groups = _groups(0, monkeypatch)
    assert groups, "the fixture must produce a natural event group"
    assert all("negative_count" not in o for _, _, o in groups)


def test_negatives_are_false_instances_and_kept_out_of_the_gold_mean(monkeypatch):
    off = _groups(0, monkeypatch)
    on = _groups(4, monkeypatch)
    for (g, recs, o_off), (_, _, o_on) in zip(off, on):
        assert torch.allclose(o_on["field_loss"], o_off["field_loss"]), "negatives changed the gold field loss"
        assert o_on["field_count"] == o_off["field_count"], "negatives entered the gold denominator"
    picked = [o for _, _, o in on if o.get("negative_count")]
    assert picked, "no negative instance was trained"
    for g, recs, o in on:
        aq = g.spec.anchor_query_id
        gold = [(int(s[0]), int(s[1])) for r in recs if (a := r.field_for_query(aq)) and a.values for s in a.values[0]]
        for i in R._negative_instances(g, recs, aq, set(), 4):
            s, e = g.instance_spans[i]
            assert not any(s < ge and gs < e for gs, ge in gold), "a negative overlaps a gold trigger"


def test_shared_pool_refuses_negative_instances():
    with pytest.raises(ValueError):
        validate_boundary_head({"record_negative_instances": 4, "candidate_pool": "shared"})
