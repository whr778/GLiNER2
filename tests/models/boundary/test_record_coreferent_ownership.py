"""record_coreferent_ownership: every seeded mention of an event owns its arguments, averaged per record.

COREFERENT_OWNERSHIP_SPEC.md (TODO #29). Traced on cc_news_events_sonnet55_v2: 59 of 101 seeded mentions
had no role targets. Pinned: off is the historical single owner; on trains every mention; the record's
loss is the MEAN over its mentions (a 3-mention event must not weigh 3x); a one-mention record is
unchanged by the flag; bad settings are refused.
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

TEXT = "Rebels attacked the base on Monday . The assault killed four soldiers ."
TWO = {"events": [{"event_type": "attack", "triggers": ["attacked", "assault"],
                   "arguments": [{"role": "attacker", "entity": "Rebels"}, {"role": "target", "entity": "base"}]}]}
FIELD_LOSS = R._instance_field_loss   # the ORIGINAL: wrapping a wrapper chains the spies across calls

ONE = {"events": [{"event_type": "attack", "triggers": ["attacked"],
                   "arguments": [{"role": "attacker", "entity": "Rebels"}]}]}


def _owners_and_loss(monkeypatch, gold, flag):
    torch.manual_seed(0)
    model = build_tiny_boundary_model()
    T._apply_boundary_head_overrides(model, {"record_coreferent_ownership": flag})
    model.processor.sampling_config.remove_events_prob = 0.0
    seen = []
    monkeypatch.setattr(R, "_instance_field_loss",
                        lambda g, i, *a, **k: (seen.append(i), FIELD_LOSS(g, i, *a, **k))[1])
    random.seed(0)
    batch = ExtractorCollator(model.processor, is_training=True, architecture="boundary", event_records=True)([(TEXT, gold)])
    model.train()
    torch.manual_seed(0)
    out = model(batch)
    return seen, float(out.losses["record_field_loss"])


def test_off_trains_one_mention_on_trains_both(monkeypatch):
    off, _ = _owners_and_loss(monkeypatch, TWO, False)
    on, _ = _owners_and_loss(monkeypatch, TWO, True)
    assert len(set(off)) == 1 and len(set(on)) == 2


def test_record_loss_is_the_mean_over_its_mentions(monkeypatch):
    calls = []
    real = R._instance_field_loss
    def spy(g, i, *a, **k):
        v = real(g, i, *a, **k)
        calls.append(float(v))
        return v
    monkeypatch.setattr(R, "_instance_field_loss", spy)
    torch.manual_seed(0)
    model = build_tiny_boundary_model()
    T._apply_boundary_head_overrides(model, {"record_coreferent_ownership": True})
    model.processor.sampling_config.remove_events_prob = 0.0
    random.seed(0)
    batch = ExtractorCollator(model.processor, is_training=True, architecture="boundary", event_records=True)([(TEXT, TWO)])
    captured = []
    real_group = R.compute_group_loss
    monkeypatch.setattr(R, "compute_group_loss", lambda *a, **k: (captured.append(real_group(*a, **k)), captured[-1])[1])
    model.train()
    torch.manual_seed(0)
    model(batch)
    ev = [c for c in captured if c["field_count"]][0]
    assert len(calls) == 2
    assert float(ev["field_loss"]) == pytest.approx(sum(calls) / 2, rel=1e-6)


def test_one_mention_record_is_unchanged_by_the_flag(monkeypatch):
    _, off = _owners_and_loss(monkeypatch, ONE, False)
    _, on = _owners_and_loss(monkeypatch, ONE, True)
    assert on == pytest.approx(off, abs=0)


def test_shared_pool_is_refused():
    with pytest.raises(ValueError):
        validate_boundary_head({"record_coreferent_ownership": True, "candidate_pool": "shared"})
    assert validate_boundary_head({})["record_coreferent_ownership"] is False


def test_merge_keeps_instances_with_no_shared_high_candidate_apart():
    """Gate 4 on the mechanics: instances that score no common candidate above the gate stay separate."""
    from types import SimpleNamespace
    big = 10.0
    rows = torch.full((3, 4), -big)          # 3 instances x (null + 3 candidates)
    rows[0, 1] = big                          # inst 0 -> cand 0
    rows[1, 1] = big                          # inst 1 -> cand 0  (shares with 0)
    rows[2, 3] = big                          # inst 2 -> cand 2  (shares with nobody)
    group = SimpleNamespace(field_specs=[SimpleNamespace(cardinality=SimpleNamespace(is_scalar=False))],
                            assign_logits=[rows])
    reps, members = R._merge_coreferent_instances(group, [0, 1, 2], field_threshold=0.5, temperature=1.0)
    assert reps == [0, 2] and members == {0: [1]}
