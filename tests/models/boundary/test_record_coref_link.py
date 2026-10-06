"""record_coref_link: the trigger x trigger coreference link (COREFERENT_LINK_SPEC.md).

Pinned: built only when enabled (strict checkpoints still load) and saves/reloads strictly when on;
the pair loss labels same-record mentions 1, different records 0 and false triggers 0, as its OWN mean;
off and on-with-weight-0 change no loss; the link decode merges at the threshold; bad settings refused.
"""
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gliner2.models.boundary.records as R
from gliner2.configuration import validate_boundary_head
from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "train"))
import train as T  # noqa: E402

TEXT = "Rebels attacked the base on Monday . The assault killed four . Police raided a camp ."
GOLD = {"events": [
    {"event_type": "attack", "triggers": ["attacked", "assault"], "arguments": [{"role": "attacker", "entity": "Rebels"}]},
    {"event_type": "attack", "triggers": ["raided"], "arguments": [{"role": "attacker", "entity": "Police"}]}]}


def _group(monkeypatch, **bh):
    torch.manual_seed(0)
    model = build_tiny_boundary_model()
    T._apply_boundary_head_overrides(model, bh)
    model.processor.sampling_config.remove_events_prob = 0.0
    seen, real = [], R.compute_group_loss
    monkeypatch.setattr(R, "compute_group_loss", lambda g, rs, *a, **k: (seen.append((g, rs)), real(g, rs, *a, **k))[1])
    random.seed(0)
    batch = ExtractorCollator(model.processor, is_training=True, architecture="boundary", event_records=True)([(TEXT, GOLD)])
    model.train()
    torch.manual_seed(0)
    out = model(batch)
    return model, out, next((g, rs) for g, rs in seen if g.spec.mode == "natural" and len(rs) == 2)


def test_built_only_when_enabled_and_reloads_strictly():
    assert build_tiny_boundary_model().record_decoder.coref is None
    torch.manual_seed(0)
    model = build_tiny_boundary_model()
    T._apply_boundary_head_overrides(model, {"record_coref_link": True})
    with torch.no_grad():
        model.record_decoder.coref.dist.weight.fill_(0.7)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        back = type(model).from_pretrained(d, map_location="cpu")
    assert torch.equal(back.record_decoder.coref.dist.weight, model.record_decoder.coref.dist.weight)


def test_pairs_are_labelled_from_gold_clusters_and_false_triggers(monkeypatch):
    _, _, (g, rs) = _group(monkeypatch, record_coref_link=True)
    aq = g.spec.anchor_query_id
    spans = [R._span_index(s) for s in g.field_spans]
    af = g.field_query_ids.index(aq)
    seed = {s[1]: i for i, s in enumerate(g.instance_seed) if s is not None and s[0] == af}
    gold = {seed[c - 1] for rec in rs for c in R._resolve_value_cols(rec.field_for_query(aq).values[0], spans[af])
            if (c - 1) in seed}
    false = R._negative_instances(g, rs, aq, gold, 4)
    targets = []
    real = R.F.binary_cross_entropy_with_logits
    monkeypatch.setattr(R.F, "binary_cross_entropy_with_logits",
                        lambda x, t, **k: (targets.append(t.tolist()), real(x, t, **k))[1])
    _, n, _, _ = R._coref_link_loss(g, rs, aq, spans, 4)
    assert len(gold) == 3 and false, "fixture: 3 seeded gold mentions and some false triggers"
    t = targets[-1]
    assert t.count(1.0) == 1                                # attacked ~ assault
    assert t.count(0.0) == 2 + 3 * len(false)               # 2 hard negatives + every mention vs each false trigger
    assert n == len(t)


def test_off_and_weight_zero_change_no_loss(monkeypatch):
    _, off, _ = _group(monkeypatch)
    _, on0, _ = _group(monkeypatch, record_coref_link=True)
    assert float(off.loss) == pytest.approx(float(on0.loss), abs=0)
    _, on, _ = _group(monkeypatch, record_coref_link=True, record_coref_link_weight=1.0)
    assert float(on.losses["record_field_loss"]) != pytest.approx(float(off.losses["record_field_loss"]), abs=1e-9)


def test_link_decode_merges_at_the_threshold():
    logits = torch.tensor([[0.0, 4.0, -4.0], [4.0, 0.0, -4.0], [-4.0, -4.0, 0.0]])
    group = SimpleNamespace(coref_logits=logits)
    reps, members = R._merge_by_link(group, [0, 1, 2], threshold=0.5)
    assert reps == [0, 2] and members == {0: [1]}
    assert R._merge_by_link(group, [0, 1, 2], threshold=0.99)[1] == {}


def test_settings_are_validated():
    assert validate_boundary_head({})["record_merge_coreferent"] == "off"
    assert validate_boundary_head({"record_merge_coreferent": True})["record_merge_coreferent"] == "args"
    for bad in ({"record_coref_link_weight": 1.0}, {"record_merge_coreferent": "link"},
                {"record_merge_coreferent": "both"}, {"record_coref_link": True, "candidate_pool": "shared"}):
        with pytest.raises(ValueError):
            validate_boundary_head(bad)


def test_group_coref_loss_is_the_mean_over_its_pairs(monkeypatch):
    _, _, (g, rs) = _group(monkeypatch, record_coref_link=True)
    aq = g.spec.anchor_query_id
    spans = [R._span_index(s) for s in g.field_spans]
    summed, n, _, _ = R._coref_link_loss(g, rs, aq, spans, 4)
    out = R.compute_group_loss(g, rs, coref_negatives=4)
    assert out["coref_count"] == n and n > 1
    assert float(out["coref_loss"]) == pytest.approx(float(summed) / n, rel=1e-6)


def test_link_merge_without_the_module_fails_loudly():
    group = SimpleNamespace(object_logits=torch.tensor([3.0, 3.0]), num_instances=2, spec=SimpleNamespace(mode="natural"),
                            coref_logits=None, field_specs=[], assign_logits=[], instance_seed=[None, None],
                            instance_spans=[(0, 1), (2, 3)], field_spans=[], field_query_ids=[])
    with pytest.raises(ValueError, match="no coreference link"):
        R.decode_group(group, anchor_threshold=0.5, merge_coreferent="link")
