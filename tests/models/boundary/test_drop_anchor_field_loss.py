"""record_drop_anchor_field_loss (code review Issue 2): the anchor's own row leaves the field loss, the divisor does not.

Group: one event type, fields [trigger (anchor, scalar), Location (list)]; instances 0..4 are the trigger candidates.
Record A's trigger is candidate 0 with Location (5,7); B's is candidate 2 with Locations (5,7) and (25,27).
"""
import random
import sys
from pathlib import Path

import pytest
import torch

import gliner2.models.boundary.records as R
from gliner2.configuration import validate_boundary_head
from gliner2.processing.records import FieldCardinality, RecordFieldSpec, RecordSpec
from gliner2.processing.targets import RecordFieldTarget, RecordTarget
from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "train"))
import train as T  # noqa: E402

TRIG = RecordFieldSpec(query_id=0, name="trigger", role_index=0, cardinality=FieldCardinality.REQUIRED_ONE, is_anchor=True)
LOC = RecordFieldSpec(query_id=1, name="Location", role_index=1, cardinality=FieldCardinality.ZERO_OR_MORE)


def group():
    torch.manual_seed(0)
    spec = RecordSpec(task_index=0, task_name="Deploy", task_type="events", mode="natural", fields=(TRIG, LOC), anchor_query_id=0)
    anchor_spans = torch.tensor([[2, 3], [10, 11], [20, 21], [30, 31], [40, 41]])
    loc_spans = torch.tensor([[5, 7], [25, 27], [50, 52]])
    ni = anchor_spans.shape[0]
    return R.RecordGroupOutput(
        spec=spec, object_logits=torch.zeros(ni),
        assign_logits=[torch.randn(ni, 1 + ni), torch.randn(ni, 1 + loc_spans.shape[0])],
        field_query_ids=[0, 1], field_specs=[TRIG, LOC], field_spans=[anchor_spans, loc_spans],
        field_cand_mask=[torch.ones(ni, dtype=torch.bool), torch.ones(3, dtype=torch.bool)],
        field_cand_logits=[torch.zeros(ni), torch.zeros(3)],
        instance_seed=[(0, c) for c in range(ni)], instance_spans=[tuple(s) for s in anchor_spans.tolist()],
    )


def record(rid, trigger, locs):
    return RecordTarget(rid, 0, fields=(RecordFieldTarget(0, values=((trigger,),)),
                                        RecordFieldTarget(1, values=tuple((s,) for s in locs))), anchor_query_id=0)


def test_drop_keeps_the_role_terms_at_their_exact_weight():
    g, a = group(), record("A", (2, 3), [(5, 7)])
    spans = [R._span_index(s) for s in g.field_spans]
    anchor = R._scalar_field_nll(g.assign_logits[0][0], [1])
    role = R._list_field_bce(g.assign_logits[1][0], [1], 0)
    assert torch.allclose(R._instance_field_loss(g, 0, a, spans), (anchor + role) / 2)
    assert torch.allclose(R._instance_field_loss(g, 0, a, spans, drop_anchor=True), role / 2)


def test_group_loss_drops_one_anchor_row_per_trained_owner_and_counts_it():
    g = group()
    recs = [record("A", (2, 3), [(5, 7)]), record("B", (20, 21), [(5, 7), (25, 27)])]
    spans = [R._span_index(s) for s in g.field_spans]
    before = R._ANCHOR_FIELD_DROPS
    off = R.compute_group_loss(g, recs)["field_loss"]
    assert R._ANCHOR_FIELD_DROPS == before
    on = R.compute_group_loss(g, recs, drop_anchor_field=True)["field_loss"]
    assert R._ANCHOR_FIELD_DROPS == before + 2
    roles = sum(R._list_field_bce(g.assign_logits[1][i], cols, 0) / 2 for i, cols in ((0, [1]), (2, [1, 2]))) / 2
    assert torch.allclose(on, roles)
    assert float(off) > float(on)


def test_setting_parses_refuses_the_shared_pool_and_reaches_the_group_loss():
    assert validate_boundary_head({})["record_drop_anchor_field_loss"] is False
    assert validate_boundary_head({"record_drop_anchor_field_loss": True})["record_drop_anchor_field_loss"] is True
    with pytest.raises(ValueError, match="per_query only"):
        validate_boundary_head({"record_drop_anchor_field_loss": True, "candidate_pool": "shared"})
    text = "hackers attacked the bank and stole money ; police attacked the hackers later ."
    schema = {"events": [{"event_type": "attack", "triggers": ["attacked"],
                          "arguments": [{"role": "attacker", "entity": "hackers"}, {"role": "target", "entity": "bank"}]}]}
    seen = {}
    for flag in (False, True):
        torch.manual_seed(0)
        random.seed(0)
        model = build_tiny_boundary_model()
        T._apply_boundary_head_overrides(model, {"record_drop_anchor_field_loss": flag})
        batch = ExtractorCollator(model.processor, is_training=True, architecture="boundary", event_records=True)([(text, schema)])
        calls = []
        real = R.compute_group_loss
        R.compute_group_loss = lambda g, rs, *a, **k: (calls.append(k), real(g, rs, *a, **k))[1]
        try:
            model.train()
            model(batch)
        finally:
            R.compute_group_loss = real
        seen[flag] = [k.get("drop_anchor_field") for k in calls]
    assert seen[True] and all(v is True for v in seen[True])
    assert seen[False] and all(v is None for v in seen[False])
