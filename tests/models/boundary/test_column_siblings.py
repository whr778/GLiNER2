"""record_link_column_siblings (outside review Issue 4): same-type sibling gold triggers become column negatives
only for columns their own gold does not hold; off, the column loss is the historical one.

Group: one event type, fields [trigger (anchor), Location (list)]. Instances 0,1 = A's two coreferent trigger
mentions, 2 = B's trigger, 3,4 = false triggers. Location columns: 1 = (5,7) gold for A AND B, 2 = (25,27) B only.
"""
import random
import sys
from pathlib import Path

import torch

import gliner2.models.boundary.records as R
from gliner2.configuration import validate_boundary_head
from gliner2.processing.records import FieldCardinality, RecordFieldSpec, RecordSpec
from gliner2.processing.targets import RecordFieldTarget, RecordTarget
from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "train"))
import train as T  # noqa: E402


def two_event_group():
    torch.manual_seed(0)
    trig = RecordFieldSpec(query_id=0, name="trigger", role_index=0, cardinality=FieldCardinality.REQUIRED_ONE, is_anchor=True)
    loc = RecordFieldSpec(query_id=1, name="Location", role_index=1, cardinality=FieldCardinality.ZERO_OR_MORE)
    spec = RecordSpec(task_index=0, task_name="Deploy", task_type="events", mode="natural", fields=(trig, loc), anchor_query_id=0)
    a = RecordTarget("0:0", 0, fields=(RecordFieldTarget(0, values=(((2, 3), (10, 11)),)),
                                       RecordFieldTarget(1, values=(((5, 7),),))), anchor_query_id=0)
    b = RecordTarget("0:1", 0, fields=(RecordFieldTarget(0, values=(((20, 21),),)),
                                       RecordFieldTarget(1, values=(((5, 7),), ((25, 27),)))), anchor_query_id=0)
    anchor_spans = torch.tensor([[2, 3], [10, 11], [20, 21], [30, 31], [40, 41]])
    loc_spans = torch.tensor([[5, 7], [25, 27], [50, 52]])
    ni = anchor_spans.shape[0]
    group = R.RecordGroupOutput(
        spec=spec, object_logits=torch.tensor([0.1, 0.2, 0.3, 2.0, 1.5]),
        assign_logits=[torch.randn(ni, 1 + ni), torch.randn(ni, 1 + loc_spans.shape[0])],
        field_query_ids=[0, 1], field_specs=[trig, loc], field_spans=[anchor_spans, loc_spans],
        field_cand_mask=[torch.ones(ni, dtype=torch.bool), torch.ones(3, dtype=torch.bool)],
        field_cand_logits=[torch.zeros(ni), torch.zeros(3)],
        instance_seed=[(0, c) for c in range(ni)], instance_spans=[tuple(s) for s in anchor_spans.tolist()],
    )
    return group, [a, b], [R._span_index(s) for s in group.field_spans]


def term(column, pos, neg):
    p, n = column[pos], column[neg]
    return -(torch.logsumexp(p, 0) - torch.logsumexp(torch.cat([p, n]), 0))


def test_siblings_are_negatives_only_where_they_do_not_own_the_column():
    group, records, spans = two_event_group()
    col = group.assign_logits[1]
    false = [3, 4]
    # A's column 1 is B's gold too: no sibling. B's column 1: A owns it: no sibling. B's column 2: A's triggers 0,1.
    expected = term(col[:, 1], [0, 1], false) + term(col[:, 1], [2], false) + term(col[:, 2], [2], false + [0, 1])
    total, terms, wins, pairs, sib_wins, sib_pairs = R._column_loss(group, records, 0, spans, 8, siblings=True)
    assert terms == 3 and pairs == 6 and sib_pairs == 2
    assert torch.allclose(total, expected)
    assert sib_wins == int((col[[0, 1], 2] < col[2, 2]).sum())


def test_off_is_the_historical_loss():
    group, records, spans = two_event_group()
    col = group.assign_logits[1]
    false = [3, 4]
    expected = term(col[:, 1], [0, 1], false) + term(col[:, 1], [2], false) + term(col[:, 2], [2], false)
    total, terms, wins, pairs, sib_wins, sib_pairs = R._column_loss(group, records, 0, spans, 8)
    assert (terms, pairs, sib_wins, sib_pairs) == (3, 6, 0, 0)
    assert torch.allclose(total, expected)


def test_siblings_train_a_column_even_without_false_negatives():
    group, records, spans = two_event_group()
    col = group.assign_logits[1]
    off = R._column_loss(group, records, 0, spans, 0)
    on = R._column_loss(group, records, 0, spans, 0, siblings=True)
    assert off[1:] == (0, 0, 0, 0, 0) and float(off[0]) == 0.0
    assert on[1:] == (1, 0, 0, int((col[[0, 1], 2] < col[2, 2]).sum()), 2)
    assert torch.allclose(on[0], term(col[:, 2], [2], [0, 1]))


def test_setting_parses_and_reaches_the_group_loss():
    assert validate_boundary_head({"record_link_column_siblings": True})["record_link_column_siblings"] is True
    assert validate_boundary_head({})["record_link_column_siblings"] is False
    text = "hackers attacked the bank and stole money ; police attacked the hackers later ."
    schema = {"events": [{"event_type": "attack", "triggers": ["attacked"],
                          "arguments": [{"role": "attacker", "entity": "hackers"}, {"role": "target", "entity": "bank"}]}]}
    seen = {}
    for flag in (False, True):
        torch.manual_seed(0)
        random.seed(0)
        model = build_tiny_boundary_model()
        T._apply_boundary_head_overrides(model, {"record_link_mode": "junction", "record_link_column_weight": 0.3,
                                                 "record_link_column_siblings": flag})
        batch = ExtractorCollator(model.processor, is_training=True, architecture="boundary", event_records=True)([(text, schema)])
        calls = []
        real = R.compute_group_loss
        R.compute_group_loss = lambda g, rs, *a, **k: (calls.append(k), real(g, rs, *a, **k))[1]
        try:
            model.train()
            model(batch)
        finally:
            R.compute_group_loss = real
        seen[flag] = [k.get("column_siblings") for k in calls if k.get("column_negatives")]
    assert seen[True] and all(v is True for v in seen[True])
    assert seen[False] and all(v is None for v in seen[False])
