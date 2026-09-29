"""A gold record whose ANCHOR the model did not propose trains NOTHING — and is counted.

THE MECHANISM. `compute_group_loss` in natural mode resolves each gold record's anchor
against `span_to_idx` — the MODEL'S OWN candidate spans (`_resolve_value_cols`). On a miss
it does `continue`, so that record contributes no object loss, no field loss, and no
gradient of any kind. It did this silently until `_note_anchor_gate` existed.

WHY IT MATTERS FOR EVENTS SPECIFICALLY. Events compile as mode="natural" with the TRIGGER
as the anchor — `processing/records.py:70` synthesizes `{"mode": "natural", "anchor":
queries[0].role_name}` and its docstring says "role_index 0 is always the trigger and is
always the anchor". So a trigger the model failed to propose discards that instance's
ENTIRE argument supervision. That is self-reinforcing: propose few triggers -> train on few
events -> keep proposing few triggers.

IT IS THE FOURTH CANDIDATE FOR THE RECALL FLOOR. EVENT_ARGUMENT_DIAGNOSIS 4f cleared three
— no capacity caps, per-role cardinality at 7.5%, the mention-path `skip_sample` — and
concluded "undertraining, nothing structural capping it". This is not a capacity cap, it is
a supervision gate, and 4f's search did not cover it.

These tests do NOT claim the gate is the cause of anything. They pin the mechanism and make
it countable, so the share can be measured instead of argued about.
"""

from __future__ import annotations

import torch

from gliner2.models.boundary import records as R
from gliner2.models.boundary.records import RecordHead
from gliner2.models.boundary.record_loss import compute_group_loss
from gliner2.processing.records import FieldCardinality, RecordFieldSpec, RecordSpec
from gliner2.processing.targets import RecordFieldTarget, RecordTarget

from tests.models.boundary.test_record_head_pipeline import make_candidates

HIDDEN = 24


def _spec():
    return RecordSpec(
        task_index=0, task_name="purchase", task_type="json_structures", mode="natural",
        fields=(
            RecordFieldSpec(0, "buyer", 0, FieldCardinality.REQUIRED_ONE, is_anchor=True),
            RecordFieldSpec(1, "item", 1, FieldCardinality.OPTIONAL_ONE),
        ),
        anchor_query_id=0,
    )


def _record(anchor_span, item_span):
    return RecordTarget("0:0", 0, (
        RecordFieldTarget(0, ((anchor_span,),)),
        RecordFieldTarget(1, ((item_span,),)),
    ), anchor_query_id=0)


def _run(anchor_span):
    """Score one gold record whose anchor is `anchor_span`, against fixed candidates."""
    torch.manual_seed(0)
    cands = make_candidates([[(0, 1), (3, 4)], [(1, 2), (4, 5)]], HIDDEN, high_logit_field=0)
    head = RecordHead(HIDDEN, record_dim=24, instance_queries=8)
    group = head.forward_group(_spec(), torch.randn(2, HIDDEN), cands, 0)
    R.reset_anchor_gate()
    losses = compute_group_loss(group, [_record(anchor_span, (1, 2))])
    return losses, R.anchor_gate_stats()


def test_a_proposed_anchor_trains_and_is_counted():
    """Control. (0, 1) IS in the candidate set, so the record supervises normally."""
    losses, gate = _run((0, 1))
    assert gate["seen"] == 1
    assert gate["trained_n"] == 1
    assert gate["anchor_not_proposed_n"] == 0
    assert losses["field_count"] > 0


def test_an_unproposed_anchor_trains_nothing_and_is_counted():
    """(9, 10) is NOT a candidate span. The record is skipped entirely."""
    losses, gate = _run((9, 10))
    assert gate["seen"] == 1
    assert gate["trained_n"] == 0
    assert gate["anchor_not_proposed_n"] == 1, (
        "the whole point: a gold record whose anchor was never proposed is dropped"
    )
    assert losses["field_count"] == 0, "and it contributes NO field supervision"


def test_the_dropped_record_produces_no_gradient():
    """'No loss' has to mean no gradient, not merely a zero in the report."""
    losses, _ = _run((9, 10))
    total = losses["object_loss"] + losses["field_loss"]
    assert float(total) == 0.0
    assert not total.requires_grad or total.grad_fn is None or float(total) == 0.0


def test_the_three_outcomes_are_distinguished():
    """A single 'skipped' tally would hide which failure is happening; they need
    different fixes (propose more spans vs. seed more instances)."""
    R.reset_anchor_gate()
    for outcome in ("trained", "anchor_not_proposed", "anchor_not_seeded", "no_gold_anchor"):
        R._note_anchor_gate(outcome)
    gate = R.anchor_gate_stats()
    assert gate["seen"] == 4
    assert all(gate[f"{k}_n"] == 1 for k in
               ("trained", "anchor_not_proposed", "anchor_not_seeded", "no_gold_anchor"))
    assert abs(sum(gate[f"{k}_share"] for k in
                   ("trained", "anchor_not_proposed", "anchor_not_seeded",
                    "no_gold_anchor")) - 1.0) < 1e-9
