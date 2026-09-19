"""Absent-label negatives enter the listwise denominator, PER TASK, and only when real.

Until 2026-09-19 `proposal_listwise_loss` skipped any query with no gold, so an injected
label negative -- a label mapped to an empty list -- contributed EXACTLY ZERO to both
listwise losses. These tests fail without the fix.

Pooling is per task on purpose: a global pool would make an absent ENTITY label a negative
for an EVENT role's gold, mixing task semantics and letting one dimension dominate.
"""

import pytest
import torch

from gliner2.models.boundary.losses import proposal_listwise_loss

ENTITIES, RELATIONS, EVENTS, STRUCTURES = 0, 1, 2, 3


def _one_gold_one_absent(task_a=ENTITIES, task_b=ENTITIES):
    """Query 0 has gold; query 1 is an ABSENT label. Tasks are caller-chosen."""
    logits = torch.tensor([[[2.0, 0.5, -1.0], [3.0, 1.0, 0.0]]])
    gold = torch.tensor([[[True, False, False], [False, False, False]]])
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    qmask = torch.ones(1, 2, dtype=torch.bool)
    tids = torch.tensor([[task_a, task_b]])
    return logits, gold, valid, qmask, tids


def test_absent_label_of_the_SAME_task_enters_the_denominator():
    logits, gold, valid, qmask, tids = _one_gold_one_absent(ENTITIES, ENTITIES)
    off = proposal_listwise_loss(logits, gold, valid, qmask)
    on = proposal_listwise_loss(logits, gold, valid, qmask,
                                absent_negatives=True, task_ids=tids)
    assert on > off, f"same-task absent label did not reach the denominator: {on} !> {off}"


def test_absent_label_of_a_DIFFERENT_task_does_NOT_leak_in():
    """The whole point of per-task pooling."""
    logits, gold, valid, qmask, tids = _one_gold_one_absent(ENTITIES, EVENTS)
    off = proposal_listwise_loss(logits, gold, valid, qmask)
    on = proposal_listwise_loss(logits, gold, valid, qmask,
                                absent_negatives=True, task_ids=tids)
    assert torch.allclose(off, on), "an absent EVENT label leaked into an ENTITY query"


def test_it_refuses_to_pool_globally_without_task_ids():
    logits, gold, valid, qmask, _ = _one_gold_one_absent()
    with pytest.raises(ValueError, match="requires task_ids"):
        proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True)


def test_gate_reports_what_entered():
    logits, gold, valid, qmask, tids = _one_gold_one_absent()
    cap = {}
    proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True,
                           task_ids=tids, capture=cap)
    assert float(cap["absent_negatives_used"]) == 1.0, cap


def test_flag_cannot_fire_without_an_absent_label():
    logits = torch.tensor([[[2.0, 0.5, -1.0], [3.0, 1.0, 0.0]]])
    gold = torch.tensor([[[True, False, False], [True, False, False]]])   # BOTH have gold
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    qmask = torch.ones(1, 2, dtype=torch.bool)
    tids = torch.zeros(1, 2, dtype=torch.long)
    cap = {}
    off = proposal_listwise_loss(logits, gold, valid, qmask)
    on = proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True,
                                task_ids=tids, capture=cap)
    assert torch.allclose(off, on)
    assert float(cap["absent_negatives_used"]) == 0.0, cap


def test_padding_queries_are_not_absent_labels():
    logits = torch.tensor([[[2.0, 0.5, -1.0], [9.0, 9.0, 9.0]]])
    gold = torch.tensor([[[True, False, False], [False, False, False]]])
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    qmask = torch.tensor([[True, False]])                                 # query 1 is PADDING
    tids = torch.tensor([[ENTITIES, ENTITIES]])
    off = proposal_listwise_loss(logits, gold, valid, qmask)
    on = proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True, task_ids=tids)
    assert torch.allclose(off, on), "padding was pooled in as an absent label"


def test_out_of_range_task_id_joins_no_pool():
    """reduce_by_task's contract: an id outside range(num_tasks) is dropped."""
    logits, gold, valid, qmask, _ = _one_gold_one_absent()
    tids = torch.tensor([[-1, -1]])
    off = proposal_listwise_loss(logits, gold, valid, qmask)
    on = proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True, task_ids=tids)
    assert torch.allclose(off, on)


def test_every_extractive_task_is_covered():
    """Not events-only: the same axis carries all four extractive task types."""
    for t in (ENTITIES, RELATIONS, EVENTS, STRUCTURES):
        logits, gold, valid, qmask, tids = _one_gold_one_absent(t, t)
        off = proposal_listwise_loss(logits, gold, valid, qmask)
        on = proposal_listwise_loss(logits, gold, valid, qmask,
                                    absent_negatives=True, task_ids=tids)
        assert on > off, f"task {t} did not pool its absent label"


def test_no_gold_anywhere_is_still_zero():
    logits = torch.randn(2, 3, 4)
    gold = torch.zeros(2, 3, 4, dtype=torch.bool)
    valid = torch.ones(2, 3, 4, dtype=torch.bool)
    qmask = torch.ones(2, 3, dtype=torch.bool)
    tids = torch.zeros(2, 3, dtype=torch.long)
    on = proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True, task_ids=tids)
    assert torch.allclose(on, torch.zeros_like(on))
