"""Absent-label negatives must enter the listwise denominator, and only when they exist.

Until 2026-09-19 `proposal_listwise_loss` skipped any query with no gold, so an injected
label negative -- a label mapped to an empty list -- contributed EXACTLY ZERO to both
listwise losses. These tests fail without the fix.
"""

import torch

from gliner2.models.boundary.losses import proposal_listwise_loss


def _batch():
    """One sample: query 0 has gold, query 1 is an ABSENT label (no gold)."""
    logits = torch.tensor([[[2.0, 0.5, -1.0], [3.0, 1.0, 0.0]]])   # [1, 2, 3]
    gold = torch.tensor([[[True, False, False], [False, False, False]]])
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    qmask = torch.ones(1, 2, dtype=torch.bool)
    return logits, gold, valid, qmask


def test_absent_label_enters_the_denominator():
    logits, gold, valid, qmask = _batch()
    off = proposal_listwise_loss(logits, gold, valid, qmask)
    on = proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True)
    # The denominator grew by the absent query's candidates, so the loss must RISE.
    assert on > off, f"absent negatives did not reach the denominator: {on} !> {off}"


def test_gate_reports_what_entered():
    logits, gold, valid, qmask = _batch()
    cap = {}
    proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True, capture=cap)
    assert float(cap["absent_negatives_used"]) == 1.0, cap


def test_flag_cannot_fire_without_an_absent_label():
    """A gate that fires when it should not is as bad as one that never fires."""
    logits = torch.tensor([[[2.0, 0.5, -1.0], [3.0, 1.0, 0.0]]])
    gold = torch.tensor([[[True, False, False], [True, False, False]]])  # BOTH have gold
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    qmask = torch.ones(1, 2, dtype=torch.bool)
    cap = {}
    off = proposal_listwise_loss(logits, gold, valid, qmask)
    on = proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True, capture=cap)
    assert torch.allclose(off, on), "the flag changed a batch with no absent labels"
    assert float(cap["absent_negatives_used"]) == 0.0, cap


def test_padding_queries_are_not_treated_as_absent_labels():
    """query_mask False is PADDING, not an absent label -- it must not become a negative."""
    logits = torch.tensor([[[2.0, 0.5, -1.0], [9.0, 9.0, 9.0]]])   # padded query scores high
    gold = torch.tensor([[[True, False, False], [False, False, False]]])
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    qmask = torch.tensor([[True, False]])                          # query 1 is PADDING
    cap = {}
    off = proposal_listwise_loss(logits, gold, valid, qmask)
    on = proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True, capture=cap)
    assert torch.allclose(off, on), "padding was pooled in as an absent label"
    assert float(cap["absent_negatives_used"]) == 0.0, cap


def test_no_gold_anywhere_is_still_zero():
    """A sample with no gold at all must stay at zero loss, not become negative."""
    logits = torch.randn(2, 3, 4)
    gold = torch.zeros(2, 3, 4, dtype=torch.bool)
    valid = torch.ones(2, 3, 4, dtype=torch.bool)
    qmask = torch.ones(2, 3, dtype=torch.bool)
    on = proposal_listwise_loss(logits, gold, valid, qmask, absent_negatives=True)
    assert torch.allclose(on, torch.zeros_like(on)), on
