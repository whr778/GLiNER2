"""record_role_hard_negatives: a role field's BCE over gold + the K hardest wrong candidates.

The historical mean over every candidate divides a missed gold argument's surprisal by the
candidate count. Pinned: K = 0 is the historical value exactly; K > 0 equals the hand-computed
mean over exactly the gold candidates and the K highest-scoring wrong ones; a missed gold
argument then reads as a large loss, not as ~0.
"""
import pytest
import torch
import torch.nn.functional as F

from gliner2.configuration import validate_boundary_head
from gliner2.models.boundary.records import _list_field_bce


def _row(n=768, gold_col=5, gold_logit=-7.0, seed=0):
    torch.manual_seed(seed)
    cand = torch.randn(n) * 0.5 - 6.0          # 768 easy negatives, P ~ 0.0025
    cand[gold_col - 1] = gold_logit             # a MISSED gold argument, P ~ 0.0009
    cand[10] = 4.0                              # two confidently claimed wrong candidates
    cand[11] = 3.0
    return torch.cat([torch.zeros(1), cand])    # column 0 is ABSENT


def test_k0_is_the_historical_mean_over_every_candidate():
    row = _row()
    target = torch.zeros(768)
    target[4] = 1.0
    assert _list_field_bce(row, [5]) == pytest.approx(float(F.binary_cross_entropy_with_logits(row[1:], target)))
    assert _list_field_bce(row, [5], hard_k=0) == pytest.approx(_list_field_bce(row, [5]))


def test_hard_k_is_the_mean_over_gold_plus_the_k_hardest_wrong():
    row = _row()
    cand = row[1:]
    wrong = cand.clone()
    wrong[4] = float("-inf")
    keep = torch.topk(wrong, 3).indices.tolist() + [4]
    target = torch.zeros(len(keep))
    target[-1] = 1.0
    expected = float(F.binary_cross_entropy_with_logits(cand[keep], target))
    assert _list_field_bce(row, [5], hard_k=3) == pytest.approx(expected)


def test_a_missed_gold_argument_is_no_longer_diluted():
    row = _row()
    assert _list_field_bce(row, [5]) < 0.05, "the fixture must reproduce the diluted reading"
    assert _list_field_bce(row, [5], hard_k=8) > 1.0


def test_an_empty_role_keeps_only_the_hardest_wrong_candidates():
    row = _row()
    assert _list_field_bce(row, [], hard_k=2) > 3.0      # the two claimed candidates dominate
    assert _list_field_bce(row, []) < 0.05


def test_setting_validates():
    assert validate_boundary_head({})["record_role_hard_negatives"] == 0
    with pytest.raises(ValueError):
        validate_boundary_head({"record_role_hard_negatives": -1})
