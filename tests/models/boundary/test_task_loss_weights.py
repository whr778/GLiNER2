"""Per-query task loss weighting (EVENT_LOSS_PHASE3_PLAN.md, gate 1).

The boundary loss decomposes by mechanism (start/end/pair/...), not by task, so
rebalancing event supervision against entity/relation supervision means scaling
each query's contribution by its task type. These tests pin the two properties
the experiment depends on:

* **inert at 1.0** -- an all-ones weight vector reproduces the unweighted loss
  exactly, so any arm-to-arm difference is the weight and not the plumbing;
* **live above 1.0** -- a non-uniform weight moves exactly the three terms that
  route through ``_reduce`` (start, end, pair) and nothing else.

Written against ``_reduce`` directly: it is the single choke point all three
losses share, and testing it here needs no model, no tokenizer and no download.
"""

from __future__ import annotations

import pytest
import torch

from gliner2.models.boundary.losses import (
    _reduce,
    _safe_bce,
    balanced_multilabel_bce,
    reduce_by_task,
)


@pytest.fixture
def masked_batch():
    torch.manual_seed(1701)
    elementwise = torch.rand(2, 4, 5)
    keep = torch.ones(2, 4, 5, dtype=torch.bool)
    keep[0, 3] = False  # one fully-masked query
    query_mask = torch.ones(2, 4, dtype=torch.bool)
    return elementwise, keep, query_mask


@pytest.mark.parametrize("mode", ["global", "per_query"])
def test_all_ones_weights_are_exactly_inert(masked_batch, mode):
    elementwise, keep, query_mask = masked_batch
    ones = torch.ones(2, 4)

    unweighted = _reduce(elementwise, keep, query_mask, mode)
    weighted = _reduce(elementwise, keep, query_mask, mode, ones)

    assert torch.equal(unweighted, weighted)


@pytest.mark.parametrize("mode", ["global", "per_query"])
def test_response_is_linear_in_weight_minus_one(masked_batch, mode):
    """Scaling one query's weight to w moves the loss by (w-1) x its share.

    Linearity is what makes a dose-response arm sweep interpretable: the arms
    differ by a known multiple, not by an arbitrary monotone transform.
    """
    elementwise, keep, query_mask = masked_batch
    base = _reduce(elementwise, keep, query_mask, mode)

    def at(w: float) -> float:
        weights = torch.ones(2, 4)
        weights[0, 1] = w
        return float(_reduce(elementwise, keep, query_mask, mode, weights))

    unit = at(2.0) - float(base)          # the (w-1)=1 step
    assert unit > 0
    assert at(4.0) - float(base) == pytest.approx(3 * unit, rel=1e-5)
    assert at(0.5) - float(base) == pytest.approx(-0.5 * unit, rel=1e-5)


def test_zero_weight_removes_a_query_from_the_loss(masked_batch):
    """A weight of 0.0 must drop that query's contribution entirely, while the
    denominator stays the unweighted active count -- so the loss falls rather
    than renormalizing back up."""
    elementwise, keep, query_mask = masked_batch
    weights = torch.ones(2, 4)
    weights[0, 1] = 0.0

    base = float(_reduce(elementwise, keep, query_mask, "per_query"))
    dropped = float(_reduce(elementwise, keep, query_mask, "per_query", weights))

    assert dropped < base


def test_weights_reach_the_public_loss_entry_point():
    """Guard the kwarg wiring, not just ``_reduce``: an unthreaded kwarg would
    silently make every arm identical."""
    torch.manual_seed(7)
    logits = torch.randn(1, 2, 6)
    targets = (torch.rand(1, 2, 6) > 0.7).float()
    valid = torch.ones(1, 2, 6, dtype=torch.bool)
    query_mask = torch.ones(1, 2, dtype=torch.bool)

    plain = float(balanced_multilabel_bce(logits, targets, valid, query_mask=query_mask))
    ones = float(
        balanced_multilabel_bce(
            logits, targets, valid, query_mask=query_mask, query_weights=torch.ones(1, 2)
        )
    )
    heavy = float(
        balanced_multilabel_bce(
            logits,
            targets,
            valid,
            query_mask=query_mask,
            query_weights=torch.tensor([[1.0, 3.0]]),
        )
    )

    assert plain == pytest.approx(ones, rel=1e-6)
    assert heavy != pytest.approx(plain, rel=1e-6)


# ---------------------------------------------------------------------------
# Per-task loss buckets (EVENT_LOSS_PHASE3_PLAN step 4)
# ---------------------------------------------------------------------------

def _task_ids(rows, width, assignment):
    ids = torch.full((rows, width), -1, dtype=torch.long)
    for (b, q), t in assignment.items():
        ids[b, q] = t
    return ids


@pytest.mark.parametrize("mode", ["global", "per_query"])
def test_task_buckets_sum_to_the_unweighted_total(masked_batch, mode):
    """The phase-2 invariant, carried to a mechanism-decomposed loss:
    ``structure + event_structure == old structure``. If the buckets do not
    reconcile against the scalar the optimizer sees, they are decoration."""
    elementwise, keep, query_mask = masked_batch
    ids = _task_ids(2, 4, {(b, q): q % 3 for b in range(2) for q in range(4)})

    total = _reduce(elementwise, keep, query_mask, mode)
    buckets = reduce_by_task(elementwise, keep, query_mask, mode, ids, 3)

    assert float(buckets.sum()) == pytest.approx(float(total), rel=1e-6)


def test_a_task_with_no_queries_contributes_zero(masked_batch):
    elementwise, keep, query_mask = masked_batch
    ids = _task_ids(2, 4, {(b, q): 0 for b in range(2) for q in range(4)})

    buckets = reduce_by_task(elementwise, keep, query_mask, "per_query", ids, 3)

    assert float(buckets[1]) == 0.0 and float(buckets[2]) == 0.0
    assert float(buckets[0]) > 0.0


def test_unassigned_queries_are_dropped_not_misattributed(masked_batch):
    """A padded query carries id -1. Silently folding it into task 0 would
    inflate whichever task happens to be first."""
    elementwise, keep, query_mask = masked_batch
    all_zero = _task_ids(2, 4, {(b, q): 0 for b in range(2) for q in range(4)})
    some_pad = _task_ids(2, 4, {(b, q): 0 for b in range(2) for q in range(3)})  # q=3 -> -1

    full = reduce_by_task(elementwise, keep, query_mask, "per_query", all_zero, 2)
    padded = reduce_by_task(elementwise, keep, query_mask, "per_query", some_pad, 2)

    assert float(padded[0]) < float(full[0])


@pytest.mark.parametrize("mode", ["global", "per_query"])
def test_positive_mass_splits_the_same_whole(masked_batch, mode):
    """``numerator_mask`` restricts the numerator only. Positive and negative
    mass must therefore add back to the unrestricted bucket -- that identity is
    what makes a ``pos_weight`` dose computable: scaling positives by ``k``
    multiplies a task's contribution by ``(k*pos + neg) / (pos + neg)``."""
    elementwise, keep, query_mask = masked_batch
    ids = _task_ids(2, 4, {(b, q): q % 3 for b in range(2) for q in range(4)})
    positive = torch.zeros_like(keep)
    positive[..., ::2] = True

    total = reduce_by_task(elementwise, keep, query_mask, mode, ids, 3)
    pos = reduce_by_task(
        elementwise, keep, query_mask, mode, ids, 3, numerator_mask=positive
    )
    neg = reduce_by_task(
        elementwise, keep, query_mask, mode, ids, 3, numerator_mask=~positive
    )

    assert pos.sum() > 0 and neg.sum() > 0
    torch.testing.assert_close(pos + neg, total, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Per-task pos_weight (EVENT_LOSS_PHASE3_PLAN step 9)
# ---------------------------------------------------------------------------

@pytest.fixture
def bce_batch():
    """Logits/targets with a per-query positive rate that differs by query."""
    torch.manual_seed(90210)
    logits = torch.randn(2, 4, 6)
    targets = (torch.rand(2, 4, 6) < 0.4).float()
    keep = torch.ones(2, 4, 6, dtype=torch.bool)
    keep[0, 3] = False
    return logits, targets, keep


def test_pos_weight_of_one_is_exactly_inert(bce_batch):
    """A dose of 1.0 must reproduce the unweighted BCE bit-for-bit, or an arm's
    difference is the plumbing rather than the treatment."""
    logits, targets, keep = bce_batch
    plain = _safe_bce(logits, targets, keep)
    ones = _safe_bce(logits, targets, keep, torch.ones(2, 4, 1))

    assert torch.equal(plain, ones)


def test_pos_weight_scales_only_the_positive_term(bce_batch):
    """The negative term must be untouched -- that is the whole distinction from
    `task_loss_weights`, which scales positives and negatives alike."""
    logits, targets, keep = bce_batch
    plain = _safe_bce(logits, targets, keep)
    heavy = _safe_bce(logits, targets, keep, torch.full((2, 4, 1), 4.0))

    # The EFFECTIVE positives: a masked position carries a zeroed target, so it is
    # a pure negative term no matter what the label said. pos_weight must not be
    # able to resurrect supervision the mask removed.
    positive = (targets > 0.5) & keep
    torch.testing.assert_close(heavy[~positive], plain[~positive])
    torch.testing.assert_close(heavy[positive], 4.0 * plain[positive])
    masked_positive = (targets > 0.5) & ~keep
    assert masked_positive.any()
    torch.testing.assert_close(heavy[masked_positive], plain[masked_positive])


def test_pos_weight_is_per_query_not_global(bce_batch):
    """Events must be scalable without touching entity queries in the same batch."""
    logits, targets, keep = bce_batch
    dose = torch.ones(2, 4, 1)
    dose[:, 1] = 8.0

    out = _safe_bce(logits, targets, keep, dose)
    plain = _safe_bce(logits, targets, keep)

    torch.testing.assert_close(out[:, 0], plain[:, 0])
    positive = targets[:, 1] > 0.5
    torch.testing.assert_close(out[:, 1][positive], 8.0 * plain[:, 1][positive])


@pytest.mark.parametrize("k", [2.0, 4.0, 8.0, 16.0])
def test_the_dose_formula_predicts_the_bucket(bce_batch, k):
    """`(k*pos + neg) / (pos + neg)` is how the arm doses were chosen from the
    probe's measured positive fraction. If the formula does not predict the
    bucket, the doses are guesses again."""
    logits, targets, keep = bce_batch
    query_mask = torch.ones(2, 4, dtype=torch.bool)
    ids = _task_ids(2, 4, {(b, q): 0 for b in range(2) for q in range(4)})
    positive = targets > 0.5

    plain = _safe_bce(logits, targets, keep)
    pos = float(reduce_by_task(plain, keep, query_mask, "global", ids, 1,
                               numerator_mask=positive)[0])
    neg = float(reduce_by_task(plain, keep, query_mask, "global", ids, 1,
                               numerator_mask=~positive)[0])
    predicted = (k * pos + neg) / (pos + neg)

    dosed = _safe_bce(logits, targets, keep, torch.full((2, 4, 1), k))
    actual = float(reduce_by_task(dosed, keep, query_mask, "global", ids, 1)[0]) / (pos + neg)

    assert actual == pytest.approx(predicted, rel=1e-5)
