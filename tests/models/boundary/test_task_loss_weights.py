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
