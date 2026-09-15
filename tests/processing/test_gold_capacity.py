"""`skip_sample` does not withhold supervision -- it teaches the model to ABSTAIN.

WHY THIS FILE EXISTS. `on_capacity_exceeded="skip_sample"` drops gold for the whole sample
when ANY ONE of its queries exceeds `max_gold_per_query`. That sounds like "this document
contributes nothing", and it is worse than that: an all-False `mention_mask` row is the
POSITIVE target for abstention. `abstention_loss` trains its per-query gate on
`~mention_mask.any(-1)` (models/boundary/losses.py:601) and `count_log_rate_loss` supervises
a count of zero, so a skipped sample actively teaches the model to emit nothing -- on exactly
the documents that carry the most gold.

Measured on the eb16-eventrecords-tr run: it fired 3,520 times, on overflows of 34-66 against
the default cap of 32, and plausibly explains that run's entity regression (strict 0.5340 ->
0.4343).

The fix is a raised cap, sized from the data and aligned to the eleven configs that already
raised it: `max_gold_per_query: 256` with `training_candidate_budget: 384`. Of 89,392
(doc, label) query groups across those corpora, 0.789% exceed 32 and 0.011% exceed 256;
tier2/archive/casie-* records a single query carrying 188. These tests pin the hazard so the
cap is never quietly lowered back, and pin that a sufficient cap keeps every query's gold.
"""

import pytest

from gliner2.processing.targets import (
    MentionTarget,
    TargetGraph,
    pad_target_graphs,
)


def _sample(n_overflow: int, n_ok: int = 3) -> TargetGraph:
    """One sample, two queries: query 0 crowded, query 1 comfortably under any cap."""
    mentions = [MentionTarget(query_id=0, start=i, end=i + 1) for i in range(n_overflow)]
    mentions += [MentionTarget(query_id=1, start=i, end=i + 2) for i in range(n_ok)]
    return TargetGraph(mentions=tuple(mentions))


def _pad(graph, cap, policy):
    return pad_target_graphs(
        [graph], query_counts=[2], text_lengths=[256],
        max_gold_per_query=cap, on_capacity_exceeded=policy, build_dense=False,
    )


def test_skip_sample_destroys_gold_for_the_INNOCENT_query_too():
    """The documented hazard, pinned. Query 1 never overflowed and still loses everything."""
    out = _pad(_sample(n_overflow=40), cap=32, policy="skip_sample")

    assert not out.mention_mask[0, 0].any(), "the overflowing query is cleared, as documented"
    assert not out.mention_mask[0, 1].any(), (
        "query 1 holds 3 gold spans and never came near the cap, but skip_sample clears the "
        "WHOLE sample -- this is the defect the raised cap exists to avoid"
    )


def test_an_empty_mask_is_the_ABSTAIN_target_not_an_absence_of_supervision():
    """Why the above is harmful rather than merely wasteful.

    This is the exact expression abstention_loss trains on, so a skipped sample supplies a
    target of 1.0 ("nothing here") for every query in the document.
    """
    out = _pad(_sample(n_overflow=40), cap=32, policy="skip_sample")

    abstain_target = (~out.mention_mask.any(-1)).float()
    assert abstain_target[0, 0].item() == 1.0
    assert abstain_target[0, 1].item() == 1.0, (
        "a cleared query is supervised as ABSENT, i.e. the model is taught to emit nothing"
    )


def test_a_cap_sized_for_the_data_keeps_every_query():
    """256 is what the live configs set; the same sample now trains on all of its gold."""
    out = _pad(_sample(n_overflow=40), cap=256, policy="skip_sample")

    assert out.mention_mask[0, 0].sum().item() == 40
    assert out.mention_mask[0, 1].sum().item() == 3


def test_truncate_keeps_the_innocent_query_but_still_loses_the_overflow():
    """The alternative policy, so the trade-off is on record rather than assumed.

    Truncation is more localised than skip_sample -- query 1 survives -- but the 8 dropped
    spans in query 0 become implicit negatives, and the count target is wrong. Raising the cap
    is preferable to either.
    """
    out = _pad(_sample(n_overflow=40), cap=32, policy="truncate_with_warning")

    assert out.mention_mask[0, 0].sum().item() == 32, "capped, losing 8 real spans"
    assert out.mention_mask[0, 1].sum().item() == 3, "the innocent query survives truncation"


def test_raise_is_the_default_so_silent_loss_is_opt_in():
    from gliner2.processing.targets import TargetCapacityError

    with pytest.raises(TargetCapacityError):
        pad_target_graphs(
            [_sample(n_overflow=40)], query_counts=[2], text_lengths=[256],
            max_gold_per_query=32, build_dense=False,
        )
