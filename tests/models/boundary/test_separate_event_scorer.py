"""separate_event_scorer: event queries take their OWN proposer + pair scorer, copied from the shared pair.

The eb22c separation A/B (2026-10-21) needs four properties to read as one change: off by default; the copies start
identical to their sources and leave every other module's initialisation untouched; each query is routed to its own
pair; and a forward that forgets the routing mask refuses instead of silently using the shared pair.
"""
import pytest
import torch

from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head
from gliner2.models.boundary.model import BoundaryHead
from gliner2.processing.targets import MentionTarget, TargetGraph, pad_target_graphs


def _settings(flag):
    return BoundaryHeadSettings(boundary_dim=8, pair_dim=8, start_top_k=4, end_top_k=4, ends_per_start=3,
                                starts_per_end=3, candidate_budget=12, training_candidate_budget=12,
                                max_gold_per_query=4, dropout=0.0, separate_event_scorer=flag)


def _head(flag):
    torch.manual_seed(0)
    return BoundaryHead(8, _settings(flag))


def _inputs(q=4):
    torch.manual_seed(1)
    return (torch.randn(1, 6, 8), torch.ones(1, 6, dtype=torch.bool), torch.randn(1, q, 8),
            torch.ones(1, q, dtype=torch.bool))


def test_off_by_default_and_refused_on_shared_pool():
    assert BoundaryHeadSettings().separate_event_scorer is False
    assert _head(False).event_pair_scorer is None
    with pytest.raises(ValueError, match="separate_event_scorer"):
        validate_boundary_head({"separate_event_scorer": True, "candidate_pool": "shared"})


def test_copies_start_identical_and_leave_other_inits_untouched():
    off, on = _head(False).state_dict(), _head(True).state_dict()
    assert all(torch.equal(off[k], on[k]) for k in off)
    extra = [k for k in on if k not in off]
    assert extra and all(k.startswith(("event_proposer.", "event_pair_scorer.")) for k in extra)
    src = lambda k: k.replace("event_proposer.", "boundary_proposer.").replace("event_pair_scorer.", "pair_scorer.")
    assert all(torch.equal(on[k], on[src(k)]) for k in extra)


def test_identical_weights_give_identical_logits_and_a_missing_mask_refuses():
    off, on = _head(False).eval(), _head(True).eval()
    events = torch.tensor([[False, True, True, False]])
    with torch.no_grad():
        a = off(*_inputs(), return_candidates=True).candidates.pair_logits
        b = on(*_inputs(), return_candidates=True, event_queries=events).candidates.pair_logits
        assert torch.equal(a, b)
        with pytest.raises(ValueError, match="event_queries"):
            on(*_inputs(), return_candidates=True)


def test_each_query_routes_to_its_own_pair():
    on = _head(True).eval()
    events = torch.tensor([[False, True, True, False]])
    with torch.no_grad():
        before = on(*_inputs(), return_candidates=True, event_queries=events).candidates.pair_logits
        for p in on.event_pair_scorer.parameters():
            p.add_(0.5)
        after = on(*_inputs(), return_candidates=True, event_queries=events).candidates.pair_logits
    e = events[0]
    assert torch.equal(before[0, ~e], after[0, ~e])
    assert not torch.equal(before[0, e], after[0, e])


def test_gradients_reach_each_pair_only_from_its_own_queries():
    on = _head(True).train()
    events = torch.tensor([[False, True, True, False]])
    graph = TargetGraph(mentions=(MentionTarget(0, 1, 3), MentionTarget(1, 2, 4)))
    targets = pad_target_graphs([graph], [4], [6], 4)
    out = on(*_inputs(), targets, return_candidates=True, event_queries=events)
    logits, valid, e = out.candidates.pair_logits, out.candidates.valid_mask, events[0]
    ent, evt = logits[0][~e][valid[0][~e]].sum(), logits[0][e][valid[0][e]].sum()
    norm = lambda loss, mod: sum(float(g.norm()) for g in torch.autograd.grad(
        loss, list(mod.parameters()), retain_graph=True, allow_unused=True) if g is not None)
    assert norm(ent, on.event_pair_scorer) == 0.0 and norm(evt, on.pair_scorer) == 0.0
    assert norm(evt, on.event_pair_scorer) > 0.0 and norm(ent, on.pair_scorer) > 0.0
