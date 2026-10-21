"""decompose_pair_score.parts: the additive parts must sum to SparseBoundaryPairScorer.forward on valid candidates."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "train"))
from decompose_pair_score import PARTS, parts  # noqa: E402
from gliner2.models.boundary.proposal import ProposalSettings, SparseBoundaryProposer  # noqa: E402
from gliner2.models.boundary.scoring import SparseBoundaryPairScorer  # noqa: E402


@pytest.mark.parametrize("scorer_states", [True, False])
def test_parts_sum_to_forward_with_every_option(scorer_states):
    torch.manual_seed(0)
    b, l, d, q = 1, 10, 8, 2
    settings = ProposalSettings(start_top_k=4, end_top_k=4, ends_per_start=3, starts_per_end=3, candidate_budget=8,
                                training_candidate_budget=12, max_gold_per_query=6, end_block_size=4,
                                bidirectional=True)
    proposer = SparseBoundaryProposer(boundary_dim=d, query_dim=d, settings=settings).eval()
    scorer = SparseBoundaryPairScorer(boundary_dim=d, query_dim=d, pair_dim=d, enable_span_content=True,
                                      enable_rotary_endpoints=True, query_conditioned_inside_weight=True,
                                      endpoint_difference_features=True).eval()
    boundary = torch.randn(b, l + 1, d)
    query = torch.randn(b, q, d)
    start_logits, end_logits = torch.randn(b, q, l + 1), torch.randn(b, q, l + 1)
    s_states, e_states = scorer.project_endpoints(boundary) if scorer_states else (None, None)
    proposals = proposer(boundary, torch.ones(b, l + 1, dtype=torch.bool), query, torch.ones(b, q, dtype=torch.bool),
                         start_logits, end_logits, scorer_start_states=s_states, scorer_end_states=e_states)
    args = (boundary, query, proposals, start_logits, end_logits, torch.randn(b, q, l + 1), torch.tensor([l]),
            torch.randn(b, l, d), torch.ones(b, l, dtype=torch.bool))
    with torch.no_grad():
        logits = scorer(*args)
        total = sum(parts(scorer, *args)[k] for k in PARTS)
    valid = proposals.valid_mask
    assert valid.any()
    assert torch.allclose(total[valid], logits[valid], atol=1e-5)
