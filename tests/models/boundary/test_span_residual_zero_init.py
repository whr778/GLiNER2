"""span_residual_zero_init starts the post-eb21 span terms at the eb21 pair logit."""

import torch

from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head
from gliner2.models.boundary.model import BoundaryHead


def _head(flag: bool) -> BoundaryHead:
    torch.manual_seed(0)
    settings = BoundaryHeadSettings(
        boundary_dim=16,
        pair_dim=16,
        start_top_k=4,
        end_top_k=4,
        ends_per_start=2,
        starts_per_end=2,
        candidate_budget=8,
        training_candidate_budget=8,
        max_gold_per_query=4,
        content_dim=8,
        boundary_attention_layers=2,
        boundary_attention_heads=4,
        boundary_attention_window=4,
        dropout=0.0,
        enable_span_content=True,
        endpoint_difference_features=True,
        query_conditioned_inside_weight=True,
        span_residual_zero_init=flag,
    )
    return BoundaryHead(16, settings, query_dim=16)


def test_flag_round_trips_through_validation():
    settings = BoundaryHeadSettings(**validate_boundary_head({"span_residual_zero_init": True}))
    assert settings.span_residual_zero_init is True
    assert BoundaryHeadSettings(**validate_boundary_head({})).span_residual_zero_init is False


def test_zero_init_matches_the_eb21_score_and_the_default_does_not():
    head = _head(True)
    scorer = head.pair_scorer
    assert torch.count_nonzero(scorer.endpoint_difference_projection.weight) == 0
    assert torch.count_nonzero(scorer.endpoint_difference_projection.bias) == 0
    assert torch.count_nonzero(scorer.content_query_projection.weight) == 0
    assert torch.count_nonzero(scorer.content_bias.weight) == 0
    assert torch.count_nonzero(scorer.content_bias.bias) == 0
    assert float(scorer.inside_weight.bias.detach()) == 1.0
    assert torch.count_nonzero(scorer.inside_weight.weight) == 0
    query = torch.randn(1, 2, 16)
    assert torch.equal(scorer.inside_weight(query), torch.ones(1, 2, 1))
    block = head.boundary_encoder.attention_blocks[0]
    states = torch.randn(1, 6, 16)
    mask = torch.ones(1, 6, dtype=torch.bool)
    assert torch.equal(block(states, mask), states)

    plain = _head(False)
    assert torch.count_nonzero(plain.pair_scorer.endpoint_difference_projection.weight) > 0
    assert float(plain.pair_scorer.inside_weight.bias.detach()) != 1.0
