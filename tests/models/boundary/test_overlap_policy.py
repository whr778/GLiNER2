"""Boundary overlap-policy and export-validation tests."""

import pytest

from gliner2.configuration import validate_boundary_head
from gliner2.models.boundary.engine import _resolve_flat_spans, _resolve_spans


def test_flat_policy_uses_optimal_total_score_not_greedy():
    scored = [
        (0.9, 0, 10),
        (0.6, 0, 5),
        (0.6, 5, 10),
    ]
    assert _resolve_flat_spans(scored) == [(0.6, 0, 5), (0.6, 5, 10)]


def test_nested_policy_keeps_containment_but_rejects_crossing():
    scored = [(0.6, 1, 4), (0.9, 0, 5), (0.7, 3, 8)]
    assert _resolve_spans(scored, "nested") == [
        (0.9, 0, 5),
        (0.6, 1, 4),
    ]


def test_longest_drops_strictly_contained_lower_score_span():
    scored = [(0.9, 0, 8), (0.8, 2, 4), (0.7, 8, 10)]
    assert _resolve_spans(scored, "longest") == [
        (0.9, 0, 8),
        (0.7, 8, 10),
    ]


@pytest.mark.parametrize(
    "policy", ["allow", "disallow", "flat", "nested", "longest"]
)
def test_overlap_policy_config_accepts_supported_values(policy):
    if policy in {"allow", "disallow"}:
        # Runtime-only aliases are normalized without mutating the serialized
        # boundary configuration contract.
        from gliner2.inference.overlap import normalize_overlap_policy

        assert normalize_overlap_policy(policy)
    else:
        assert validate_boundary_head({"overlap_policy": policy})["overlap_policy"] == policy


def test_adaptive_budget_rejects_vectorized_export():
    with pytest.raises(ValueError, match="adaptive boundary budget"):
        validate_boundary_head(
            {"export_mode": "vectorized", "boundary_top_k_alpha": 0.08}
        )


def test_nested_policy_keeps_containment_and_drops_crossing_three_span():
    """Our pre-g2.5 scenario, re-baselined on the current contract.

    Before the g2.5 merge `nested` kept every ranked candidate, so this returned all
    three. It now keeps containment and DROPS crossing spans: (3, 8) crosses (0, 5)
    without being contained by it, so it goes. Kept because upstream's own nested test
    exercises only two spans and never covers the crossing-plus-contained mix.
    """
    scored = [(0.6, 1, 4), (0.9, 0, 5), (0.7, 3, 8)]
    assert _resolve_spans(scored, "nested") == [
        (0.9, 0, 5),
        (0.6, 1, 4),
    ]


