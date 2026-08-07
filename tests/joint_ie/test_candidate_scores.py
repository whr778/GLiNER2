"""Sparse CandidateScoreSet + span-lattice bridge + optimizer integration."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from gliner2.joint_ie.candidate_scores import (
    CandidateScoreSet,
    MentionScore,
    ScoredRelationEdge,
    boundary_candidates_to_candidate_score_set,
    boundary_relation_pairs_to_edges,
    candidate_score_set_to_problem,
    joint_decode,
    score_lattice_to_candidate_score_set,
)
from gliner2.joint_ie.constraints import TypedEndpoints
from gliner2.joint_ie.optimizers import BeamOptimizer, GreedyOptimizer


def _score_set():
    return CandidateScoreSet(
        text="Alice works at Acme",
        mentions=(
            MentionScore(0, "person", 0, 1, 4.0, 0.98),
            MentionScore(1, "org", 3, 4, 4.0, 0.98),
            MentionScore(0, "person", 2, 3, -4.0, 0.02),  # below threshold, dropped
        ),
    )


def test_candidate_score_set_to_problem_thresholds_and_builds_edges():
    css = _score_set()
    edges = [ScoredRelationEdge("works_for", ("person", 0, 1), ("org", 3, 4), 4.0, 0.98)]
    problem = candidate_score_set_to_problem(css, edges)
    assert len(problem.nodes) == 2  # low-prob mention filtered
    assert len(problem.edges) == 1
    assert problem.edges[0].head == ("person", 0, 1)


def test_edge_referencing_dropped_mention_is_pruned():
    css = _score_set()
    bad = [ScoredRelationEdge("works_for", ("person", 2, 3), ("org", 3, 4), 4.0, 0.98)]
    problem = candidate_score_set_to_problem(css, bad)
    assert len(problem.edges) == 0


def test_typed_endpoints_and_optimizers_select_relation():
    css = _score_set()
    edges = [ScoredRelationEdge("works_for", ("person", 0, 1), ("org", 3, 4), 4.0, 0.98)]
    constraints = [TypedEndpoints("works_for", ("person",), ("org",))]
    problem = candidate_score_set_to_problem(css, edges, constraints=constraints)

    for optimizer in (GreedyOptimizer(), BeamOptimizer(beam_width=8)):
        solution = optimizer.optimize(problem)
        assert {e.relation_type for e in solution.edges} == {"works_for"}
        assert len(solution.nodes) == 2


def test_score_lattice_to_candidate_score_set_maps_halfopen_spans():
    # A minimal ScoreLattice-shaped object: one entity task, L=2, W=2.
    role_logits = torch.full((1, 2, 2, 2), -5.0)   # [count, types, L, W]
    role_logits[0, 0, 0, 0] = 5.0                   # type "a", start 0 width 0 -> [0,1)
    role_probs = torch.sigmoid(role_logits)
    hyp = SimpleNamespace(role_logits=role_logits, role_probabilities=role_probs)
    task = SimpleNamespace(
        task_type="entities", roles=("a", "b"), count_hypotheses=[hyp]
    )
    span_starts = torch.tensor([[0, 0], [1, 1]])
    span_ends = torch.tensor([[0, 1], [1, 1]])       # inclusive ends
    valid = torch.tensor([[True, True], [True, False]])
    lattice = SimpleNamespace(
        text="x y", span_starts=span_starts, span_ends=span_ends,
        valid_span_mask=valid, tasks=[task],
    )

    css = score_lattice_to_candidate_score_set(lattice)
    high = [m for m in css.mentions if m.probability > 0.5]
    assert len(high) == 1
    m = high[0]
    assert (m.entity_type, m.start, m.end) == ("a", 0, 1)  # inclusive 0 -> half-open [0,1)


def test_boundary_candidates_to_candidate_score_set_maps_sparse():
    # A minimal boundary CandidateTensorBatch: B=1, Q=2 ("person","org"), C=2.
    from gliner2.models.outputs import CandidateTensorBatch

    indices = torch.tensor([[[[0, 2], [3, 5]], [[0, 2], [0, 0]]]])   # [1,2,2,2] half-open
    pair = torch.tensor([[[5.0, -5.0], [5.0, 0.0]]])                 # [1,2,2] mention scores
    valid = torch.tensor([[[True, True], [True, False]]])            # [1,2,2]
    qmask = torch.tensor([[True, True]])                             # [1,2]
    cands = CandidateTensorBatch(
        indices=indices, proposal_logits=None, pair_logits=pair,
        valid_mask=valid, query_mask=qmask,
    )

    css = boundary_candidates_to_candidate_score_set(
        cands, ["person", "org"], "Alice works at Acme"
    )
    assert len(css.mentions) == 3  # query0: 2 valid, query1: 1 valid (second is padding)
    high = {(m.entity_type, m.start, m.end) for m in css.mentions if m.probability > 0.5}
    assert high == {("person", 0, 2), ("org", 0, 2)}

    problem = candidate_score_set_to_problem(css)  # shared builder, unchanged
    assert len(problem.nodes) == 2
    assert {n.entity_type for n in problem.nodes} == {"person", "org"}


def test_boundary_relation_pairs_to_edges_link_to_mentions():
    from gliner2.models.boundary.relations import RelationPairBatch

    pairs = RelationPairBatch(
        batch_index=torch.tensor([0]), relation_index=torch.tensor([0]),
        head_start=torch.tensor([0]), head_end=torch.tensor([2]),
        tail_start=torch.tensor([3]), tail_end=torch.tensor([5]),
        head_prob=torch.tensor([0.9]), tail_prob=torch.tensor([0.9]),
        head_keys=[("person", 0, 2)], tail_keys=[("org", 3, 5)],
        relation_types=["works_for"],
    )
    edges = boundary_relation_pairs_to_edges(pairs, [3.0])
    assert len(edges) == 1
    assert (edges[0].relation_type, edges[0].head, edges[0].tail) == (
        "works_for", ("person", 0, 2), ("org", 3, 5),
    )
    assert edges[0].probability > 0.9

    # edge keys match mention keys -> the shared builder + beam select the relation
    css = CandidateScoreSet(
        text="Alice works at Acme",
        mentions=(
            MentionScore(0, "person", 0, 2, 4.0, 0.98),
            MentionScore(1, "org", 3, 5, 4.0, 0.98),
        ),
    )
    constraints = [TypedEndpoints("works_for", ("person",), ("org",))]
    problem = candidate_score_set_to_problem(css, edges, constraints=constraints)
    assert len(problem.edges) == 1
    solution = BeamOptimizer(beam_width=8).optimize(problem)
    assert {e.relation_type for e in solution.edges} == {"works_for"}


def test_joint_decode_end_to_end_from_boundary_outputs():
    from gliner2.models.boundary.relations import RelationPairBatch
    from gliner2.models.outputs import CandidateTensorBatch

    # boundary candidates: person [0,2), org [3,5) (both confident)
    cands = CandidateTensorBatch(
        indices=torch.tensor([[[[0, 2], [0, 0]], [[3, 5], [0, 0]]]]),
        proposal_logits=None,
        pair_logits=torch.tensor([[[5.0, -5.0], [5.0, -5.0]]]),
        valid_mask=torch.tensor([[[True, False], [True, False]]]),
        query_mask=torch.tensor([[True, True]]),
    )
    pairs = RelationPairBatch(
        batch_index=torch.tensor([0]), relation_index=torch.tensor([0]),
        head_start=torch.tensor([0]), head_end=torch.tensor([2]),
        tail_start=torch.tensor([3]), tail_end=torch.tensor([5]),
        head_prob=torch.tensor([0.9]), tail_prob=torch.tensor([0.9]),
        head_keys=[("person", 0, 2)], tail_keys=[("org", 3, 5)],
        relation_types=["works_for"],
    )
    solution = joint_decode(
        cands, ["person", "org"], pairs, [3.0],
        constraints=[TypedEndpoints("works_for", ("person",), ("org",))],
        text="Alice works at Acme",
    )
    assert {n.entity_type for n in solution.nodes} == {"person", "org"}
    assert {e.relation_type for e in solution.edges} == {"works_for"}
