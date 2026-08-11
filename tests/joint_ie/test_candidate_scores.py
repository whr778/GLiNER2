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


def _two_relation_layout():
    """Two relation types sharing the Standard Format's ``head``/``tail`` field names."""
    from gliner2.models.base import QueryLayout, QuerySpec

    names = [("deaths_in", "head"), ("deaths_in", "tail"),
             ("injured_in", "head"), ("injured_in", "tail")]
    return QueryLayout(queries=tuple(
        QuerySpec(query_id=i, task_index=i // 2, task_type="relations",
                  task_name=task, role_index=i % 2, role_name=role,
                  field_path=(task, role), extractive=True)
        for i, (task, role) in enumerate(names)
    ))


def test_two_relation_types_sharing_field_names_keep_distinct_nodes_and_edges():
    """Regression: relation types sharing field names must not collide, and edges must live.

    Two failures hide here and only one of them raises. Bare ``(field_name, start, end)``
    keys make two relation types produce identical node ids and ``JointProblem`` rejects the
    problem. Qualifying only *one* side instead silently drops every edge through
    ``candidate_score_set_to_problem``'s ``keep_ids`` filter and the decode returns empty --
    which reads as "the model found nothing". So this asserts edges SURVIVE, not merely that
    nothing raised.
    """
    from gliner2.models.base import qualified_query_type
    from gliner2.models.boundary.relations import (
        RelationTypeSpec,
        TypedRelationPairGenerator,
    )
    from gliner2.models.outputs import CandidateTensorBatch

    layout = _two_relation_layout()
    # All four queries propose the SAME two spans -- the colliding case.
    spans = [[(0, 2), (3, 5)]] * 4
    indices = torch.zeros(1, 4, 2, 2, dtype=torch.long)
    valid = torch.zeros(1, 4, 2, dtype=torch.bool)
    pair_logits = torch.zeros(1, 4, 2)
    for q, per_query in enumerate(spans):
        for c, (s, e) in enumerate(per_query):
            indices[0, q, c, 0], indices[0, q, c, 1] = s, e
            valid[0, q, c] = True
            pair_logits[0, q, c] = 5.0 + 0.1 * q  # distinct per query, as in a real model
    cands = CandidateTensorBatch(
        indices=indices, proposal_logits=None, pair_logits=pair_logits,
        valid_mask=valid, query_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    specs = [RelationTypeSpec("deaths_in", (0,), (1,)),
             RelationTypeSpec("injured_in", (2,), (3,))]
    pairs = TypedRelationPairGenerator().generate(cands, [layout], specs)
    assert len(pairs) > 0

    query_types = [qualified_query_type(q.query_id, q.role_name) for q in layout.queries]
    css = boundary_candidates_to_candidate_score_set(cands, query_types, "text")
    keys = [m.key for m in css.mentions]
    assert len(keys) == len(set(keys)), "mention keys collide across relation types"

    edges = boundary_relation_pairs_to_edges(pairs, [3.0] * len(pairs))
    problem = candidate_score_set_to_problem(css, edges)   # would raise on collision
    assert len(problem.nodes) == len(css.mentions)
    # The assertion that catches a one-sided fix: keys must MATCH, not merely be unique.
    assert len(problem.edges) == len(edges) > 0, "edges dropped by the keep_ids filter"

    both = {e.relation_type for e in problem.edges}
    assert both == {"deaths_in", "injured_in"}


def test_typed_endpoints_discriminate_between_relation_types():
    """Qualified endpoint types make a relation's constraint actually constrain.

    With bare role names both relations above declare ``("head",)/("tail",)``, so
    ``TypedEndpoints`` is satisfied by any endpoint of either relation and forbids nothing.
    """
    from gliner2.models.base import qualified_query_type

    layout = _two_relation_layout()
    heads = [qualified_query_type(q.query_id, q.role_name)
             for q in layout.queries if q.role_name == "head"]
    assert len(set(heads)) == 2, "both relations' head queries share one type name"

    # An edge that borrows the other relation's endpoints is now rejected.
    css = CandidateScoreSet(text="t", mentions=(
        MentionScore(0, heads[0], 0, 2, 4.0, 0.98),
        MentionScore(1, "1::tail", 3, 5, 4.0, 0.98),
        MentionScore(2, heads[1], 6, 8, 4.0, 0.98),
    ))
    crossed = ScoredRelationEdge("deaths_in", (heads[1], 6, 8), ("1::tail", 3, 5), 3.0, 0.95)
    constraints = [TypedEndpoints("deaths_in", (heads[0],), ("1::tail",))]
    problem = candidate_score_set_to_problem(css, [crossed], constraints=constraints)
    solution = BeamOptimizer(beam_width=8).optimize(problem)
    assert not solution.edges, "cross-type edge should be forbidden by TypedEndpoints"


def test_decision_threshold_moves_edge_selection():
    """Regression: the caller's threshold must reach EDGE selection, not just node admission.

    `joint_decode` used to filter mentions by `mention_threshold` while leaving every
    utility centered on 0.5, so `gain > 0` still demanded p > 0.5 for edges no matter what
    threshold was asked for. Nothing raised -- the decode just stopped responding to
    `--threshold`, which reads as a model that is insensitive to calibration rather than a
    plumbing bug. Measured on Re-DocRED before the fix: joint recall moved 0.1498 -> 0.1591
    across thresholds 0.5 -> 0.1 while the greedy arm moved 0.0461 -> 0.4134.
    """
    # One sub-0.5 edge between two sub-0.5 mentions: selected only at a low threshold.
    css = CandidateScoreSet(text="t", mentions=(
        MentionScore(0, "person", 0, 2, -1.0, 0.269),
        MentionScore(1, "org", 3, 5, -1.0, 0.269),
    ))
    edges = [ScoredRelationEdge("works_for", ("person", 0, 2), ("org", 3, 5), -1.0, 0.269)]

    strict = candidate_score_set_to_problem(css, edges, mention_threshold=0.1)
    assert [e.score for e in strict.edges] == [-1.0]      # centered on 0.5 -> negative
    assert BeamOptimizer(beam_width=8).optimize(strict).edges == ()

    loose = candidate_score_set_to_problem(
        css, edges, mention_threshold=0.1, decision_threshold=0.1)
    assert loose.edges[0].score > 0.0                     # now above the asked-for cutoff
    assert {e.relation_type for e in BeamOptimizer(beam_width=8).optimize(loose).edges} \
        == {"works_for"}


def test_record_role_edges_are_not_recentered_by_the_threshold():
    """Role-edge utilities are ABSENT-relative and must bypass threshold centering.

    A scalar role's utility is ``logit_c - logit_ABSENT`` -- a comparison against the
    record head's own ABSENT class, not against a probability cutoff. Shifting it by a
    threshold offset would move scalar roles against a baseline they do not have.
    """
    css = CandidateScoreSet(text="t", mentions=(
        MentionScore(0, "trigger", 0, 2, 4.0, 0.98),
        MentionScore(1, "place", 3, 5, 4.0, 0.98),
    ))
    role = ScoredRelationEdge("Attack::place", ("trigger", 0, 2), ("place", 3, 5),
                              0.75, 0.68, slot="place", hypothesis=("trigger", 0, 2))

    for threshold in (0.5, 0.1, 0.9):
        problem = candidate_score_set_to_problem(
            css, (), mention_threshold=0.05, decision_threshold=threshold,
            pre_scored_edges=[role])
        assert [e.score for e in problem.edges] == [0.75], threshold

    # ...while an ordinary relation edge at the same logit DOES move with the threshold.
    plain = ScoredRelationEdge("works_for", ("trigger", 0, 2), ("place", 3, 5), 0.75, 0.68)
    moved = {candidate_score_set_to_problem(
        css, [plain], mention_threshold=0.05, decision_threshold=t).edges[0].score
        for t in (0.5, 0.1, 0.9)}
    assert len(moved) == 3
