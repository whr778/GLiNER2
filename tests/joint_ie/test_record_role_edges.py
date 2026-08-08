"""Event records as trigger node + role edges (JOINT_IE_SCALING sec 3b).

An event instance *is* its trigger node, so records need no new candidate class.
The load-bearing property is the same one the relation path had to prove: role
edges must key their endpoints exactly as the mention adapter keys its nodes, or
every role edge references a node that does not exist and gets dropped.
"""

from __future__ import annotations

import torch

from gliner2.joint_ie.candidate_scores import (
    CandidateScoreSet,
    MentionScore,
    boundary_record_groups_to_role_edges,
    candidate_score_set_to_problem,
)
from gliner2.models.boundary.records import RecordGroupOutput
from gliner2.processing.records import FieldCardinality, RecordFieldSpec, RecordSpec

# query 0 = trigger (anchor), 1 = target (scalar role), 2 = instrument (list role)
QUERY_TYPES = ["trigger", "target", "instrument"]

TRIGGER = RecordFieldSpec(query_id=0, name="trigger", role_index=0,
                          cardinality=FieldCardinality.REQUIRED_ONE, is_anchor=True)
TARGET = RecordFieldSpec(query_id=1, name="target", role_index=1,
                         cardinality=FieldCardinality.OPTIONAL_ONE)
INSTRUMENT = RecordFieldSpec(query_id=2, name="instrument", role_index=2,
                             cardinality=FieldCardinality.ZERO_OR_MORE)

SPEC = RecordSpec(
    task_index=0, task_name="attack", task_type="events", mode="natural",
    fields=(TRIGGER, TARGET, INSTRUMENT), anchor_query_id=0,
)


def _group():
    """One instance: trigger [0,1); target candidate [2,3); instrument cand [4,5)."""
    return RecordGroupOutput(
        spec=SPEC,
        object_logits=torch.tensor([3.0]),
        assign_logits=[
            torch.tensor([[0.0, 0.0]]),    # anchor field, skipped
            torch.tensor([[-1.0, 2.0]]),   # target: ABSENT=-1.0, cand=2.0 -> 3.0
            torch.tensor([[0.0, 1.5]]),    # instrument (list): raw 1.5
        ],
        field_query_ids=[0, 1, 2],
        field_specs=[TRIGGER, TARGET, INSTRUMENT],
        field_spans=[
            torch.tensor([[0, 1]]),
            torch.tensor([[2, 3]]),
            torch.tensor([[4, 5]]),
        ],
        field_cand_mask=[
            torch.tensor([True]), torch.tensor([True]), torch.tensor([True]),
        ],
        field_cand_logits=[torch.tensor([5.0])] * 3,
        instance_seed=[(0, 0)],
        instance_spans=[(0, 1)],
    )


def _mentions():
    """The mention nodes the record spans correspond to, keyed as the adapter keys them."""
    return CandidateScoreSet(
        text="bombed the market with explosives today",
        mentions=(
            MentionScore(0, "trigger", 0, 1, 4.0, 0.98),
            MentionScore(1, "target", 2, 3, 4.0, 0.98),
            MentionScore(2, "instrument", 4, 5, 4.0, 0.98),
        ),
    )


def test_role_edges_link_arguments_to_their_trigger():
    edges = boundary_record_groups_to_role_edges([_group()], QUERY_TYPES)
    assert len(edges) == 2  # anchor field skipped; one target + one instrument

    by_type = {e.relation_type: e for e in edges}
    target = by_type["attack::target"]
    instrument = by_type["attack::instrument"]

    # Both hang off the trigger node, and instance identity is the trigger key.
    assert target.head == ("trigger", 0, 1) == target.hypothesis
    assert instrument.head == ("trigger", 0, 1) == instrument.hypothesis
    assert target.tail == ("target", 2, 3)
    assert instrument.tail == ("instrument", 4, 5)


def test_scalar_role_is_absent_relative_and_slotted():
    """Decision C: scalar utility is logit - ABSENT (softmax row), list is raw (BCE).

    Decision B: only the scalar role takes a slot, so cardinality falls out of
    exclusion_keys without also blocking a list role's second filler.
    """
    edges = {e.relation_type: e for e in
             boundary_record_groups_to_role_edges([_group()], QUERY_TYPES)}

    assert edges["attack::target"].logit == 3.0      # 2.0 - (-1.0)
    assert edges["attack::instrument"].logit == 1.5  # raw

    assert edges["attack::target"].slot == "target"
    assert edges["attack::instrument"].slot is None


def test_role_edges_reference_mention_nodes():
    """The crux: role edges survive into the problem only if keys match the nodes."""
    edges = boundary_record_groups_to_role_edges([_group()], QUERY_TYPES)
    problem = candidate_score_set_to_problem(_mentions(), edges)
    assert len(problem.edges) == 2
    assert {e.slot for e in problem.edges} == {"target", None}
    assert all(e.hypothesis == ("trigger", 0, 1) for e in problem.edges)


def test_miskeyed_query_types_drop_every_role_edge():
    """The regression guard: type role endpoints from a different source than the
    mentions and every edge references a node that does not exist."""
    wrong = ["0", "1", "2"]  # what an empty QueryLayout would yield
    edges = boundary_record_groups_to_role_edges([_group()], wrong)
    assert len(edges) == 2  # edges are still built...
    problem = candidate_score_set_to_problem(_mentions(), edges)
    assert len(problem.edges) == 0  # ...but reference nothing, so all are pruned


def _anchorless_group():
    anchorless = RecordSpec(
        task_index=0, task_name="order", task_type="json_structures",
        mode="anchorless", fields=(TARGET, INSTRUMENT), anchor_query_id=None,
    )
    group = _group()
    object.__setattr__(group, "spec", anchorless)
    return group


def test_anchorless_structures_hang_off_a_synthetic_instance_node():
    """A structure has no trigger, so its roles hang off a synthetic instance node."""
    from gliner2.joint_ie.candidate_scores import boundary_record_instance_nodes

    group = _anchorless_group()
    edges = boundary_record_groups_to_role_edges([group], QUERY_TYPES)
    nodes = boundary_record_instance_nodes([group], QUERY_TYPES)

    key = ("__instance__", "order", 0)
    assert len(nodes) == 1 and nodes[0].candidate_id == key
    assert all(e.head == key and e.hypothesis == key for e in edges)

    # The synthetic node must be in the problem or every role edge is pruned.
    problem = candidate_score_set_to_problem(_mentions(), edges, extra_nodes=nodes)
    assert len(problem.edges) == len(edges) == 2


def test_anchorless_roles_are_pruned_without_the_instance_node():
    """Proves the synthetic node is load-bearing, not decorative."""
    group = _anchorless_group()
    edges = boundary_record_groups_to_role_edges([group], QUERY_TYPES)
    problem = candidate_score_set_to_problem(_mentions(), edges)  # no extra_nodes
    assert len(problem.edges) == 0


def test_latent_groups_are_skipped():
    """`latent` is deferred, not broken -- it keeps using the greedy record path."""
    latent = RecordSpec(
        task_index=0, task_name="deal", task_type="json_structures",
        mode="latent", fields=(TARGET, INSTRUMENT), anchor_query_id=None,
    )
    group = _group()
    object.__setattr__(group, "spec", latent)
    assert boundary_record_groups_to_role_edges([group], QUERY_TYPES) == []
