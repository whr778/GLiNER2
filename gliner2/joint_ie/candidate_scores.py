"""Architecture-neutral sparse candidate score format for joint IE.

The span architecture produces a dense width-oriented :class:`ScoreLattice`;
the boundary architecture produces sparse candidates directly. Both are mapped
onto :class:`CandidateScoreSet` — a flat list of mention scores plus optional
relation-role scores — which then feeds the *unchanged* ``NodeCandidate`` /
``EdgeCandidate`` / ``JointProblem`` optimizer contract. Coordinates are
half-open ``[start, end)`` token offsets throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, List, Mapping, Optional, Sequence, Tuple

from gliner2.joint_ie.candidates import (
    EdgeCandidate,
    JointProblem,
    NodeCandidate,
    center_logit,
    sigmoid,
)


@dataclass(frozen=True)
class MentionScore:
    """One scored span mention (half-open ``[start, end)`` token offsets)."""

    query_id: int
    entity_type: str
    start: int
    end: int
    logit: float
    probability: float

    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.entity_type, self.start, self.end)


@dataclass(frozen=True)
class RelationRoleScore:
    """A mention's compatibility with one role of one relation type."""

    relation_type: str
    role: str  # "head" | "tail"
    mention_id: Hashable
    logit: float
    probability: float


@dataclass
class CandidateScoreSet:
    """Sparse, architecture-neutral candidate scores for one text."""

    text: str
    mentions: Tuple[MentionScore, ...]
    relation_roles: Tuple[RelationRoleScore, ...] = ()
    classifications: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def score_lattice_to_candidate_score_set(lattice: Any) -> CandidateScoreSet:
    """Convert a dense span :class:`ScoreLattice` into a sparse score set.

    Reads the entity task's single count hypothesis (``role_logits[0]`` shaped
    ``[num_types, L, W]``) and emits one :class:`MentionScore` per valid,
    above-floor span cell, mapping inclusive width cells to half-open spans.
    """
    span_starts = lattice.span_starts
    span_ends = lattice.span_ends
    valid = lattice.valid_span_mask

    mentions: List[MentionScore] = []
    query_id = 0
    for task in lattice.tasks:
        if task.task_type != "entities" or not task.count_hypotheses:
            continue
        hyp = task.count_hypotheses[0]
        role_logits = hyp.role_logits[0]           # [num_types, L, W]
        role_probs = hyp.role_probabilities[0]
        num_types = role_logits.shape[0]
        for t in range(num_types):
            entity_type = task.roles[t] if t < len(task.roles) else str(t)
            length = role_logits.shape[1]
            width = role_logits.shape[2]
            for i in range(length):
                for w in range(width):
                    if not bool(valid[i, w]):
                        continue
                    prob = float(role_probs[t, i, w])
                    start = int(span_starts[i, w])
                    end = int(span_ends[i, w]) + 1  # inclusive -> half-open
                    mentions.append(
                        MentionScore(
                            query_id=query_id,
                            entity_type=entity_type,
                            start=start,
                            end=end,
                            logit=float(role_logits[t, i, w]),
                            probability=prob,
                        )
                    )
            query_id += 1

    return CandidateScoreSet(text=lattice.text, mentions=tuple(mentions))


def boundary_candidates_to_candidate_score_set(
    candidates: Any,
    query_types: Sequence[str],
    text: str,
    sample_index: int = 0,
    *,
    pair_temperature: float = 1.0,
) -> CandidateScoreSet:
    """Map one sample's boundary ``CandidateTensorBatch`` to a sparse score set (mentions).

    The boundary architecture produces sparse candidates directly, so this is a flat walk
    over real ``(query, candidate)`` cells -- no lattice. Each candidate becomes a
    :class:`MentionScore` typed by ``query_types[query_id]`` (the schema field/role of that
    query), with the span from ``indices`` and the score from ``pair_logits`` (the
    mention-in-context score the boundary decode itself uses). Relation edges are built
    separately from the relation pair generator -> :class:`ScoredRelationEdge`.
    """
    idx = candidates.indices[sample_index]          # [Q, C, 2]
    pair = candidates.pair_logits[sample_index]     # [Q, C]
    valid = candidates.valid_mask[sample_index]     # [Q, C]
    qmask = candidates.query_mask[sample_index]     # [Q]
    num_queries, num_cands = int(valid.shape[0]), int(valid.shape[1])

    mentions: List[MentionScore] = []
    for q in range(num_queries):
        if not bool(qmask[q]):
            continue
        entity_type = query_types[q] if q < len(query_types) else str(q)
        for c in range(num_cands):
            if not bool(valid[q, c]):
                continue
            logit = float(pair[q, c]) / pair_temperature
            mentions.append(
                MentionScore(
                    query_id=q,
                    entity_type=entity_type,
                    start=int(idx[q, c, 0]),
                    end=int(idx[q, c, 1]),
                    logit=logit,
                    probability=sigmoid(logit),
                )
            )
    return CandidateScoreSet(text=text, mentions=tuple(mentions))


@dataclass(frozen=True)
class ScoredRelationEdge:
    """A scored (head, tail) relation proposal referencing mention keys.

    ``slot``/``hypothesis`` carry record semantics: a *role edge* of an event
    instance sets ``hypothesis`` to its trigger node key and, for a scalar role,
    ``slot`` to the role name -- which is what makes scalar cardinality fall out of
    the optimizer's ``exclusion_keys`` for free. Plain relations leave both ``None``.
    """

    relation_type: str
    head: Hashable
    tail: Hashable
    logit: float
    probability: float
    slot: Optional[Hashable] = None
    hypothesis: Optional[Hashable] = None


def boundary_relation_pairs_to_edges(
    pairs: Any,
    logits: Sequence[float],
    *,
    relation_temperature: float = 1.0,
) -> List["ScoredRelationEdge"]:
    """Map a boundary ``RelationPairBatch`` + per-pair relation logits to scored edges.

    Each proposed pair already records ``head_keys`` / ``tail_keys`` as
    ``(query role_name, start, end)`` — the *same* key `boundary_candidates_to_candidate_
    score_set` uses for its mentions — so the edges reference the mention nodes directly.
    ``head_keys`` / ``tail_keys`` are populated on the decode-time (compact) pair batch.
    """
    edges: List[ScoredRelationEdge] = []
    for i in range(len(pairs)):
        logit = float(logits[i]) / relation_temperature
        edges.append(
            ScoredRelationEdge(
                relation_type=pairs.relation_types[i],
                head=tuple(pairs.head_keys[i]),
                tail=tuple(pairs.tail_keys[i]),
                logit=logit,
                probability=sigmoid(logit),
            )
        )
    return edges


def boundary_record_groups_to_role_edges(
    groups: Sequence[Any],
    query_types: Sequence[str],
) -> List["ScoredRelationEdge"]:
    """Map boundary ``RecordGroupOutput``s to event **role edges** (design §3b).

    An event instance *is* its trigger node, so a record needs no new candidate class:
    each (instance, role, filler) assignment becomes an edge from the trigger node to
    the argument node, both keyed ``(role_name, start, end)`` from the same layout the
    mention adapter uses -- so role edges reference mention nodes by construction.

    Reads the *raw* lattice (``object_logits``/``assign_logits``/``field_spans``), never
    ``decode_group`` output, so the beam sees the scores rather than re-ranking already
    greedy decisions.

    Utilities follow the record head's own loss shapes: a **scalar** field row is a
    ``log_softmax`` over ``{ABSENT, candidates}``, so its utility is the ABSENT-relative
    log-odds ``logit_c - logit_ABSENT``; a **list** field is BCE over candidates alone, so
    its raw logit is used and centers like a relation. (Both are already threshold-0.5
    calibrated; a non-default ``decision_threshold`` would shift scalar roles, which is
    why the engine leaves it at the default.)

    Only ``natural`` (anchored/trigger) groups are emitted. ``anchorless`` structures need
    a synthetic instance node and ``latent`` is deferred -- both keep working through the
    greedy record path.
    """
    edges: List[ScoredRelationEdge] = []
    for group in groups:
        spec = group.spec
        if spec.mode != "natural" or spec.anchor_query_id is None:
            continue
        anchor_type = _query_type_name(query_types, spec.anchor_query_id)
        for inst in range(group.num_instances):
            anchor_span = group.instance_spans[inst]
            if anchor_span is None:
                continue
            trigger = (anchor_type, int(anchor_span[0]), int(anchor_span[1]))
            for f, fspec in enumerate(group.field_specs):
                if fspec.is_anchor:
                    continue  # the trigger is not its own argument
                role = _query_type_name(query_types, group.field_query_ids[f])
                row = group.assign_logits[f][inst]
                absent = float(row[0])
                spans = group.field_spans[f]
                mask = group.field_cand_mask[f]
                for c in range(int(spans.shape[0])):
                    if not bool(mask[c]):
                        continue
                    logit = float(row[1 + c])
                    utility = logit - absent if fspec.is_scalar else logit
                    edges.append(ScoredRelationEdge(
                        relation_type=f"{spec.task_name}::{role}",
                        head=trigger,
                        tail=(role, int(spans[c][0]), int(spans[c][1])),
                        logit=utility,
                        probability=sigmoid(utility),
                        slot=role if fspec.is_scalar else None,
                        hypothesis=trigger,
                    ))
    return edges


def _query_type_name(query_types: Sequence[str], query_id: int) -> str:
    qid = int(query_id)
    return query_types[qid] if qid < len(query_types) else str(qid)


def candidate_score_set_to_problem(
    score_set: CandidateScoreSet,
    edges: Sequence[ScoredRelationEdge] = (),
    *,
    mention_threshold: float = 0.5,
    constraints: Sequence[Any] = (),
    decision_threshold: float = 0.5,
) -> JointProblem:
    """Build a :class:`JointProblem` from sparse mention + edge scores.

    Node/edge utilities are centered log-odds (positive => above threshold), so
    the existing greedy/beam optimizers and constraints work unchanged.
    """
    nodes: List[NodeCandidate] = []
    keep_ids = set()
    for m in score_set.mentions:
        if m.probability < mention_threshold:
            continue
        node = NodeCandidate(
            entity_type=m.entity_type,
            start=m.start,
            end=m.end,
            score=center_logit(m.logit, decision_threshold),
            probability=m.probability,
            candidate_id=m.key,
        )
        nodes.append(node)
        keep_ids.add(m.key)

    edge_cands: List[EdgeCandidate] = []
    for e in edges:
        if e.head not in keep_ids or e.tail not in keep_ids:
            continue
        edge_cands.append(
            EdgeCandidate(
                relation_type=e.relation_type,
                head=e.head,
                tail=e.tail,
                score=center_logit(e.logit, decision_threshold),
                head_probability=e.probability,
                tail_probability=e.probability,
                slot=e.slot,
                hypothesis=e.hypothesis,
            )
        )

    return JointProblem(
        nodes=tuple(nodes),
        edges=tuple(edge_cands),
        constraints=tuple(constraints),
    )


def joint_decode(
    candidates: Any,
    query_types: Sequence[str],
    pairs: Any,
    relation_logits: Sequence[float],
    *,
    constraints: Sequence[Any] = (),
    sample_index: int = 0,
    text: str = "",
    mention_threshold: float = 0.5,
    beam_width: int = 16,
    pair_temperature: float = 1.0,
    relation_temperature: float = 1.0,
):
    """End-to-end boundary joint decode: candidates + relation pairs → mentions + edges →
    typed-constraint beam → the selected node/edge solution. Composes the two boundary
    adapters with the shared `BeamOptimizer`; this is the joint_ie side of the boundary
    ``--joint-decode`` wiring. The engine caller converts the solution's token spans to
    char offsets and formats the output dict (the remaining `BoundaryExtractor` piece)."""
    from gliner2.joint_ie.optimizers import BeamOptimizer

    css = boundary_candidates_to_candidate_score_set(
        candidates, query_types, text, sample_index, pair_temperature=pair_temperature)
    edges = boundary_relation_pairs_to_edges(
        pairs, relation_logits, relation_temperature=relation_temperature)
    problem = candidate_score_set_to_problem(
        css, edges, mention_threshold=mention_threshold, constraints=constraints)
    return BeamOptimizer(beam_width=beam_width).optimize(problem)


__all__ = [
    "MentionScore",
    "RelationRoleScore",
    "CandidateScoreSet",
    "ScoredRelationEdge",
    "score_lattice_to_candidate_score_set",
    "boundary_candidates_to_candidate_score_set",
    "boundary_relation_pairs_to_edges",
    "boundary_record_groups_to_role_edges",
    "candidate_score_set_to_problem",
    "joint_decode",
]
