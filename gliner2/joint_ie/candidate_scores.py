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
    CandidateSource,
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
    # None, not 0.5 -- same reason as ScoredRelationEdge.threshold. The scoring guard
    # reads `m.threshold if m.threshold is not None else decision_threshold`, so a 0.5
    # default pins NODE utilities to 0.5 and the caller's threshold never reaches node
    # admission. Upstream's own converter passes `threshold=` explicitly, so an explicit
    # per-mention value still wins.
    threshold: Optional[float] = None
    candidate_threshold: Optional[float] = None

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


@dataclass(frozen=True)
class ScoredRelationEdge:
    """A scored (head, tail) relation proposal referencing mention keys.

    ``slot``/``hypothesis`` carry record semantics: a *role edge* of an event instance
    sets ``hypothesis`` to its trigger node key and, for a scalar role, ``slot`` to the
    role name -- which is what makes scalar cardinality fall out of the optimizer's
    ``exclusion_keys`` for free. Plain relations leave both ``None``.
    """

    relation_type: str
    head: Hashable
    tail: Hashable
    logit: float
    probability: float
    # None, not 0.5. Upstream's own scoring guards `threshold is not None` and falls back
    # to the caller's `decision_threshold`, but a 0.5 DEFAULT makes that guard always take
    # the edge's value -- which silently pins edge selection to 0.5 and stops the decode
    # responding to --threshold at all. That regression is measured, not hypothetical:
    # joint recall moved only 0.1498 -> 0.1591 across thresholds 0.5 -> 0.1 on Re-DocRED
    # while the greedy arm moved 0.0461 -> 0.4134. An explicit per-edge threshold still wins.
    threshold: Optional[float] = None
    candidate_threshold: Optional[float] = None
    slot: Optional[Hashable] = None
    hypothesis: Optional[Hashable] = None


@dataclass
class CandidateScoreSet:
    """Sparse, architecture-neutral candidate scores for one text."""

    text: str
    mentions: Tuple[MentionScore, ...]
    relation_roles: Tuple[RelationRoleScore, ...] = ()
    edges: Tuple[ScoredRelationEdge, ...] = ()
    classifications: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    text_tokens: Tuple[str, ...] = ()
    start_mappings: Tuple[int, ...] = ()
    end_mappings: Tuple[int, ...] = ()


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

    return CandidateScoreSet(
        text=lattice.text,
        mentions=tuple(mentions),
        text_tokens=tuple(getattr(lattice, "text_tokens", ())),
        start_mappings=tuple(getattr(lattice, "start_mappings", ())),
        end_mappings=tuple(getattr(lattice, "end_mappings", ())),
    )


def boundary_candidates_to_candidate_score_set(
    text: str,
    candidates: Any,
    query_specs: Sequence[Any],
    *,
    sample_index: int = 0,
    token_offset: int = 0,
    text_length: Optional[int] = None,
    pair_temperature: float = 1.0,
    entity_thresholds: Optional[Mapping[str, Optional[float]]] = None,
    entity_candidate_thresholds: Optional[
        Mapping[str, Optional[float]]
    ] = None,
    extra_mentions: Sequence[MentionScore] = (),
    edges: Sequence[ScoredRelationEdge] = (),
    text_tokens: Sequence[str] = (),
    start_mappings: Sequence[int] = (),
    end_mappings: Sequence[int] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> CandidateScoreSet:
    """Convert one boundary candidate batch row into sparse joint scores."""
    if pair_temperature <= 0:
        raise ValueError("pair_temperature must be positive")
    if text_length is None:
        text_length = len(start_mappings)
    thresholds = dict(entity_thresholds or {})
    candidate_thresholds = dict(entity_candidate_thresholds or {})
    best: dict[Tuple[str, int, int], MentionScore] = {
        mention.key: mention for mention in extra_mentions
    }

    for query_id, spec in enumerate(query_specs):
        task_type = (
            spec.get("task_type")
            if isinstance(spec, Mapping)
            else getattr(spec, "task_type", None)
        )
        if task_type != "entities" or query_id >= candidates.indices.shape[1]:
            continue
        entity_type = str(
            spec.get("field_name")
            if isinstance(spec, Mapping)
            else getattr(spec, "role_name", query_id)
        )
        threshold = thresholds.get(entity_type)
        threshold = 0.5 if threshold is None else float(threshold)
        valid = (
            candidates.valid_mask[sample_index, query_id]
            & candidates.query_mask[sample_index, query_id]
        )
        candidate_ids = valid.nonzero(as_tuple=False).flatten().tolist()
        for candidate_id in candidate_ids:
            start = int(
                candidates.indices[sample_index, query_id, candidate_id, 0]
            ) - token_offset
            end = int(
                candidates.indices[sample_index, query_id, candidate_id, 1]
            ) - token_offset
            if not (0 <= start < end <= int(text_length)):
                continue
            logit = float(
                candidates.pair_logits[
                    sample_index, query_id, candidate_id
                ].detach().float()
            ) / pair_temperature
            mention = MentionScore(
                query_id=query_id,
                entity_type=entity_type,
                start=start,
                end=end,
                logit=logit,
                probability=sigmoid(logit),
                threshold=threshold,
                candidate_threshold=candidate_thresholds.get(entity_type),
            )
            previous = best.get(mention.key)
            if previous is None or mention.logit > previous.logit:
                best[mention.key] = mention

    mentions = tuple(sorted(
        best.values(),
        key=lambda item: (
            item.entity_type, item.start, item.end, -item.logit
        ),
    ))
    return CandidateScoreSet(
        text=text,
        mentions=mentions,
        edges=tuple(edges),
        metadata=dict(metadata or {}),
        text_tokens=tuple(text_tokens),
        start_mappings=tuple(start_mappings),
        end_mappings=tuple(end_mappings),
    )


def candidate_score_set_to_problem(
    score_set: CandidateScoreSet,
    edges: Optional[Sequence[ScoredRelationEdge]] = None,
    *,
    mention_threshold: float = 0.5,
    constraints: Sequence[Any] = (),
    decision_threshold: float = 0.5,
    extra_nodes: Sequence[NodeCandidate] = (),
    pre_scored_edges: Sequence["ScoredRelationEdge"] = (),
    max_mentions_per_type: Optional[int] = None,
    max_mentions_by_type: Optional[Mapping[str, int]] = None,
    rescue_relation_endpoints: bool = False,
    edge_candidate_threshold: float = 0.0,
    max_edges_per_type: Optional[int] = None,
    entity_weight: float = 1.0,
    relation_weight: float = 1.0,
) -> JointProblem:
    """Build a :class:`JointProblem` from sparse mention + edge scores.

    Node/edge utilities are centered log-odds (positive => above threshold), so
    the existing greedy/beam optimizers and constraints work unchanged.

    ``decision_threshold`` is what makes the optimizers threshold-aware: it sets where
    utility crosses zero, and the optimizers only ever take a node or edge whose utility
    is positive. Leaving it at 0.5 while the caller asked for 0.1 is not a mild
    miscalibration -- it silently pins edge selection to 0.5 and the decode stops
    responding to the threshold at all.

    ``pre_scored_edges`` carry utilities that are **already** on the right scale and must
    not be re-centered: a record role edge's scalar utility is the ABSENT-relative
    log-odds ``logit_c - logit_ABSENT``, which is a comparison against the head's own
    ABSENT class rather than against a probability cutoff. Shifting it by a threshold
    offset would move scalar roles against a baseline that does not exist for them.
    ``extra_nodes`` bypass for the same reason.
    """
    raw_edges = tuple(score_set.edges if edges is None else edges)
    edge_by_key: dict[Tuple[Any, ...], ScoredRelationEdge] = {}
    for edge in raw_edges:
        threshold = (
            edge_candidate_threshold
            if edge.candidate_threshold is None
            else edge.candidate_threshold
        )
        if edge.probability < threshold:
            continue
        key = (edge.relation_type, edge.head, edge.tail)
        previous = edge_by_key.get(key)
        if previous is None or edge.logit > previous.logit:
            edge_by_key[key] = edge
    edge_counts: dict[str, int] = {}
    retained_edges: List[ScoredRelationEdge] = []
    for edge in sorted(
        edge_by_key.values(),
        key=lambda item: (
            item.relation_type, -item.logit, str(item.head), str(item.tail)
        ),
    ):
        if (
            max_edges_per_type is not None
            and edge_counts.get(edge.relation_type, 0) >= max_edges_per_type
        ):
            continue
        retained_edges.append(edge)
        edge_counts[edge.relation_type] = (
            edge_counts.get(edge.relation_type, 0) + 1
        )
    relation_edges = tuple(retained_edges)
    rescue_ids = {
        endpoint
        for edge in relation_edges
        for endpoint in (edge.head, edge.tail)
    } if rescue_relation_endpoints else set()
    selected_mentions: List[MentionScore] = []
    per_type: dict[str, int] = {}
    type_limits = dict(max_mentions_by_type or {})
    for m in sorted(
        score_set.mentions,
        key=lambda item: (
            item.entity_type, -item.probability, item.start, item.end
        ),
    ):
        candidate_threshold = (
            mention_threshold
            if m.candidate_threshold is None
            else m.candidate_threshold
        )
        if m.probability < candidate_threshold and m.key not in rescue_ids:
            continue
        type_limit = type_limits.get(m.entity_type, max_mentions_per_type)
        if (
            type_limit is not None
            and per_type.get(m.entity_type, 0) >= type_limit
            and m.key not in rescue_ids
        ):
            continue
        selected_mentions.append(m)
        per_type[m.entity_type] = per_type.get(m.entity_type, 0) + 1

    nodes: List[NodeCandidate] = []
    keep_ids = set()
    for m in selected_mentions:
        candidate_threshold = (
            mention_threshold
            if m.candidate_threshold is None
            else m.candidate_threshold
        )
        node = NodeCandidate(
            entity_type=m.entity_type,
            start=m.start,
            end=m.end,
            score=entity_weight * center_logit(
                m.logit,
                m.threshold if m.threshold is not None else decision_threshold,
            ),
            probability=m.probability,
            source=(
                CandidateSource.RELATION_RESCUE
                if (
                    m.key in rescue_ids
                    and m.probability < candidate_threshold
                )
                else CandidateSource.ENTITY
            ),
            candidate_id=m.key,
        )
        nodes.append(node)
        keep_ids.add(m.key)

    for node in extra_nodes:  # synthetic record-instance nodes; not thresholded
        nodes.append(node)
        keep_ids.add(node.candidate_id)

    edge_cands: List[EdgeCandidate] = []
    for edge_slot, (e, recenter) in enumerate(
            [(edge, True) for edge in relation_edges]
            + [(edge, False) for edge in pre_scored_edges]):
        if e.head not in keep_ids or e.tail not in keep_ids:
            continue
        edge_cands.append(
            EdgeCandidate(
                relation_type=e.relation_type,
                head=e.head,
                tail=e.tail,
                score=(
                    relation_weight * center_logit(
                        e.logit,
                        e.threshold if getattr(e, "threshold", None) is not None
                        else decision_threshold,
                    )
                ) if recenter else e.logit,
                head_probability=e.probability,
                tail_probability=e.probability,
                # Our record edges carry slot/hypothesis (trigger key + role name),
                # which is what makes scalar cardinality fall out of exclusion_keys.
                # Plain relation edges have neither, and fall back to upstream's
                # per-edge slot index and relation type.
                # OURS, deliberately, not upstream's per-edge index. `slot is None`
                # means NO exclusion keys (candidates.py), and both optimizers tie-break
                # on `str(edge.slot)` -- so filling plain edges with 0,1,2 both invents
                # exclusion groups and reorders the beam, silently changing joint decode
                # output. Upstream needs distinct slots only for an opt-in
                # UniqueRelationSlot constraint we never pass.
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
    extra_edges: Sequence["ScoredRelationEdge"] = (),
    extra_nodes: Sequence[NodeCandidate] = (),
    decision_threshold: float = 0.5,
):
    """End-to-end boundary joint decode: candidates + relation pairs → mentions + edges →
    typed-constraint beam → the selected node/edge solution. Composes the two boundary
    adapters with the shared `BeamOptimizer`; this is the joint_ie side of the boundary
    ``--joint-decode`` wiring. The engine caller converts the solution's token spans to
    char offsets and formats the output dict (the remaining `BoundaryExtractor` piece)."""
    from gliner2.joint_ie.optimizers import BeamOptimizer

    css = boundary_candidates_to_scores(
        candidates, query_types, text, sample_index, pair_temperature=pair_temperature)
    edges = boundary_relation_pairs_to_edges(
        pairs, relation_logits, relation_temperature=relation_temperature)
    # Record role edges (sec 3b) go in pre-scored: their utilities are ABSENT-relative
    # and must not be re-centered on the caller's threshold.
    problem = candidate_score_set_to_problem(
        css, edges, mention_threshold=mention_threshold, constraints=constraints,
        decision_threshold=decision_threshold, extra_nodes=extra_nodes,
        pre_scored_edges=extra_edges)
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
    *,
    instance_threshold: float = 0.5,
    temperature: float = 1.0,
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

    **Instance-existence gate (decision D, revised 2026-08-08).** An instance is skipped
    unless ``sigmoid(object_logits[inst] / temperature) >= instance_threshold`` -- the same
    test ``decode_group`` applies before emitting a record. Without it the joint arm has no
    existence gate at all and emits events surviving on a single positive role edge that
    greedy would have rejected outright, biasing the decode-arm comparison against the beam.

    ``latent`` is deferred -- it keeps working through the greedy record path.
    """
    edges: List[ScoredRelationEdge] = []
    for group in groups:
        spec = group.spec
        if spec.mode == "latent":
            continue
        for inst in range(group.num_instances):
            if sigmoid(float(group.object_logits[inst]) / temperature) < instance_threshold:
                continue  # greedy would not emit this instance either
            trigger = _instance_key(group, inst, query_types)
            if trigger is None:
                continue
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


def _instance_key(group: Any, inst: int, query_types: Sequence[str]):
    """Identity of one record instance, or ``None`` if it has none.

    ``natural`` groups are identified by their **trigger span**, which is a real
    mention node. ``anchorless`` structures have no anchor span, so they take a
    synthetic key instead -- see :func:`boundary_record_instance_nodes`.
    """
    spec = group.spec
    if spec.mode == "anchorless":
        return ("__instance__", spec.task_name, inst)
    if spec.anchor_query_id is None:
        return None
    span = group.instance_spans[inst]
    if span is None:
        return None
    return (_query_type_name(query_types, spec.anchor_query_id),
            int(span[0]), int(span[1]))


def boundary_record_instance_nodes(
    groups: Sequence[Any],
    query_types: Sequence[str],
    *,
    decision_threshold: float = 0.5,
    instance_threshold: float = 0.5,
    temperature: float = 1.0,
) -> List[NodeCandidate]:
    """Synthetic instance nodes for ``anchorless`` structures (design §3b).

    A ``natural`` event is identified by its trigger, which is already a mention
    node. An ``anchorless`` structure has no anchor span, so its role edges need
    *some* node to hang off: one synthetic node per instance, scored by the record
    head's own ``object_logits`` (this is where decision D's per-instance existence
    signal is genuinely needed -- there is no trigger mention to carry it).

    The span is positional filler, never emitted: these nodes are dropped from the
    entity output by type and consumed only as role-edge heads. **Caveat:** under a
    non-``allow`` :class:`EntityOverlapPolicy` a synthetic span participates in
    overlap checks against real mentions. The boundary engine never sets that
    policy on the joint path, so this is recorded rather than engineered around.

    Gated by the same instance-existence test as the role edges, so a below-threshold
    instance contributes neither a node nor edges.
    """
    nodes: List[NodeCandidate] = []
    for group in groups:
        if group.spec.mode != "anchorless":
            continue
        for inst in range(group.num_instances):
            logit = float(group.object_logits[inst])
            if sigmoid(logit / temperature) < instance_threshold:
                continue
            nodes.append(NodeCandidate(
                entity_type=f"__{group.spec.task_name}__",
                start=inst, end=inst + 1,
                score=center_logit(logit, decision_threshold),
                probability=sigmoid(logit),
                candidate_id=("__instance__", group.spec.task_name, inst),
            ))
    return nodes

# Our converter. Upstream defines a function of the SAME original name with a
# different contract -- theirs is (text, candidates, query_specs) with per-type
# thresholds; ours is (candidates, query_types, text) typing mentions from a flat
# sequence of type names, which is what the joint decode is built against. Renamed
# rather than merged: both are live, and collapsing them silently breaks one.
def boundary_candidates_to_scores(
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
