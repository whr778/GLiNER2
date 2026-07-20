"""Global graph decoding over windowed event candidates (OneIE-style).

Model-free post-processing over the per-chunk results produced by the long-doc
path (``batch_extract_long`` -> ``merge_chunk_results``). Overlapping windows
re-detect the same event and split its arguments across chunks; this reconnects
them: cluster event mentions across windows by trigger overlap, union their
arguments, and emit one document-level event per cluster in the *exact* normal
``event_extraction`` shape so downstream formatting and metrics are unchanged.

Increment 1 (this module today) is the greedy assembler. The beam layer with
global constraints (role validity, cardinality, span-conflict) lands on top of
``assemble_events_global`` in a later increment.

See ``tools/events_working_papers/DOCUMENT_EXTRACTION_PLAN.md``.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from gliner2.inference.chunking import _dedupe_items, _is_span_dict, _span_key


@dataclass(frozen=True)
class GlobalDecodeConfig:
    """Knobs for the global decoder.

    - ``trigger_iou``: minimum span IoU for two mentions' triggers to be the
      same event across windows (greedy clustering).
    - ``beam_width``/``conflict_penalty``/``min_trigger_conf``/
      ``single_filler_roles``: the beam layer's global constraints.
    """

    trigger_iou: float = 0.5
    beam_width: int = 8
    conflict_penalty: float = 0.5
    min_trigger_conf: float = 0.0
    single_filler_roles: frozenset = field(default_factory=frozenset)


def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    if inter <= 0:
        return 0.0
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def _trigger_spans(mention: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [t for t in mention.get("triggers") or [] if _is_span_dict(t)]


def _same_event(a: Dict[str, Any], b: Dict[str, Any], trigger_iou: float) -> bool:
    """Two mentions are the same event if any of their trigger spans overlap by
    at least ``trigger_iou`` (identical global spans from overlapping windows
    score 1.0)."""
    return any(
        _iou(ta, tb) >= trigger_iou
        for ta in _trigger_spans(a)
        for tb in _trigger_spans(b)
    )


def _cluster_mentions(mentions: List[Dict[str, Any]], trigger_iou: float) -> List[List[int]]:
    """Union-find over mentions by trigger overlap; returns index clusters."""
    n = len(mentions)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if _same_event(mentions[i], mentions[j], trigger_iou):
                parent[find(i)] = find(j)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return list(clusters.values())


def _merge_cluster(
    mentions: List[Dict[str, Any]], valid_roles: Optional[set],
) -> Dict[str, Any]:
    """Union the triggers and arguments of one event's mentions into a single
    mention dict. Duplicate trigger/argument spans (same event seen in several
    overlapping windows) collapse via ``_dedupe_items``; the highest-confidence
    copy is kept."""
    triggers: List[Any] = []
    for m in mentions:
        triggers.extend(_trigger_spans(m))
    triggers = _dedupe_items(triggers, remove_overlaps=True)

    by_role: "OrderedDict[str, List[Any]]" = OrderedDict()
    for m in mentions:
        for arg in m.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            role, entity = arg.get("role"), arg.get("entity")
            if not isinstance(role, str) or entity is None:
                continue
            if valid_roles is not None and role not in valid_roles:
                continue
            by_role.setdefault(role, []).append(entity)

    arguments: List[Dict[str, Any]] = []
    for role, entities in by_role.items():
        spans = [e for e in entities if _is_span_dict(e)]
        for entity in _dedupe_items(spans, remove_overlaps=True):
            arguments.append({"role": role, "entity": entity})

    return {"triggers": triggers, "arguments": arguments}


def _greedy_assemble(
    remapped_results: List[Dict[str, Any]],
    event_roles: Optional[Dict[str, List[str]]],
    cfg: GlobalDecodeConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """Cluster mentions across windows by trigger overlap and union their
    arguments (the greedy substrate the beam refines)."""
    by_type: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for res in remapped_results:
        block = res.get("event_extraction")
        if not isinstance(block, dict):
            continue
        for etype, mentions in block.items():
            bucket = by_type.setdefault(etype, [])
            if isinstance(mentions, list):
                bucket.extend(m for m in mentions if isinstance(m, dict))

    out: Dict[str, List[Dict[str, Any]]] = {}
    for etype, mentions in by_type.items():
        valid_roles = set(event_roles[etype]) if event_roles and etype in event_roles else None
        if not mentions:
            out[etype] = []
            continue
        out[etype] = [
            _merge_cluster([mentions[i] for i in idxs], valid_roles)
            for idxs in _cluster_mentions(mentions, cfg.trigger_iou)
        ]
    return out


def _trigger_conf(mention: Dict[str, Any]) -> float:
    return max((t.get("confidence", 1.0) for t in _trigger_spans(mention)), default=1.0)


def beam_decode(
    assembled: Dict[str, List[Dict[str, Any]]], cfg: GlobalDecodeConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """Refine greedily-assembled events under global constraints via beam search.

    Drops events below ``min_trigger_conf``, then chooses which argument edges
    to keep to maximize summed confidence minus ``conflict_penalty`` for every
    span reused across kept edges, subject to single-filler cardinality. Triggers
    pass through unchanged. With no conflicts, no single-filler roles, and a zero
    trigger floor, every edge is kept and the output equals the greedy input.
    """
    events: List[Tuple[str, List[Any], List[Tuple[str, Any, float]]]] = []
    for etype, mentions in assembled.items():
        for m in mentions:
            if _trigger_conf(m) < cfg.min_trigger_conf:
                continue
            edges = [
                (a["role"], a["entity"],
                 a["entity"].get("confidence", 1.0) if _is_span_dict(a["entity"]) else 1.0)
                for a in m.get("arguments") or []
                if isinstance(a, dict) and a.get("entity") is not None
            ]
            events.append((etype, m.get("triggers") or [], edges))

    # Flatten to global argument edges, highest confidence first.
    edges: List[Tuple[int, str, Any, float, Optional[Tuple]]] = []
    for ei, (_etype, _trigs, cand) in enumerate(events):
        for role, entity, conf in cand:
            sk = _span_key(entity) if _is_span_dict(entity) else None
            edges.append((ei, role, entity, conf, sk))
    edges.sort(key=lambda e: -e[3])

    # Beam state: (score, kept-edge-index tuple, span-use counts, filled single roles).
    beams: List[Tuple[float, Tuple[int, ...], Dict[Any, int], frozenset]] = [
        (0.0, (), {}, frozenset())
    ]
    for idx, (ei, role, _entity, conf, sk) in enumerate(edges):
        nxt = []
        for score, kept, used, filled in beams:
            nxt.append((score, kept, used, filled))  # drop this edge
            if role in cfg.single_filler_roles and (ei, role) in filled:
                continue  # cardinality: role already filled
            penalty = cfg.conflict_penalty if (sk is not None and used.get(sk, 0) >= 1) else 0.0
            new_used = dict(used)
            if sk is not None:
                new_used[sk] = new_used.get(sk, 0) + 1
            new_filled = filled | {(ei, role)} if role in cfg.single_filler_roles else filled
            nxt.append((score + conf - penalty, kept + (idx,), new_used, new_filled))
        beams = sorted(nxt, key=lambda s: -s[0])[: cfg.beam_width]

    kept_idx = set(beams[0][1])
    out: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for ei, (etype, trigs, _cand) in enumerate(events):
        args = [
            {"role": edges[i][1], "entity": edges[i][2]}
            for i in kept_idx if edges[i][0] == ei
        ]
        out.setdefault(etype, []).append({"triggers": trigs, "arguments": args})
    # Preserve requested-but-empty event types dropped by the trigger floor.
    for etype in assembled:
        out.setdefault(etype, [])
    return dict(out)


def assemble_events_global(
    remapped_results: List[Dict[str, Any]],
    event_roles: Optional[Dict[str, List[str]]] = None,
    cfg: GlobalDecodeConfig = GlobalDecodeConfig(),
    beam: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Assemble one document's ``event_extraction`` block from its per-chunk
    results (already remapped to global offsets, spans carrying confidence).

    Greedily clusters mentions across windows and unions their arguments, then
    (``beam=True``) refines the result under global constraints. Returns
    ``{event_type: [mention_dict]}`` in the normal output shape; requested types
    with no mentions keep an empty list.
    """
    assembled = _greedy_assemble(remapped_results, event_roles, cfg)
    if not assembled:
        return {}
    return beam_decode(assembled, cfg) if beam else assembled
