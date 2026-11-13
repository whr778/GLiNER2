"""Connect a predicted event argument to a predicted coreference cluster by span.

Supports both ``strict`` (exact ``[start, end)`` equality) and ``relaxed`` (exact first,
then character-overlap fallback when unambiguous) matching, as well as configurable
propagation modes (``all``, ``canonical``, ``gated``, ``none``).

Gold ``coref_clusters`` are not an input. They are only the reference for
:func:`score_argument_clusters`, and a document that has none is left out of that
score.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from gliner2.inference.entity_coref import PRONOUNS

Span = Tuple[int, int]
Member = Tuple[str, str, Span]
RoleScoreMap = Dict[Tuple[str, Span, str, Span], float]


def attach_argument_clusters(
    pred: Dict[str, Any],
    *,
    match_mode: str = "strict",
    copy_mode: str = "all",
    min_member_score: float = 0.0,
    role_scores: Optional[RoleScoreMap] = None,
) -> Dict[str, int]:
    """Write ``coref_cluster`` and copy the cluster's mentions onto the event.

    Args:
        pred: Prediction dict with ``event_extraction`` and ``coref_clusters``.
        match_mode: ``"strict"`` (exact ``[start, end)`` match) or ``"relaxed"``
            (exact first, then unambiguous character-overlap fallback).
        copy_mode:
            * ``"all"`` — copy all mentions of the matched cluster onto the role (default).
            * ``"canonical"`` — when the matched argument is a pronoun, copy only the
              longest non-pronoun mention of the cluster.
            * ``"gated"`` — copy only cluster mentions whose ``role_scores`` probability
              meets ``min_member_score``.
            * ``"none"`` — record ``coref_cluster`` on the matched argument without
              appending extra mentions.
        min_member_score: Minimum assignment probability for ``copy_mode="gated"``.
        role_scores: Optional ``{(event_type, trigger_span, role, cand_span): prob}`` map
            from ``RecordHead`` ( defaults to ``pred.get("_role_candidate_scores")``).

    Returns:
        ``{"linked": int, "ambiguous": int, "none": int}`` describing the arguments
        present at the start of the call. A second call does not append those mentions again.
    """
    if match_mode not in ("strict", "relaxed"):
        raise ValueError(f"unknown match_mode {match_mode!r}; expected 'strict' or 'relaxed'")
    if copy_mode not in ("all", "canonical", "gated", "none"):
        raise ValueError(
            f"unknown copy_mode {copy_mode!r}; expected 'all', 'canonical', 'gated', or 'none'"
        )
    counts = {"linked": 0, "ambiguous": 0, "none": 0}
    if not isinstance(pred, dict):
        return counts
    if role_scores is None:
        raw_scores = pred.get("_role_candidate_scores")
        if isinstance(raw_scores, dict):
            role_scores = raw_scores
    clusters = _clusters(pred)
    block = pred.get("event_extraction")
    if not isinstance(block, dict):
        return counts
    for etype, mentions in block.items():
        if not isinstance(mentions, list):
            continue
        for event in mentions:
            if not isinstance(event, dict):
                continue
            arguments = event.get("arguments")
            if not isinstance(arguments, list):
                continue
            trig_span = _first_trigger_span(event.get("triggers"))
            originals = [
                arg for arg in arguments
                if isinstance(arg, dict) and not arg.get("_copied_from_cluster")
            ]
            # Remove any previously copied cluster mentions so a second call is idempotent.
            if len(originals) != len(arguments):
                arguments[:] = originals
            present: Set[Tuple[str, Span]] = set()
            for arg in originals:
                span = _entity_span(arg.get("entity"))
                role = arg.get("role")
                if span is not None and isinstance(role, str) and role.strip():
                    present.add((role.strip(), span))
            for arg in originals:
                arg.pop("coref_cluster", None)
                arg.pop("coref_match", None)
                span = _entity_span(arg.get("entity"))
                hits, matched_how = (
                    _hits_with_mode(span, clusters, match_mode)
                    if span is not None else ([], None)
                )
                if len(hits) == 1:
                    index = hits[0]
                    arg["coref_cluster"] = index
                    if match_mode == "relaxed" and matched_how is not None:
                        arg["coref_match"] = matched_how
                    counts["linked"] += 1
                    role = arg.get("role")
                    if isinstance(role, str) and role.strip() and copy_mode != "none":
                        _copy_members(
                            arguments,
                            role.strip(),
                            clusters[index],
                            index,
                            present,
                            copy_mode=copy_mode,
                            source_surface=_surface(arg.get("entity")),
                            etype=str(etype).strip(),
                            trig_span=trig_span,
                            min_member_score=min_member_score,
                            role_scores=role_scores,
                        )
                elif len(hits) > 1:
                    counts["ambiguous"] += 1
                else:
                    counts["none"] += 1
    return counts


def score_argument_clusters(golds: Sequence[Dict[str, Any]],
                            preds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Strict and relaxed micro precision, recall and F1 of cluster mentions carried as arguments.

    Calls :func:`attach_argument_clusters` on each prediction (strict in-place on ``preds``,
    and relaxed on a copy). A gold argument whose surface sits in exactly one gold cluster
    requires every mention span in that cluster, under that role. Documents with no gold
    clusters are counted and excluded. When every document is excluded, the rates are omitted.
    """
    tp_s = fp_s = fn_s = 0
    tp_r = fp_r = fn_r = 0
    counts_s = {"linked": 0, "ambiguous": 0, "none": 0}
    counts_r = {"linked": 0, "ambiguous": 0, "none": 0}
    gold_documents = ungraded = 0
    for gold, pred in zip(golds, preds):
        pred = pred if isinstance(pred, dict) else {}
        pred_rel = copy.deepcopy(pred)
        got_s = attach_argument_clusters(pred, match_mode="strict")
        got_r = attach_argument_clusters(pred_rel, match_mode="relaxed")
        for key in counts_s:
            counts_s[key] += got_s[key]
            counts_r[key] += got_r[key]
        if not _gold_clusters(gold):
            ungraded += 1
            continue
        gold_documents += 1
        gold_spans, gold_texts = _pack(_gold_clusters(gold))
        required = _gold_members(gold, gold_texts, gold_spans)
        matched_s: Set[Member] = set()
        for member in _pred_members(pred):
            if member in required and member not in matched_s:
                matched_s.add(member)
                tp_s += 1
            else:
                fp_s += 1
        fn_s += len(required - matched_s)

        matched_r: Set[Member] = set()
        for etype, role, p_span in _pred_members(pred_rel):
            hit_gold = next(
                (
                    g_mem for g_mem in required
                    if g_mem not in matched_r
                    and g_mem[0] == etype
                    and g_mem[1] == role
                    and _spans_overlap(p_span, g_mem[2])
                ),
                None,
            )
            if hit_gold is not None:
                matched_r.add(hit_gold)
                tp_r += 1
            else:
                fp_r += 1
        fn_r += len(required - matched_r)

    out: Dict[str, Any] = {
        "eval_argument_cluster_link_support": tp_s + fn_s,
        "eval_argument_cluster_link_linked": counts_s["linked"],
        "eval_argument_cluster_link_ambiguous": counts_s["ambiguous"],
        "eval_argument_cluster_link_none": counts_s["none"],
        "eval_argument_cluster_link_relaxed_linked": counts_r["linked"],
        "eval_argument_cluster_link_relaxed_ambiguous": counts_r["ambiguous"],
        "eval_argument_cluster_link_relaxed_none": counts_r["none"],
        "eval_argument_cluster_link_gold_documents": gold_documents,
        "eval_argument_cluster_link_ungraded_documents": ungraded,
    }
    if gold_documents:
        p_s = tp_s / (tp_s + fp_s) if tp_s + fp_s else 0.0
        r_s = tp_s / (tp_s + fn_s) if tp_s + fn_s else 0.0
        f1_s = 2 * p_s * r_s / (p_s + r_s) if p_s + r_s else 0.0
        out["eval_argument_cluster_link_strict_micro_precision"] = p_s
        out["eval_argument_cluster_link_strict_micro_recall"] = r_s
        out["eval_argument_cluster_link_strict_micro_f1"] = f1_s

        p_r = tp_r / (tp_r + fp_r) if tp_r + fp_r else 0.0
        r_r = tp_r / (tp_r + fn_r) if tp_r + fn_r else 0.0
        f1_r = 2 * p_r * r_r / (p_r + r_r) if p_r + r_r else 0.0
        out["eval_argument_cluster_link_relaxed_micro_precision"] = p_r
        out["eval_argument_cluster_link_relaxed_micro_recall"] = r_r
        out["eval_argument_cluster_link_relaxed_micro_f1"] = f1_r
    return out


def _clusters(pred: Dict[str, Any]) -> List[Any]:
    clusters = pred.get("coref_clusters") if isinstance(pred, dict) else None
    return clusters if isinstance(clusters, list) else []


def _gold_clusters(gold: Dict[str, Any]) -> List[Any]:
    clusters = _clusters(gold if isinstance(gold, dict) else {})
    if any(isinstance(cluster, list) and cluster for cluster in clusters):
        return clusters
    return []


def _first_trigger_span(triggers: Any) -> Optional[Span]:
    if not isinstance(triggers, list):
        return None
    for trig in triggers:
        sp = _span(trig)
        if sp is not None:
            return sp
    return None


def _copy_members(
    arguments: List[Dict[str, Any]],
    role: str,
    cluster: Any,
    index: int,
    present: Set[Tuple[str, Span]],
    *,
    copy_mode: str = "all",
    source_surface: Optional[str] = None,
    etype: str = "",
    trig_span: Optional[Span] = None,
    min_member_score: float = 0.0,
    role_scores: Optional[RoleScoreMap] = None,
) -> None:
    if not isinstance(cluster, list) or copy_mode == "none":
        return
    if copy_mode == "canonical":
        if not source_surface or source_surface.casefold() not in PRONOUNS:
            return
        non_pronouns = [
            m for m in cluster
            if isinstance(m, dict)
            and _span(m) is not None
            and isinstance(m.get("text"), str)
            and m["text"].strip()
            and m["text"].strip().casefold() not in PRONOUNS
        ]
        if not non_pronouns:
            return
        canonical = max(non_pronouns, key=lambda m: (len(m["text"].strip()), -_span(m)[0]))
        candidates = [canonical]
    else:
        candidates = cluster

    for mention in candidates:
        span = _span(mention)
        text = mention.get("text") if isinstance(mention, dict) else None
        if span is None or not isinstance(text, str) or (role, span) in present:
            continue
        if copy_mode == "gated":
            score = (
                role_scores.get((etype, trig_span, role, span), 0.0)
                if (role_scores is not None and trig_span is not None)
                else 0.0
            )
            if score < min_member_score:
                continue
        arguments.append({
            "role": role,
            "entity": {"text": text, "start": span[0], "end": span[1]},
            "coref_cluster": index,
            "_copied_from_cluster": True,
        })
        present.add((role, span))


def _span(mention: Any) -> Optional[Span]:
    if not isinstance(mention, dict):
        return None
    start, end = mention.get("start"), mention.get("end")
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    if isinstance(start, int) and isinstance(end, int):
        return start, end
    return None


def _entity_span(entity: Any) -> Optional[Span]:
    return _span(entity)


def _surface(entity: Any) -> Optional[str]:
    if isinstance(entity, dict):
        text = entity.get("text")
    else:
        text = entity
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _spans_overlap(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _hits(span: Span, clusters: Sequence[Any], match_mode: str = "strict") -> List[int]:
    return _hits_with_mode(span, clusters, match_mode)[0]


def _hits_with_mode(
    span: Span, clusters: Sequence[Any], match_mode: str = "strict"
) -> Tuple[List[int], Optional[str]]:
    exact = []
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, list):
            continue
        if any(_span(mention) == span for mention in cluster):
            exact.append(index)
    if exact or match_mode == "strict":
        return exact, ("strict" if len(exact) == 1 else None)
    relaxed = []
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, list):
            continue
        if any(
            (m_sp := _span(mention)) is not None and _spans_overlap(span, m_sp)
            for mention in cluster
        ):
            relaxed.append(index)
    return relaxed, ("relaxed" if len(relaxed) == 1 else None)


def _pack(clusters: Sequence[Any]) -> Tuple[List[frozenset], List[set]]:
    spans, texts = [], []
    for cluster in clusters:
        found_spans, found_texts = set(), set()
        if isinstance(cluster, list):
            for mention in cluster:
                span = _span(mention)
                if span is not None:
                    found_spans.add(span)
                if isinstance(mention, dict) and isinstance(mention.get("text"), str) and mention["text"].strip():
                    found_texts.add(mention["text"].strip())
        spans.append(frozenset(found_spans))
        texts.append(found_texts)
    return spans, texts


def _gold_members(gold: Dict[str, Any], texts: Sequence[set], spans: Sequence[frozenset]) -> Set[Member]:
    required: Set[Member] = set()
    events = gold.get("events") if isinstance(gold, dict) else None
    if not isinstance(events, list):
        return required
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("event_type"), str):
            continue
        etype = event["event_type"].strip()
        if not etype:
            continue
        for arg in event.get("arguments") or []:
            if not isinstance(arg, dict) or not isinstance(arg.get("role"), str):
                continue
            role, surface = arg["role"].strip(), _surface(arg.get("entity"))
            if not role or not surface:
                continue
            hits = [i for i, names in enumerate(texts) if surface in names]
            if len(hits) == 1:
                for span in spans[hits[0]]:
                    required.add((etype, role, span))
    return required


def _pred_members(pred: Dict[str, Any]) -> Iterable[Member]:
    block = pred.get("event_extraction") if isinstance(pred, dict) else None
    if not isinstance(block, dict):
        return
    for etype, mentions in block.items():
        if not isinstance(etype, str) or not etype.strip() or not isinstance(mentions, list):
            continue
        for event in mentions:
            if not isinstance(event, dict):
                continue
            for arg in event.get("arguments") or []:
                if not isinstance(arg, dict) or not isinstance(arg.get("role"), str):
                    continue
                span = _entity_span(arg.get("entity"))
                index = arg.get("coref_cluster")
                if span is None or isinstance(index, bool) or not isinstance(index, int):
                    continue
                yield etype.strip(), arg["role"].strip(), span
