"""Connect a predicted event argument to a predicted coreference cluster by exact span.

The mention link and the argument head stay separate. This is the decode-time join:
an argument whose ``[start, end)`` equals a mention in exactly one cluster records
that cluster's index. No cluster, an empty list, a string argument, or a span that
meets zero clusters or two or more leaves the field off. Other mentions are not
copied onto the event.

Gold ``coref_clusters`` are not an input. They are only the reference for
:func:`score_argument_clusters`, and a document that has none is left out of that
score.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Span = Tuple[int, int]
Link = Tuple[str, str, str, frozenset]


def attach_argument_clusters(pred: Dict[str, Any]) -> Dict[str, int]:
    """Write ``coref_cluster`` on uniquely matched arguments. Return linked, ambiguous, none.

    Idempotent: a previous index is cleared before the span test runs again.
    """
    counts = {"linked": 0, "ambiguous": 0, "none": 0}
    if not isinstance(pred, dict):
        return counts
    clusters = _clusters(pred)
    for arg in _pred_arguments(pred):
        arg.pop("coref_cluster", None)
        span = _entity_span(arg.get("entity"))
        hits = _hits(span, clusters) if span is not None else []
        if len(hits) == 1:
            arg["coref_cluster"] = hits[0]
            counts["linked"] += 1
        elif len(hits) > 1:
            counts["ambiguous"] += 1
        else:
            counts["none"] += 1
    return counts


def score_argument_clusters(golds: Sequence[Dict[str, Any]],
                            preds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Strict micro precision, recall and F1 of argument-to-cluster links.

    Calls :func:`attach_argument_clusters` on each prediction. A gold link is one
    argument string that sits in exactly one gold cluster. A predicted link matches
    it when the event type, role and surface agree and the two clusters share a
    mention span. Documents with no gold clusters are counted and excluded. When
    every document is excluded, the three rates are omitted.
    """
    tp = fp = fn = 0
    counts = {"linked": 0, "ambiguous": 0, "none": 0}
    gold_documents = ungraded = 0
    for gold, pred in zip(golds, preds):
        pred = pred if isinstance(pred, dict) else {}
        got = attach_argument_clusters(pred)
        for key in counts:
            counts[key] += got[key]
        if not _gold_clusters(gold):
            ungraded += 1
            continue
        gold_documents += 1
        gold_spans, gold_texts = _pack(_gold_clusters(gold))
        pred_spans, _ = _pack(_clusters(pred))
        gold_links = list(_gold_links(gold, gold_texts, gold_spans))
        pred_links = list(_pred_links(pred, pred_spans))
        used = [False] * len(pred_links)
        for etype, role, surface, spans in gold_links:
            match = _take(pred_links, used, etype, role, surface, spans)
            if match:
                tp += 1
            else:
                fn += 1
        fp += sum(1 for flag in used if not flag)
    out: Dict[str, Any] = {
        "eval_argument_cluster_link_support": tp + fn,
        "eval_argument_cluster_link_linked": counts["linked"],
        "eval_argument_cluster_link_ambiguous": counts["ambiguous"],
        "eval_argument_cluster_link_none": counts["none"],
        "eval_argument_cluster_link_gold_documents": gold_documents,
        "eval_argument_cluster_link_ungraded_documents": ungraded,
    }
    if gold_documents:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out["eval_argument_cluster_link_strict_micro_precision"] = precision
        out["eval_argument_cluster_link_strict_micro_recall"] = recall
        out["eval_argument_cluster_link_strict_micro_f1"] = f1
    return out


def _clusters(pred: Dict[str, Any]) -> List[Any]:
    clusters = pred.get("coref_clusters") if isinstance(pred, dict) else None
    return clusters if isinstance(clusters, list) else []


def _gold_clusters(gold: Dict[str, Any]) -> List[Any]:
    clusters = _clusters(gold if isinstance(gold, dict) else {})
    if any(isinstance(cluster, list) and cluster for cluster in clusters):
        return clusters
    return []


def _pred_arguments(pred: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    block = pred.get("event_extraction")
    if not isinstance(block, dict):
        return
    for mentions in block.values():
        if not isinstance(mentions, list):
            continue
        for event in mentions:
            if not isinstance(event, dict):
                continue
            for arg in event.get("arguments") or []:
                if isinstance(arg, dict):
                    yield arg


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


def _hits(span: Span, clusters: Sequence[Any]) -> List[int]:
    found = []
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, list):
            continue
        if any(_span(mention) == span for mention in cluster):
            found.append(index)
    return found


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


def _gold_links(gold: Dict[str, Any], texts: Sequence[set], spans: Sequence[frozenset]) -> Iterable[Link]:
    events = gold.get("events") if isinstance(gold, dict) else None
    if not isinstance(events, list):
        return
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
                yield etype, role, surface, spans[hits[0]]


def _pred_links(pred: Dict[str, Any], spans: Sequence[frozenset]) -> Iterable[Link]:
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
                role, surface = arg["role"].strip(), _surface(arg.get("entity"))
                index = arg.get("coref_cluster")
                if (not role or not surface or isinstance(index, bool) or not isinstance(index, int)
                        or not 0 <= index < len(spans)):
                    continue
                yield etype.strip(), role, surface, spans[index]


def _take(links: Sequence[Link], used: List[bool], etype: str, role: str, surface: str,
          spans: frozenset) -> bool:
    for index, (petype, prole, psurface, pspans) in enumerate(links):
        if used[index] or petype != etype or prole != role or psurface != surface or not (pspans & spans):
            continue
        used[index] = True
        return True
    return False
