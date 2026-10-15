"""Gold coreference clusters, carried from a record to the collated batch as word spans.

A record's top-level ``coref_clusters`` (lists of mentions ``{text, start, end}``, code-point offsets
into ``input``) rides to the collator inside the item's schema under ``SCHEMA_KEY`` -- the way label
negatives ride -- and is taken out again before anything compiles the schema. The collator maps each
mention onto the batch's text words through ``start_mappings`` / ``end_mappings`` (character spans
of the words in the sample's text) and keeps only mentions whose edges are word edges inside the
encoded window; the rest are counted, never guessed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_KEY = "_coref_clusters"
WordMention = Tuple[int, int, int]  # (first word, last word inclusive, cluster index)
COUNTS: Dict[str, int] = {"mentions": 0, "placed": 0, "unaligned": 0, "outside_window": 0}


def attach(schema: Dict[str, Any], clusters: Optional[List[List[Dict]]]) -> Dict[str, Any]:
    """The item schema carrying ``clusters`` (a copy; the record's own output is never mutated)."""
    return dict(schema, **{SCHEMA_KEY: clusters}) if clusters else schema


def take(batch: Sequence[Tuple[str, Any]]) -> Tuple[List[Tuple[str, Any]], Dict[str, List[List[Dict]]]]:
    """Strip the clusters from every schema; return the clean batch and ``{text: clusters}``."""
    clean, clusters = [], {}
    for text, schema in batch:
        if isinstance(schema, dict) and SCHEMA_KEY in schema:
            clusters[text] = schema[SCHEMA_KEY]
            schema = {k: v for k, v in schema.items() if k != SCHEMA_KEY}
        clean.append((text, schema))
    return clean, clusters


def word_mentions(clusters: List[List[Dict]], starts: Sequence[int], ends: Sequence[int],
                  n_words: int) -> List[WordMention]:
    """Mentions as (first word, last word, cluster) where both edges are word edges in the window."""
    first = {s: w for w, s in enumerate(starts[:n_words])}
    last = {e: w for w, e in enumerate(ends[:n_words])}
    out = []
    for ci, cluster in enumerate(clusters):
        for m in cluster:
            COUNTS["mentions"] += 1
            ws, we = first.get(m["start"]), last.get(m["end"])
            if ws is not None and we is not None and ws <= we:
                out.append((ws, we, ci))
                COUNTS["placed"] += 1
            elif m["start"] >= (ends[n_words - 1] if n_words else 0):
                COUNTS["outside_window"] += 1
            else:
                COUNTS["unaligned"] += 1
    return out


def batch_mentions(out, clusters: Dict[str, List[List[Dict]]]) -> Tuple[List[WordMention], ...]:
    """Per collated sample, its word mentions (empty when the record had no clusters). Matched by text,
    because a skipped record shifts positions; the processor appends a final '.', so match by prefix."""
    if not clusters:
        return ()
    res = []
    for i, text in enumerate(out.original_texts):
        src = clusters.get(text) or next((c for t, c in clusters.items() if text.startswith(t)), None)
        res.append(word_mentions(src, out.start_mappings[i], out.end_mappings[i], out.text_word_counts[i])
                   if src else [])
    return tuple(res)
