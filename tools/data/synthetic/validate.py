"""Validate a model's JSON reply and build a strict GLiNER2 training record.

Mirrors the tools/data converters: every span must be a verbatim substring of
the document, labels must be in the fixed ontology, and anything that fails is
dropped (never repaired or hallucinated into place). Returns the record in the
exact shape ``InputExample.from_dict`` consumes, plus per-task kept/dropped
counts for reporting.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from schema_spec import (
    CLASSIFICATION_TASKS, ENTITY_TYPES, EVENT_ONTOLOGY, MULTI_LABEL_TASKS,
    RELATION_TYPES, STRUCTURE_TEMPLATES,
)

_ENTITY_SET = set(ENTITY_TYPES)
_RELATION_SET = set(RELATION_TYPES)


def parse_reply(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the model reply into a dict, tolerating stray prose around the JSON."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _in(text: str, surface: Any) -> Optional[str]:
    """Return the stripped surface if it is a verbatim substring of text, else None."""
    if not isinstance(surface, str):
        return None
    s = surface.strip()
    return s if s and s in text else None


def _entities(text: str, items: Any, stats: Counter,
              declared: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Entity annotations, keyed by type.

    ``declared`` is this document's SAMPLED type subset. Two things depend on it and
    neither works without the other: annotations for a type that was not asked about are
    rejected (otherwise a type sampled as a NEGATIVE for this document can arrive as a
    positive anyway, contaminating it), and every asked-about type is seeded with an
    empty list so a type that is genuinely absent survives as ``{type: []}``. That empty
    list is the negative: the query is still emitted and every candidate span under it
    becomes a negative -- measured, three queries and one gold mention on a
    one-present/two-absent record.
    """
    allowed = set(declared) if declared else _ENTITY_SET
    # Seed FIRST: an asked-about type with no valid span must survive as a negative.
    out: Dict[str, List[str]] = {t: [] for t in (declared or [])}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        etype = it.get("type")
        surface = _in(text, it.get("text"))
        if etype not in allowed or surface is None:
            stats["entities_dropped"] += 1
            continue
        bucket = out.setdefault(etype, [])
        if surface not in bucket:
            bucket.append(surface)
            stats["entities_kept"] += 1
    return out


def _relations(text: str, items: Any, stats: Counter) -> List[Dict[str, Dict[str, str]]]:
    out: List[Dict[str, Dict[str, str]]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        rtype = it.get("type")
        head = _in(text, it.get("head"))
        tail = _in(text, it.get("tail"))
        if rtype not in _RELATION_SET or head is None or tail is None:
            stats["relations_dropped"] += 1
            continue
        out.append({rtype: {"head": head, "tail": tail}})
        stats["relations_kept"] += 1
    return out


def _events(text: str, items: Any, stats: Counter) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        etype = it.get("event_type")
        trigger = _in(text, it.get("trigger"))
        allowed = EVENT_ONTOLOGY.get(etype)
        if allowed is None or trigger is None:
            stats["events_dropped"] += 1
            continue
        roles = set(allowed)
        args: List[Dict[str, str]] = []
        seen = set()
        for arg in it.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            entity = _in(text, arg.get("entity"))
            if role not in roles or entity is None or (role, entity) in seen:
                stats["arguments_dropped"] += 1
                continue
            seen.add((role, entity))
            args.append({"role": role, "entity": entity})
            stats["arguments_kept"] += 1
        out.append({"event_type": etype, "triggers": [trigger], "arguments": args})
        stats["events_kept"] += 1
    return out


def _classifications(items: Any, stats: Counter) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        task = it.get("task")
        vocab = CLASSIFICATION_TASKS.get(task)
        if vocab is None:
            stats["classifications_dropped"] += 1
            continue
        chosen = it.get("labels")
        if isinstance(chosen, str):
            chosen = [chosen]
        true = [c for c in (chosen or []) if c in vocab]
        if not true:
            stats["classifications_dropped"] += 1
            continue
        multi = task in MULTI_LABEL_TASKS or len(true) > 1
        out.append({
            "task": task, "labels": list(vocab),
            "true_label": true, "multi_label": multi,
        })
        stats["classifications_kept"] += 1
    return out


def _structures(text: str, items: Any, stats: Counter) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        stype = it.get("type")
        template = STRUCTURE_TEMPLATES.get(stype)
        fields_in = it.get("fields")
        if template is None or not isinstance(fields_in, dict):
            stats["structures_dropped"] += 1
            continue
        fields_out: Dict[str, Any] = {}
        for fname, choices in template.items():
            if fname not in fields_in:
                continue
            value = fields_in[fname]
            if choices is not None:  # choice field
                if isinstance(value, str) and value in choices:
                    fields_out[fname] = {"value": value, "choices": list(choices)}
            else:                    # extractive free-text field
                v = _in(text, value)
                if v is not None:
                    fields_out[fname] = v
        if fields_out:
            out.append({stype: fields_out})
            stats["structures_kept"] += 1
        else:
            stats["structures_dropped"] += 1
    return out


def _merge_output(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge freshly-annotated tasks into an existing gold ``output`` (kept).

    Entities merge by type (unique surfaces); relations/events/structures are
    appended; classifications keep the gold and add only tasks not already
    present -- so re-annotating never discards existing supervision.
    """
    out = copy.deepcopy(base) if base else {}
    if new.get("entities"):
        dst = out.setdefault("entities", {})
        for t, surfaces in new["entities"].items():
            bucket = dst.setdefault(t, [])
            for s in surfaces:
                if s not in bucket:
                    bucket.append(s)
    for key in ("relations", "events", "json_structures"):
        if new.get(key):
            out.setdefault(key, []).extend(new[key])
    if new.get("classifications"):
        have = {c.get("task") for c in out.get("classifications", [])}
        add = [c for c in new["classifications"] if c.get("task") not in have]
        if add:
            out.setdefault("classifications", []).extend(add)
    return out


def build_record(reply: Dict[str, Any], tasks: List[str], stats: Counter,
                 text_override: Optional[str] = None,
                 base_output: Optional[Dict[str, Any]] = None,
                 declared: Optional[Dict[str, List[str]]] = None,
                 ) -> Optional[Dict[str, Any]]:
    """Turn a parsed reply into a GLiNER2 record, or None if nothing survives.

    ``text_override`` annotates existing text (verbatim-checked against it)
    instead of the model's generated ``reply["text"]``. ``base_output`` merges
    the new annotations onto existing gold (re-annotation mode).
    """
    if text_override is not None:
        text = text_override
    else:
        text = reply.get("text")
        if not isinstance(text, str) or not text.strip():
            stats["no_text"] += 1
            return None
        text = text.strip()
    if not text:
        stats["no_text"] += 1
        return None

    declared = declared or {}
    output: Dict[str, Any] = {}
    if "entities" in tasks:
        ents = _entities(text, reply.get("entities"), stats, declared.get("entities"))
        if ents:
            output["entities"] = ents
            neg = sum(1 for v in ents.values() if not v)
            stats["entity_negatives"] += neg
            stats["entity_positives"] += len(ents) - neg
    if "relations" in tasks:
        rels = _relations(text, reply.get("relations"), stats)
        if rels:
            output["relations"] = rels
    if "events" in tasks:
        evs = _events(text, reply.get("events"), stats)
        if evs:
            output["events"] = evs
    if "classifications" in tasks:
        cls = _classifications(reply.get("classifications"), stats)
        if cls:
            output["classifications"] = cls
    if "structures" in tasks:
        structs = _structures(text, reply.get("structures"), stats)
        if structs:
            output["json_structures"] = structs

    merged = _merge_output(base_output, output) if base_output else output
    if not merged:
        stats["empty_output"] += 1
        return None
    return {"input": text, "output": merged}
