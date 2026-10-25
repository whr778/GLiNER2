"""Gold argument fillers include the other mentions of their coreference cluster.

The junction's positive cells are the spans listed on the event. ``coref_clusters``
is stripped before that list is compiled, so a pronoun in the cluster never becomes
a cell. Expansion runs while both are still on the record and adds the other
mentions as offset spans of the same role.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _events(record: Dict[str, Any]) -> List[Any]:
    output = record.get("output")
    if not isinstance(output, dict):
        output = record.get("schema") if isinstance(record.get("schema"), dict) else None
    if not isinstance(output, dict):
        return []
    events = output.get("events")
    return events if isinstance(events, list) else []


def attach_coref_copies(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Copy clusters from coref-only records onto event records with the same text.

    A coref-only record has clusters and no events. An event record that already
    has clusters is left alone. Coref-only records stay in the list. Two coref-only
    copies of one text are not copied.
    """
    donors: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        if record.get("coref_clusters") and not _events(record):
            text = record.get("input", record.get("text"))
            if isinstance(text, str):
                donors.setdefault(text, []).append(index)
    attached = ambiguous = 0
    for record in records:
        if record.get("coref_clusters") or not _events(record):
            continue
        text = record.get("input", record.get("text"))
        found = donors.get(text) if isinstance(text, str) else None
        if not found:
            continue
        if len(found) != 1:
            ambiguous += 1
            continue
        record["coref_clusters"] = records[found[0]]["coref_clusters"]
        attached += 1
    return {"attached": attached, "ambiguous": ambiguous, "coref_only": sum(map(len, donors.values()))}


def expand_argument_coreferents(schema: Dict[str, Any], clusters, text: str) -> Dict[str, Any]:
    """Return ``schema``, or a copy whose arguments gained their cluster's other mentions.

    An argument string that sits in exactly one cluster gains that cluster's mentions
    whose text differs, as ``{text, start, end}`` fillers of the same role. Offsets are
    that mention's. A string in two clusters is not expanded. The caller's schema is
    not mutated.
    """
    events = schema.get("events") if isinstance(schema, dict) else None
    if not isinstance(events, list) or not events or not clusters or not isinstance(text, str):
        return schema
    by_text: Dict[str, List[int]] = {}
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, list):
            continue
        for mention in cluster:
            if isinstance(mention, dict) and isinstance(mention.get("text"), str):
                by_text.setdefault(mention["text"].strip(), []).append(index)
    added = skipped = 0
    new_events = []
    changed = False
    for event in events:
        if not isinstance(event, dict):
            new_events.append(event)
            continue
        arguments = event.get("arguments")
        if not isinstance(arguments, list):
            new_events.append(event)
            continue
        extra: List[Dict[str, Any]] = []
        seen = {(a.get("role"), a["entity"]["start"], a["entity"]["end"])
                for a in arguments
                if isinstance(a, dict) and isinstance(a.get("entity"), dict)
                and {"start", "end"} <= a["entity"].keys()}
        for argument in arguments:
            if not isinstance(argument, dict):
                continue
            entity = argument.get("entity")
            surface = entity.get("text") if isinstance(entity, dict) else entity
            if not isinstance(surface, str) or not surface.strip():
                continue
            hits = set(by_text.get(surface.strip(), ()))
            if len(hits) != 1:
                if len(hits) > 1:
                    skipped += 1
                continue
            cluster = clusters[next(iter(hits))]
            role = argument.get("role")
            for mention in cluster:
                if not isinstance(mention, dict):
                    continue
                mention_text = mention.get("text")
                start, end = mention.get("start"), mention.get("end")
                if (not isinstance(mention_text, str) or mention_text.strip() == surface.strip()
                        or not isinstance(start, int) or not isinstance(end, int)):
                    continue
                if text[start:end] != mention_text or (role, start, end) in seen:
                    continue
                extra.append({"role": role, "entity": {"text": mention_text, "start": start, "end": end}})
                seen.add((role, start, end))
                added += 1
        if extra:
            changed = True
            new_events.append(dict(event, arguments=list(arguments) + extra))
        else:
            new_events.append(event)
    if not changed and skipped == 0:
        return schema
    out = dict(schema)
    out["events"] = new_events
    out["_argument_coref_added"] = added
    out["_argument_coref_skipped"] = skipped
    return out
