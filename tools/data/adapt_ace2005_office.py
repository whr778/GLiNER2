"""Convert the office ACE 2005 JSON into the GLiNER2 records this repo trains on.

The office file keeps a string summary in ``output.entities``, ``output.relations``
and ``output.events``, and the character offsets beside it:

* ``output.entity_facts`` — one mention, with ``start``, ``end``, ``text``,
  ``mention_type``, ``entity_type``
* ``output.relation_facts`` — ``relation_type`` plus ``head`` and ``tail`` spans
* ``output.event_facts`` — ``event_type``, trigger spans, argument spans
* ``output.clusters`` — ``cluster_id`` plus ``mentions`` of ``[start, end]``

``InputExample.from_dict`` reads the string summary and does not read those four
keys. A string trigger or argument then matches every copy of that surface.
``coref.attach`` reads top-level ``coref_clusters`` and finds nothing.

This adapter writes the shape ``convert_ace2005.py --offsets`` writes. Event
triggers and arguments become ``{text, start, end}``. Clusters become
``coref_clusters``. Entities and relations stay strings, as they do in that
converter: a relation's head and tail are the span text. ``mention_types`` is
the first ``mention_type`` seen for each surface.

A span whose slice of ``input`` is not its ``text`` raises. Re-running the
adapter on its own output leaves the record as it is.

    uv run python tools/data/adapt_ace2005_office.py \\
        --input data/tmp/meh.json --out data/ace2005_office.jsonl

``data.office_ace: true`` on a training config runs this adapter as each split
is read. The ACE configs under ``tools/train/config/ace2005/`` set that flag.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402


def _checked_span(text: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """``{text, start, end}`` for a span that covers exactly that slice of ``text``."""
    start, end = item.get("start"), item.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
        raise ValueError(f"span [{start}, {end}) is outside a text of length {len(text)}")
    covered = text[start:end]
    given = item.get("text")
    if isinstance(given, str) and given != covered:
        raise ValueError(f"span [{start}, {end}) covers {covered!r}, not {given!r}")
    return {"text": covered, "start": start, "end": end}


def _endpoint(text: str, value: Any) -> str:
    """Relation endpoint text. A span dict is checked against ``text``; a string passes through."""
    if isinstance(value, dict):
        return _checked_span(text, value)["text"]
    if isinstance(value, str):
        return value
    raise ValueError(f"relation endpoint must be a span or a string, got {type(value).__name__}")


def _entities(text: str, output: Dict[str, Any], stats: Counter) -> Dict[str, List[str]]:
    facts = output.get("entity_facts") or []
    if not facts:
        return output.get("entities") or {}
    by_type: Dict[str, List[str]] = {}
    for fact in facts:
        etype = fact.get("entity_type")
        if not isinstance(etype, str) or not etype:
            raise ValueError("entity_facts entry has no entity_type")
        surface = _checked_span(text, fact)["text"]
        bucket = by_type.setdefault(etype, [])
        if surface not in bucket:
            bucket.append(surface)
        stats["entity_mentions"] += 1
    for etype, values in (output.get("entities") or {}).items():
        bucket = by_type.setdefault(etype, [])
        for value in values or []:
            if isinstance(value, str) and value and value not in bucket:
                bucket.append(value)
    return {etype: values for etype, values in by_type.items() if etype and values}


def _mention_types(text: str, output: Dict[str, Any]) -> Dict[str, str]:
    facts = output.get("entity_facts") or []
    if not facts:
        return output.get("mention_types") or {}
    types: Dict[str, str] = {}
    for fact in facts:
        mention_type = fact.get("mention_type")
        if not isinstance(mention_type, str) or not mention_type:
            continue
        types.setdefault(_checked_span(text, fact)["text"], mention_type)
    return types


def _relations(text: str, output: Dict[str, Any], stats: Counter) -> List[Dict[str, Dict[str, str]]]:
    facts = output.get("relation_facts") or []
    if not facts:
        return output.get("relations") or []
    out: List[Dict[str, Dict[str, str]]] = []
    seen = set()
    for fact in facts:
        rel_type = fact.get("relation_type")
        if not isinstance(rel_type, str) or not rel_type:
            raise ValueError("relation_facts entry has no relation_type")
        head = _endpoint(text, fact.get("head"))
        tail = _endpoint(text, fact.get("tail"))
        stats["relation_facts"] += 1
        if not head or not tail or head == tail:
            stats["relations_skipped"] += 1
            continue
        key = (rel_type, head, tail)
        if key in seen:
            continue
        seen.add(key)
        out.append({rel_type: {"head": head, "tail": tail}})
    return out


def _events(text: str, output: Dict[str, Any], stats: Counter) -> List[Dict[str, Any]]:
    facts = output.get("event_facts") or []
    if not facts:
        return output.get("events") or []
    events: List[Dict[str, Any]] = []
    for fact in facts:
        event_type = fact.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_facts entry has no event_type")
        stats["event_facts"] += 1
        triggers: List[Any] = []
        seen_triggers = set()
        for trigger in fact.get("triggers") or []:
            if isinstance(trigger, dict) and "start" in trigger:
                span = _checked_span(text, trigger)
                key = (span["start"], span["end"])
                if key in seen_triggers:
                    continue
                seen_triggers.add(key)
                triggers.append(span)
            elif isinstance(trigger, str) and trigger:
                triggers.append(trigger)
        if not triggers:
            stats["events_without_trigger"] += 1
            continue
        arguments: List[Dict[str, Any]] = []
        seen_args = set()
        for arg in fact.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            role = arg.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError("event argument has no role")
            stats["event_arguments_in"] += 1
            if "start" in arg and "end" in arg:
                entity: Any = _checked_span(text, arg)
                key = (role, entity["start"], entity["end"])
            else:
                raw = arg.get("text") if isinstance(arg.get("text"), str) else arg.get("entity")
                if not isinstance(raw, str) or not raw:
                    raise ValueError(f"event argument {role!r} has no span and no text")
                entity = raw
                key = (role, raw)
            if key in seen_args:
                continue
            seen_args.add(key)
            arguments.append({"role": role, "entity": entity})
        events.append({"event_type": event_type, "triggers": triggers, "arguments": arguments})
    return events


def _one_cluster_mention(text: str, mention: Any) -> Dict[str, Any]:
    if isinstance(mention, dict):
        return _checked_span(text, mention)
    if isinstance(mention, (list, tuple)) and len(mention) == 2:
        start, end = mention
        return _checked_span(text, {"start": start, "end": end})
    raise ValueError(f"cluster mention must be [start, end] or a span, got {mention!r}")


def _clusters(text: str, output: Dict[str, Any], record: Dict[str, Any], stats: Counter) -> Optional[List[List[Dict[str, Any]]]]:
    raw = output.get("clusters")
    if raw is None:
        raw = record.get("clusters")
    if raw is None:
        existing = record.get("coref_clusters")
        return existing if existing else None
    clusters: List[List[Dict[str, Any]]] = []
    for cluster in raw:
        stats["clusters_in"] += 1
        mentions: List[Dict[str, Any]] = []
        seen = set()
        for mention in (cluster.get("mentions") if isinstance(cluster, dict) else None) or []:
            span = _one_cluster_mention(text, mention)
            key = (span["start"], span["end"])
            if key in seen:
                continue
            seen.add(key)
            mentions.append(span)
        if not mentions:
            continue
        mentions.sort(key=lambda span: (span["start"], span["end"]))
        clusters.append(mentions)
    clusters.sort(key=lambda cluster: (cluster[0]["start"], cluster[0]["end"]))
    stats["coref_mentions"] += sum(len(cluster) for cluster in clusters)
    return clusters or None


def adapt_record(record: Dict[str, Any], stats: Optional[Counter] = None) -> Dict[str, Any]:
    """One office record to one GLiNER2 record. ``stats`` counts the facts read."""
    stats = stats if stats is not None else Counter()
    text = record.get("input")
    if not isinstance(text, str):
        raise ValueError("record has no string input")
    output = record.get("output") or {}
    if not isinstance(output, dict):
        raise ValueError("record output is not an object")
    built: Dict[str, Any] = {}
    entities = _entities(text, output, stats)
    if entities:
        built["entities"] = entities
    relations = _relations(text, output, stats)
    if relations:
        built["relations"] = relations
    events = _events(text, output, stats)
    if events:
        built["events"] = events
    mention_types = _mention_types(text, output)
    if mention_types:
        built["mention_types"] = mention_types
    if not built:
        raise ValueError("record has no entities, relations, or events")
    adapted: Dict[str, Any] = {"input": text, "output": built}
    clusters = _clusters(text, output, record, stats)
    if clusters:
        adapted["coref_clusters"] = clusters
    stats["records"] += 1
    return adapted


def load_records(path: Path) -> Iterable[Dict[str, Any]]:
    """A JSONL file, a JSON list, or one JSON object."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        yield from payload
        return
    if isinstance(payload, dict):
        yield payload
        return
    raise ValueError(f"{path} is not a JSON object or list")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Office JSON or JSONL")
    parser.add_argument("--out", required=True, type=Path, help="GLiNER2 JSONL to write")
    args = parser.parse_args()
    stats: Counter = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in load_records(args.input):
            handle.write(dumps_record(adapt_record(record, stats)) + "\n")
    print(
        f"records={stats['records']} entity_mentions={stats['entity_mentions']} "
        f"relation_facts={stats['relation_facts']} relations_skipped={stats['relations_skipped']} "
        f"event_facts={stats['event_facts']} events_without_trigger={stats['events_without_trigger']} "
        f"event_arguments_in={stats['event_arguments_in']} "
        f"clusters_in={stats['clusters_in']} coref_mentions={stats['coref_mentions']}"
    )


if __name__ == "__main__":
    main()
