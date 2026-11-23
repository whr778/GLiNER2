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
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import clean_text, dumps_record  # noqa: E402


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _checked_span(text: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """``{text, start, end}`` for a span that covers exactly that slice of ``text``."""
    start, end = item.get("start"), item.get("end")
    if not _is_int(start) or not _is_int(end) or not (0 <= start < end <= len(text)):
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


def _parse_trigger(text: str, trigger: Any) -> Optional[Tuple[Optional[str], Any, Any]]:
    """``(trigger_id, span_or_str, dedup_key)`` for one event_facts trigger entry."""
    if isinstance(trigger, dict):
        tid = trigger.get("trigger_id")
        tid = tid if isinstance(tid, str) and tid else None
        if trigger.get("start") is not None and trigger.get("end") is not None:
            span = _checked_span(text, trigger)
            return tid, span, (span["start"], span["end"])
        raw = trigger.get("text")
        if isinstance(raw, str) and raw:
            return tid, raw, raw
        return None
    if isinstance(trigger, str) and trigger:
        return None, trigger, trigger
    return None


def _parse_argument(text: str, arg: Dict[str, Any], stats: Counter) -> Tuple[Optional[str], Dict[str, Any], Any]:
    """``(trigger_id, argument_dict, dedup_key)`` for one event_facts argument entry."""
    role = arg.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError("event argument has no role")
    stats["event_arguments_in"] += 1
    tid = arg.get("trigger_id")
    tid = tid if isinstance(tid, str) and tid else None
    if arg.get("start") is not None and arg.get("end") is not None:
        entity: Any = _checked_span(text, arg)
        key: Any = (role, entity["start"], entity["end"])
    else:
        raw = arg.get("text") if isinstance(arg.get("text"), str) else arg.get("entity")
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"event argument {role!r} has no span and no text")
        entity = raw
        key = (role, raw)
    return tid, {"role": role, "entity": entity}, key


def _span_distance(a: Any, b: Any) -> Optional[int]:
    """Character gap between two offset dicts, or None if either lacks offsets."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return None
    if a["end"] <= b["start"]:
        return b["start"] - a["end"]
    if b["end"] <= a["start"]:
        return a["start"] - b["end"]
    return 0


def _nearest_group(groups: List[Dict[str, Any]], entity: Any) -> Dict[str, Any]:
    """Trigger group nearest to ``entity`` by character distance, falling back to the first."""
    best = groups[0]
    best_dist: Optional[int] = None
    for group in groups:
        for trig in group["triggers"]:
            d = _span_distance(trig, entity)
            if d is not None and (best_dist is None or d < best_dist):
                best_dist = d
                best = group
    return best


def _events(text: str, output: Dict[str, Any], stats: Counter) -> List[Dict[str, Any]]:
    """Build per-mention GLiNER2 events from ``event_facts``.

    Each ``trigger_id`` becomes its own event dict (matching ``convert_ace2005.py``'s
    default ``merge_event_mentions=False``), carrying only the arguments attached to
    that trigger. When triggers carry no ``trigger_id``, all triggers in the fact stay
    in one event dict.
    """
    facts = output.get("event_facts") or []
    if not facts:
        return output.get("events") or []
    events: List[Dict[str, Any]] = []
    for fact in facts:
        event_type = fact.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_facts entry has no event_type")
        stats["event_facts"] += 1
        parsed_triggers = [
            parsed
            for trigger in (fact.get("triggers") or [])
            if (parsed := _parse_trigger(text, trigger)) is not None
        ]
        if not parsed_triggers:
            stats["events_without_trigger"] += 1
            continue

        split_by_tid = any(tid is not None for tid, _, _ in parsed_triggers)
        groups: List[Dict[str, Any]] = []
        by_tid: Dict[str, Dict[str, Any]] = {}
        if split_by_tid:
            for tid, trig_item, trig_key in parsed_triggers:
                group = by_tid.get(tid) if tid is not None else None
                if group is None:
                    group = {"triggers": [], "seen_triggers": set(), "arguments": [], "seen_args": set()}
                    groups.append(group)
                    if tid is not None:
                        by_tid[tid] = group
                if trig_key not in group["seen_triggers"]:
                    group["seen_triggers"].add(trig_key)
                    group["triggers"].append(trig_item)
        else:
            group = {"triggers": [], "seen_triggers": set(), "arguments": [], "seen_args": set()}
            for _, trig_item, trig_key in parsed_triggers:
                if trig_key not in group["seen_triggers"]:
                    group["seen_triggers"].add(trig_key)
                    group["triggers"].append(trig_item)
            groups.append(group)

        for arg in fact.get("arguments") or []:
            if not isinstance(arg, dict):
                continue
            tid, arg_dict, arg_key = _parse_argument(text, arg, stats)
            if tid is not None and tid in by_tid:
                target = by_tid[tid]
            elif len(groups) == 1:
                target = groups[0]
            else:
                target = _nearest_group(groups, arg_dict["entity"])
            if arg_key in target["seen_args"]:
                continue
            target["seen_args"].add(arg_key)
            target["arguments"].append(arg_dict)

        for group in groups:
            events.append({
                "event_type": event_type,
                "triggers": group["triggers"],
                "arguments": group["arguments"],
            })
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


_OFFICE_OUTPUT_KEYS = ("entity_facts", "relation_facts", "event_facts", "clusters")


def is_office_record(record: Dict[str, Any]) -> bool:
    """True when the record still carries the office side-list keys."""
    if not isinstance(record, dict):
        return False
    output = record.get("output") if isinstance(record.get("output"), dict) else {}
    return any(key in output for key in _OFFICE_OUTPUT_KEYS) or "clusters" in record


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
    adapted: Dict[str, Any] = {"input": text, "output": built}
    clusters = _clusters(text, output, record, stats)
    if clusters:
        adapted["coref_clusters"] = clusters
    stats["records"] += 1
    return adapted


def _clean_text_and_map(text: str) -> Tuple[str, Optional[List[int]]]:
    """Apply ``_split.clean_text`` and return an index map ``old_idx -> new_idx`` when text changed."""
    cleaned = clean_text(text)
    if cleaned == text:
        return text, None
    pos = [0] * (len(text) + 1)
    pieces: List[str] = []
    cur = 0
    i = 0
    n = len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        piece = clean_text(text[i:j])
        for k in range(i, j):
            pos[k] = cur
        pieces.append(piece)
        cur += len(piece)
        i = j
    pos[n] = cur
    rebuilt = "".join(pieces)
    if rebuilt != cleaned:
        raise ValueError("clean_text changed cross-character boundaries; cannot rebase offsets safely")
    return rebuilt, pos


def _rebase_span(cleaned: str, pos: List[int], span: Any) -> Any:
    """Rebase one ``{text, start, end}`` dict onto ``cleaned`` using ``pos``."""
    if not (isinstance(span, dict) and _is_int(span.get("start")) and _is_int(span.get("end"))):
        return span
    ns, ne = pos[span["start"]], pos[span["end"]]
    if not (0 <= ns < ne <= len(cleaned)):
        raise ValueError(f"span [{span['start']}, {span['end']}) collapsed under clean_text")
    return {"text": cleaned[ns:ne], "start": ns, "end": ne}


def clean_and_rebase_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ``record['input']`` via ``clean_text`` and rebase all offset spans before ``dumps_record``."""
    text = record["input"]
    cleaned, pos = _clean_text_and_map(text)
    if pos is None:
        return record
    out = dict(record)
    out["input"] = cleaned
    output = dict(record.get("output") or {})
    if "events" in output:
        rebased_events = []
        for ev in output["events"]:
            rebased_events.append({
                "event_type": ev["event_type"],
                "triggers": [_rebase_span(cleaned, pos, t) for t in ev.get("triggers") or []],
                "arguments": [
                    {"role": a["role"], "entity": _rebase_span(cleaned, pos, a["entity"])}
                    for a in ev.get("arguments") or []
                ],
            })
        output["events"] = rebased_events
    out["output"] = output
    if "coref_clusters" in record:
        out["coref_clusters"] = [
            [_rebase_span(cleaned, pos, m) for m in cluster]
            for cluster in record["coref_clusters"]
        ]
    return out


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
            handle.write(dumps_record(clean_and_rebase_record(adapt_record(record, stats))) + "\n")
    print(
        f"records={stats['records']} entity_mentions={stats['entity_mentions']} "
        f"relation_facts={stats['relation_facts']} relations_skipped={stats['relations_skipped']} "
        f"event_facts={stats['event_facts']} events_without_trigger={stats['events_without_trigger']} "
        f"event_arguments_in={stats['event_arguments_in']} "
        f"clusters_in={stats['clusters_in']} coref_mentions={stats['coref_mentions']}"
    )


if __name__ == "__main__":
    main()
