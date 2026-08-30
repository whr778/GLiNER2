"""Schema presets for the viewer: a few built-ins plus schemas auto-derived
from any ``data/<name>.train.jsonl`` present in the repo (reusing
``scripts/infer_file.py::_derive_schema``), so a user can one-click the exact
ontology a trained model expects."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from config import data_root, scripts_root

# User-friendly SchemaInput dicts (entities / classifications / events /
# relations / structures). These are what the /extract endpoint validates.
BUILTIN: Dict[str, Dict[str, Any]] = {
    "General entities": {
        "entities": ["person", "organization", "location", "date", "money", "product", "event"],
    },
    "Sentiment": {
        "classifications": [
            {"task": "sentiment", "labels": ["positive", "negative", "neutral"], "multi_label": False},
        ],
    },
    "Events (ACE-style)": {
        "entities": ["person", "organization", "location"],
        "events": {
            "Attack": ["Attacker", "Target", "Place", "Time"],
            "Transport": ["Agent", "Artifact", "Origin", "Destination"],
            "Meet": ["Participant", "Place", "Time"],
        },
    },
    "Contact (structure)": {
        "structures": {
            "contact": {
                # mode/anchor are declared, not omitted. Without them a structure cannot
                # be decoded on the BOUNDARY architecture and extraction returns {} with
                # no error -- which reads in the UI as "the model found nothing". Span
                # models ignore the metadata, so declaring it is free. The backend also
                # defaults it (`_declare_records`); this keeps the preset honest as an
                # example someone might copy.
                "mode": "natural",
                "anchor": "name",
                "fields": [
                    {"name": "name", "dtype": "str"},
                    {"name": "email", "dtype": "str"},
                    {"name": "phone", "dtype": "str"},
                    {"name": "organization", "dtype": "str"},
                ],
            },
        },
    },
    # The EKF pipeline's own two schemas, so the viewer can reproduce what stages 0 and 2
    # actually ask of a model. See tools/ekf_showcase/run_pipeline.py.
    "Casualty report (EKF stage 2)": {
        "structures": {
            "casualty_report": {
                "mode": "natural",
                "anchor": "dead",
                "fields": [
                    {"name": "location", "dtype": "str",
                     "description": "the country or place these deaths occurred in"},
                    {"name": "dead", "dtype": "str",
                     "description": "number of people killed or confirmed dead, not injured/missing/displaced"},
                    {"name": "injured", "dtype": "str",
                     "description": "number of people injured or hurt, not killed/missing/displaced/homeless"},
                    {"name": "missing", "dtype": "str",
                     "description": "number of people missing or unaccounted for, not killed/injured"},
                ],
            },
        },
    },
    "Relevance gate (EKF stage 0)": {
        # ONE classification task deliberately. A second task collapses `relevance` to
        # `other` at confidence 1.0 on boundary models, and a gate that admits nothing has
        # a perfect false-positive rate -- so the collapse reads as success.
        "classifications": [
            {"task": "relevance",
             "labels": ["mass_casualty", "other"],
             "multi_label": False},
        ],
    },
}


def _raw_to_schema_input(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ``_derive_schema``'s internal dict to the SchemaInput form.

    entities (list) and events ({type:[roles]}) already match; relations come
    back as ``[{name: {head, tail}}]`` and must become ``[name]``; classifications
    need an explicit ``multi_label`` and >= 2 labels to validate.
    """
    si: Dict[str, Any] = {}
    if raw.get("entities"):
        si["entities"] = raw["entities"]
    if raw.get("events"):
        si["events"] = raw["events"]

    classifications = [
        {"task": c["task"], "labels": c["labels"], "multi_label": False}
        for c in raw.get("classifications") or []
        if c.get("task") and len(c.get("labels") or []) >= 2
    ]
    if classifications:
        si["classifications"] = classifications

    names: List[str] = []
    for rel in raw.get("relations") or []:
        if isinstance(rel, dict):
            names.extend(rel.keys())
        elif isinstance(rel, str):
            names.append(rel)
    if names:
        si["relations"] = names
    return si


def _trigger_only_event_types(train_path: Path) -> List[str]:
    """Event types that appear with a trigger but never carry a role.

    ``_derive_schema`` drops role-less event types, so trigger-only corpora
    (MAVEN, mendeley_ed) would otherwise yield no schema. Event
    detection is typed-span detection, so we surface these as entity labels
    the model can detect. The first-line check skips the full scan for
    non-event corpora (every record in these sets carries events)."""
    has_trigger: set = set()
    has_role: set = set()
    with open(train_path, encoding="utf-8") as f:
        first = f.readline()
        if '"events"' not in first:
            return []
        for line in itertools.chain([first], f):
            line = line.strip()
            if not line:
                continue
            for ev in (json.loads(line).get("output") or {}).get("events") or []:
                etype = ev.get("event_type")
                if not etype:
                    continue
                if ev.get("triggers"):
                    has_trigger.add(etype)
                for arg in ev.get("arguments") or []:
                    if arg.get("role"):
                        has_role.add(etype)
    return sorted(has_trigger - has_role)


def _corpus_presets() -> List[Dict[str, Any]]:
    sys.path.insert(0, str(scripts_root()))
    try:
        from infer_file import _derive_schema
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    data_dir = data_root()
    if not data_dir.is_dir():
        return out
    for train in sorted(data_dir.glob("*.train.jsonl")):
        name = train.name[: -len(".train.jsonl")]
        try:
            si = _raw_to_schema_input(_derive_schema(str(train)))
        except Exception:
            continue
        # Recover trigger-only event corpora (no role-bearing events, so
        # `_derive_schema` returns nothing) as entity labels.
        if "events" not in si:
            trig = _trigger_only_event_types(train)
            if trig:
                si["entities"] = sorted(set(si.get("entities") or []) | set(trig))
        if si:
            out.append({"name": f"corpus: {name}", "schema": si, "source": "corpus"})
    return out


def list_presets() -> List[Dict[str, Any]]:
    presets = [{"name": name, "schema": schema, "source": "builtin"}
               for name, schema in BUILTIN.items()]
    presets.extend(_corpus_presets())
    return presets
