"""Schema presets for the viewer: a few built-ins plus schemas auto-derived
from any ``data/<name>.train.jsonl`` present in the repo (reusing
``scripts/infer_file.py::_derive_schema``), so a user can one-click the exact
ontology a trained model expects."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[2]

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
                "fields": [
                    {"name": "name", "dtype": "str"},
                    {"name": "email", "dtype": "str"},
                    {"name": "phone", "dtype": "str"},
                    {"name": "organization", "dtype": "str"},
                ],
            },
        },
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


def _corpus_presets() -> List[Dict[str, Any]]:
    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        from infer_file import _derive_schema
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    data_dir = _REPO / "data"
    if not data_dir.is_dir():
        return out
    for train in sorted(data_dir.glob("*.train.jsonl")):
        name = train.name[: -len(".train.jsonl")]
        try:
            si = _raw_to_schema_input(_derive_schema(str(train)))
        except Exception:
            continue
        if si:
            out.append({"name": f"corpus: {name}", "schema": si, "source": "corpus"})
    return out


def list_presets() -> List[Dict[str, Any]]:
    presets = [{"name": name, "schema": schema, "source": "builtin"}
               for name, schema in BUILTIN.items()]
    presets.extend(_corpus_presets())
    return presets
