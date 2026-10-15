"""Link every event argument to an entity by span: a decode-time post-process, no training.

No head links an argument to an entity, but both are spans of one document, so the span IS the
identity. Each distinct entity span gets an ``id`` (``e1``, ``e2``, ... in text order), shared by
every label on that span -- one span often carries several (``guns`` = WEA, VEH, COM). Each
argument gets the ``entity_id`` of the entity on its exact span.

An argument with no entity on its span gets a DERIVED entity, listed under ``derived_entities``,
never under ``entities``, so the scorer never sees it and no metric moves. Its ``types`` come
from the ``(event_type, role) -> entity types`` map (``tools/data/build_role_type_map.py``),
empty when the map has no entry. Linking is exact-span only: a boundary overlap is not a link.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

Span = Tuple[int, int]


def load_role_types(path: str) -> Dict[str, Dict[str, List[str]]]:
    """The ``role_types`` block of a build_role_type_map.py output."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["role_types"]


def _key(sp: Dict[str, Any]) -> Span:
    return sp["start"], sp["end"]


def link_arguments(pred: Dict[str, Any], role_types: Optional[Dict[str, Dict[str, List[str]]]] = None
                   ) -> Dict[str, Any]:
    """Add ``id`` to entity spans, ``entity_id`` to arguments, and ``derived_entities``, in place.
    Needs span output (``include_spans``)."""
    role_types = role_types or {}
    ents = [sp for spans in (pred.get("entities") or {}).values() for sp in spans or []]
    args = [(etype, a) for etype, ms in (pred.get("event_extraction") or {}).items()
            for m in ms or [] for a in m.get("arguments") or []]
    spans = sorted({_key(sp) for sp in ents} | {_key(a["entity"]) for _, a in args})
    ids = {s: f"e{i + 1}" for i, s in enumerate(spans)}
    for sp in ents:
        sp["id"] = ids[_key(sp)]
    entity_spans = {_key(sp) for sp in ents}
    derived: Dict[Span, Dict[str, Any]] = {}
    for etype, a in args:
        k = _key(a["entity"])
        a["entity_id"] = ids[k]
        if k in entity_spans:
            continue
        d = derived.setdefault(k, {"id": ids[k], "text": a["entity"]["text"], "start": k[0], "end": k[1],
                                   "types": [], "roles": []})
        for t in role_types.get(etype, {}).get(a["role"], []):
            if t not in d["types"]:
                d["types"].append(t)
        if [etype, a["role"]] not in d["roles"]:
            d["roles"].append([etype, a["role"]])
    pred["derived_entities"] = [derived[k] for k in sorted(derived)]
    return pred
