"""Replay a model's training label map on an inference schema.

Labels are an INPUT to GLiNER2: a model trained on ``Person`` is asked a different question
by ``person``. Training rewrites corpus labels through ``labels_file`` (rollup, then map);
the checkpoint stores that transform as ``config.label_map`` and this module applies the
same transform to a user's schema, so the user's spelling reaches the model as the one it
learned.

A checkpoint trained under a label STYLE (``config.label_style``, LABEL_STYLE_SPEC.md) also had every
label styled after the map, so the same style is applied here -- including to labels the map never
saw -- and structure names and field names are mapped too, as training's ``_transform_structures``
does. Without a style, structures are left alone as before, so older checkpoints are unchanged.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from gliner2.inference.label_style import styler


def label_fn(block: Mapping[str, Any], style: Optional[Callable[[str], str]] = None) -> Callable[[str], str]:
    """Roll a label up to its first ``separator`` segment (if ``rollup``), remap it, then style it."""
    rollup = bool(block.get("rollup", False))
    separator = block.get("separator", ".")
    mapping = block.get("map") or {}

    def fn(label: str) -> str:
        if rollup and separator in label:
            label = label.split(separator, 1)[0]
        label = mapping.get(label, label)
        return style(label) if style else label
    return fn


def _labels(value, fn):
    """Map a list (order-preserving dedup) or the keys of a dict.

    A list item may itself be a ``{label: spec}`` dict -- the form eval's
    ``_schema_from_gold`` gives relations (``[{"founded_by": {"head": "", "tail": ""}}]``);
    its key is mapped and its spec kept. Calling ``fn`` on the dict raised TypeError
    (unhashable) for every relation corpus under ``infer.py --gold-schema``.
    """
    if isinstance(value, dict):
        return {fn(k): v for k, v in value.items()}
    out, seen = [], set()
    for label in value:
        new = {fn(k): v for k, v in label.items()} if isinstance(label, dict) else fn(label)
        key = tuple(new) if isinstance(new, dict) else new
        if key not in seen:
            seen.add(key)
            out.append(new)
    return out


def _event(spec, fn):
    if isinstance(spec, list):
        return _labels(spec, fn)
    spec = dict(spec)
    for key in ("roles", "exclusive_roles"):
        if key in spec:
            spec[key] = _labels(spec[key], fn)
    if "role_descriptions" in spec:
        spec["role_descriptions"] = _labels(spec["role_descriptions"], fn)
    return spec


def _structures(structures: Mapping[str, Any], fn) -> Dict[str, Any]:
    """Map structure names, their field names, and the ``anchor`` that names a field."""
    out = {}
    for name, spec in structures.items():
        spec = copy.deepcopy(spec)
        if isinstance(spec, dict):
            if isinstance(spec.get("fields"), list):
                spec["fields"] = [dict(f, name=fn(f["name"])) if isinstance(f, dict) and "name" in f
                                  else fn(f) if isinstance(f, str) else f for f in spec["fields"]]
            if isinstance(spec.get("anchor"), str):
                spec["anchor"] = fn(spec["anchor"])
        out[fn(name)] = spec
    return out


def apply_label_map(schema: Mapping[str, Any], label_map: Optional[Mapping[str, Any]],
                    label_style: Optional[Mapping[str, Any]] = None
                    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    """Return ``(mapped schema, {category: {sent: model spelling}})`` for changed labels.

    ``label_style`` is the checkpoint's ``config.label_style``; None keeps the map-only behaviour.
    """
    out = copy.deepcopy(dict(schema))
    applied: Dict[str, Dict[str, str]] = {}
    style = styler(label_style)
    if not label_map and not style:
        return out, applied
    label_map = label_map or {}

    def tracked(category):
        fn = label_fn(label_map.get(category) or {}, style)

        def record(label):
            new = fn(label)
            if new != label:
                applied.setdefault(category, {})[label] = new
            return new
        return record

    if out.get("entities") is not None:
        out["entities"] = _labels(out["entities"], tracked("entities"))
    if out.get("relations") is not None:
        out["relations"] = _labels(out["relations"], tracked("relations"))
    if out.get("events") is not None:
        ev = tracked("events")
        out["events"] = {ev(t): _event(spec, ev) for t, spec in out["events"].items()}
    if out.get("classifications") is not None:
        cls = tracked("classifications")
        out["classifications"] = [dict(c, labels=_labels(c["labels"], cls)) if "labels" in c else c
                                  for c in out["classifications"]]
    if style and isinstance(out.get("structures"), dict):
        out["structures"] = _structures(out["structures"], tracked("structures"))
    return out, applied
