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
from gliner2.inference.schema import expand_entity_wildcards


def label_fn(block: Mapping[str, Any], style: Optional[Callable[[str], str]] = None) -> Callable[[str], str]:
    """Roll a label up to its first ``separator`` segment (if ``rollup``), remap it, then style it."""
    rollup = bool(block.get("rollup", False))
    separator = block.get("separator", ".")
    mapping = block.get("map") or {}

    def fn(label: str) -> Optional[str]:
        if rollup and separator in label:
            label = label.split(separator, 1)[0]
        label = mapping.get(label, label)
        if label is None:  # mapped to null: the training map dropped this label
            return None
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
        return {fn(k): v for k, v in value.items() if fn(k) is not None}
    out, seen = [], set()
    for label in value:
        new = ({fn(k): v for k, v in label.items() if fn(k) is not None} if isinstance(label, dict)
               else fn(label))
        if not new:  # dropped by a null mapping
            continue
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


def _json_structures(json_structures: list, fn) -> list:
    """Map built ``json_structures`` entries (``[{struct_name: {field: spec}}]``)."""
    out = []
    for entry in json_structures:
        if not isinstance(entry, Mapping):
            out.append(entry)
            continue
        renamed = {}
        for name, body in entry.items():
            new_name = fn(name)
            if isinstance(body, Mapping):
                fields = renamed.setdefault(new_name, type(body)())
                for field, val in body.items():
                    new_field = fn(field)
                    if new_field not in fields:
                        fields[new_field] = copy.deepcopy(val)
            elif new_name not in renamed:
                renamed[new_name] = copy.deepcopy(body)
        out.append(renamed)
    return out


def _nested_struct_field_map(mapping: Mapping[str, Any], fn) -> Dict[str, Any]:
    """Map ``{struct_name: {field_name: value}}`` (used by ``json_descriptions`` and ``field_dtypes``)."""
    out: Dict[str, Any] = {}
    for name, body in mapping.items():
        new_name = fn(name)
        if isinstance(body, Mapping):
            sub = out.setdefault(new_name, {})
            for field, val in body.items():
                new_field = fn(field)
                if new_field not in sub:
                    sub[new_field] = copy.deepcopy(val)
        elif new_name not in out:
            out[new_name] = copy.deepcopy(body)
    return out


def _record_metadata(meta: Mapping[str, Any], fn) -> Dict[str, Any]:
    """Map ``record_metadata`` structure names, ``anchor``, and per-field ``fields`` keys."""
    out: Dict[str, Any] = {}
    for name, block in meta.items():
        new_name = fn(name)
        if isinstance(block, Mapping):
            new_block = copy.deepcopy(dict(block))
            if isinstance(new_block.get("anchor"), str):
                new_block["anchor"] = fn(new_block["anchor"])
            if isinstance(new_block.get("fields"), Mapping):
                new_block["fields"] = {fn(fk): fv for fk, fv in new_block["fields"].items()}
            out.setdefault(new_name, new_block)
        else:
            out.setdefault(new_name, copy.deepcopy(block))
    return out


def apply_label_map(schema: Mapping[str, Any], label_map: Optional[Mapping[str, Any]],
                    label_style: Optional[Mapping[str, Any]] = None
                    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    """Return ``(mapped schema, {category: {sent: model spelling}})`` for changed labels.

    ``label_style`` is the checkpoint's ``config.label_style``; None keeps the map-only behaviour.
    """
    out = copy.deepcopy(dict(schema))
    if out.get("entities") is not None:
        out["entities"] = expand_entity_wildcards(out["entities"])
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
        ent = tracked("entities")
        out["entities"] = _labels(out["entities"], ent)
        if isinstance(out.get("entity_descriptions"), Mapping):
            out["entity_descriptions"] = _labels(dict(out["entity_descriptions"]), ent)
    if out.get("relations") is not None:
        rel = tracked("relations")
        out["relations"] = _labels(out["relations"], rel)
        if isinstance(out.get("relation_descriptions"), Mapping):
            out["relation_descriptions"] = _labels(dict(out["relation_descriptions"]), rel)
    orig_event_names = set(schema.get("events") or {})
    if out.get("events") is not None:
        ev = tracked("events")
        out["events"] = {ev(t): _event(spec, ev) for t, spec in out["events"].items()}
        if isinstance(out.get("record_metadata"), Mapping) and orig_event_names:
            ev_meta = {k: v for k, v in out["record_metadata"].items() if k in orig_event_names}
            if ev_meta:
                mapped_ev_meta = {}
                for k, block in ev_meta.items():
                    nb = copy.deepcopy(dict(block)) if isinstance(block, Mapping) else copy.deepcopy(block)
                    if isinstance(nb, dict) and isinstance(nb.get("fields"), Mapping):
                        nb["fields"] = {ev(fk): fv for fk, fv in nb["fields"].items()}
                    mapped_ev_meta[ev(k)] = nb
                rest = {k: v for k, v in out["record_metadata"].items() if k not in orig_event_names}
                out["record_metadata"] = {**rest, **mapped_ev_meta}
    if out.get("classifications") is not None:
        cls = tracked("classifications")
        mapped_cls = []
        for c in out["classifications"]:
            if not isinstance(c, Mapping):
                mapped_cls.append(c)
                continue
            nc = dict(c)
            if "labels" in nc:
                nc["labels"] = _labels(nc["labels"], cls)
            if isinstance(nc.get("label_descriptions"), Mapping):
                nc["label_descriptions"] = _labels(dict(nc["label_descriptions"]), cls)
            mapped_cls.append(nc)
        out["classifications"] = mapped_cls
    if style:
        st = tracked("structures")
        if isinstance(out.get("structures"), dict):
            out["structures"] = _structures(out["structures"], st)
        if isinstance(out.get("json_structures"), list):
            out["json_structures"] = _json_structures(out["json_structures"], st)
        if isinstance(out.get("json_descriptions"), Mapping):
            out["json_descriptions"] = _nested_struct_field_map(out["json_descriptions"], st)
        if isinstance(out.get("field_dtypes"), Mapping):
            out["field_dtypes"] = _nested_struct_field_map(out["field_dtypes"], st)
        if isinstance(out.get("record_metadata"), Mapping):
            mapped_event_names = set(out.get("events") or {})
            struct_meta = {k: v for k, v in out["record_metadata"].items() if k not in mapped_event_names}
            keep_meta = {k: v for k, v in out["record_metadata"].items() if k in mapped_event_names}
            out["record_metadata"] = {**keep_meta, **_record_metadata(struct_meta, st)}
    return out, applied
