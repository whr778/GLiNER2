"""WHY a span does not match: every unmatched gold or predicted item named by its reason.

Works on ``gold.record_view``'s two placed sides (both carry offsets and the strict ``match``).
An unmatched item is paired with an unmatched item on the other side that overlaps it:

    span        same label, different extent         (gold "a complex ransomware virus" / pred "complex ...")
    label/type  same extent, different label or event type
    role        argument: same extent and type, different role
    attachment  argument: same extent, type and role, under a different trigger mention
    direction   relation: same name, head and tail swapped
    surface     same extent and label, but the gold STRING differs (case, spacing) from the text
    missed      gold with nothing on the other side;  spurious: a prediction with nothing in gold

Each span gets a short ``why`` (the tooltip); ``rows`` groups identical rows across occurrences
for the Differences table. ``eid``s follow the frontend's ``collectMarks`` numbering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

LAYER_KEYS = {"entities": "entities", "events": "event_extraction", "relations": "relation_extraction"}


def _items(side: Dict[str, Any], is_gold: bool) -> List[Dict[str, Any]]:
    """Flatten one side into scored items, each pointing at the span dicts it annotates."""
    pre, out = ("g-" if is_gold else ""), []
    for label, spans in (side.get("entities") or {}).items():
        for sp in spans or []:
            if isinstance(sp, dict) and "start" in sp:
                out.append({"layer": "entities", "kind": "entity", "label": label, "span": sp, "spans": [sp]})
    n = 0
    for etype, mentions in (side.get("event_extraction") or {}).items():
        for mi, m in enumerate(mentions or []):
            eid, idx = f"{pre}evt-{n}", mi + 1
            n += 1
            trig = [t for t in m.get("triggers") or [] if isinstance(t, dict) and "start" in t]
            for t in trig:
                out.append({"layer": "events", "kind": "trigger", "label": etype, "span": t, "spans": [t],
                            "eid": eid, "idx": idx})
            for a in m.get("arguments") or []:
                e = a.get("entity")
                if isinstance(e, dict) and "start" in e:
                    out.append({"layer": "events", "kind": "argument", "label": etype, "role": a["role"],
                                "span": e, "spans": [e], "eid": eid, "idx": idx,
                                "trig": sorted({t.get("text", "") for t in trig})})
    n = 0
    for name, insts in (side.get("relation_extraction") or {}).items():
        for r in insts or []:
            eid = f"{pre}rel-{n}"
            n += 1
            h, t = (r if isinstance(r, list) else [r.get("head"), r.get("tail")])[:2]
            if isinstance(h, dict) and isinstance(t, dict) and "start" in h and "start" in t:
                out.append({"layer": "relations", "kind": "relation", "label": name, "span": h, "tail": t,
                            "spans": [h, t], "eid": eid})
    return out


def _rng(sp: Dict[str, Any]) -> tuple:
    return sp["start"], sp["end"]


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def _reason(g: Dict[str, Any], p: Dict[str, Any]) -> Optional[str]:
    """The reason gold item ``g`` and prediction ``p`` differ, or None when they are unrelated."""
    if g["layer"] != p["layer"]:
        return None
    if g["kind"] == "relation":
        same = _rng(g["span"]) == _rng(p["span"]) and _rng(g["tail"]) == _rng(p["tail"])
        swapped = _rng(g["span"]) == _rng(p["tail"]) and _rng(g["tail"]) == _rng(p["span"])
        if same:
            return "surface" if g["label"] == p["label"] else "label"
        if swapped and g["label"] == p["label"]:
            return "direction"
        if g["label"] == p["label"] and _overlap(g["span"], p["span"]) and _overlap(g["tail"], p["tail"]):
            return "span"
        return None
    if g["kind"] != p["kind"] or not _overlap(g["span"], p["span"]):
        return None
    exact = _rng(g["span"]) == _rng(p["span"])
    word = "type" if g["layer"] == "events" else "label"
    if g["kind"] == "argument":
        if g["label"] != p["label"]:
            return word if exact else None
        if g["role"] != p["role"]:
            return "role" if exact else "span+role"
        if not exact:
            return "span"
        return "attachment" if g["trig"] != p["trig"] else "surface"
    if g["label"] != p["label"]:
        return word if exact else f"span+{word}"
    return "surface" if exact else "span"


def _desc(it: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if it is None:
        return None
    d = {"label": it["label"], "text": it["span"].get("text", ""), "start": it["span"]["start"],
         "end": it["span"]["end"], "eid": it.get("eid")}
    if it["kind"] == "argument":
        d.update(role=it["role"], trig=it["trig"])
    if it["kind"] == "trigger":
        d.update(idx=it["idx"])
    if it["kind"] == "relation":
        d.update(tail=it["tail"].get("text", ""))
    return d


def _short(d: Dict[str, Any]) -> str:
    lab = d["label"] + (f" / {d['role']}" if "role" in d else "")
    txt = f"\"{d['text']}\"" + (f" -> \"{d['tail']}\"" if "tail" in d else "")
    trig = f" under trigger {', '.join(d['trig']) or '-'}" if "trig" in d else ""
    return f"{txt} ({lab}){trig}"


def _why(reason: str, g: Optional[Dict[str, Any]], p: Optional[Dict[str, Any]]) -> str:
    if g is None:
        return f"spurious: predicted {_short(p)}, nothing in gold here"
    if p is None:
        return f"missed: gold {_short(g)}, nothing predicted here"
    return f"{reason} differs: gold {_short(g)} vs pred {_short(p)}"


def annotate(view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Write ``why`` onto every unmatched span of ``view``'s two sides; return the grouped rows."""
    gold = [i for i in _items(view["gold"], True) if not i["span"].get("match")]
    pred = [i for i in _items(view["pred"], False) if not i["span"].get("match")]
    rows, paired = [], set()
    for g in gold:
        hits = [(r, p) for p in pred if (r := _reason(g, p))]
        for r, p in hits or [("missed", None)]:
            if p is not None:
                paired.add(id(p))
            rows.append((r, g, p))
    rows += [("spurious", None, p) for p in pred if id(p) not in paired]
    grouped: Dict[tuple, Dict[str, Any]] = {}
    for r, g, p in rows:
        gd, pd = _desc(g), _desc(p)
        why = _why(r, gd, pd)
        for it in (g, p):
            for sp in (it or {}).get("spans", []):
                sp.setdefault("why", [])
                if why not in sp["why"]:
                    sp["why"].append(why)
        key = (r, (g or p)["layer"], why)
        row = grouped.get(key)
        if row is None:
            grouped[key] = {"reason": r, "layer": (g or p)["layer"], "why": why, "gold": gd, "pred": pd, "count": 1}
        else:
            row["count"] += 1
    order = {"entities": 0, "events": 1, "relations": 2}
    return sorted(grouped.values(), key=lambda x: (order[x["layer"]], x["reason"] in ("missed", "spurious"),
                                                   (x["gold"] or x["pred"])["start"]))
