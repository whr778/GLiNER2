"""Gold vs prediction for one ``tools/infer.py --output`` row, in the viewer's result shape.

Gold stores bare surfaces; the viewer draws offsets. Surfaces are placed exactly as training
places them (``processor._find_sublist`` over the lowercased word split: every occurrence), and
every span carries ``match`` under the scorer's STRICT keys (``eval_metrics``), so a gold rim
agrees with the strict F1 the file scores.
"""

from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import diffs
from gliner2.processing.word_splitter import WhitespaceTokenSplitter
from gliner2.training import eval_metrics as M

_split = WhitespaceTokenSplitter()


def read_rows(path: str) -> List[Dict[str, Any]]:
    """The rows of an infer.py output file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=4)
def _cached_rows(path: str, mtime: float) -> List[Dict[str, Any]]:
    return read_rows(path)


def rows_of(path: str) -> List[Dict[str, Any]]:
    """The file's rows, re-read when it changes on disk."""
    return _cached_rows(path, os.path.getmtime(path))


def summary(path: str) -> Dict[str, Any]:
    """Per record: has gold, strict F1 per head -- what the pager shows and sorts by."""
    rows = rows_of(path)
    return {"path": path, "count": len(rows), "gold_key": gold_key(rows[0]) if rows else None,
            "records": [{"i": i, "chars": len(r["input"]), "has_gold": bool(r.get(gold_key(r))),
                         "f1": _f1(r.get(gold_key(r)) or {}, r.get("output") or {})}
                        for i, r in enumerate(rows)]}


def record(path: str, i: int) -> Dict[str, Any]:
    """Record ``i`` of the file as ``record_view``; the cached row is never mutated."""
    return dict(record_view(copy.deepcopy(rows_of(path)[i])), i=i)


_verdicts: Dict[Tuple[str, float, int], Dict[str, Any] | None] = {}


def _infer_output(path: str, mtime: float, size: int) -> Dict[str, Any] | None:
    """The picker entry when the file's first line is an infer.py row (input, output and gold), else None.
    Cached by (path, mtime, size): only new or changed files are read."""
    key = (path, mtime, size)
    if key not in _verdicts:
        try:
            with open(path, encoding="utf-8") as f:
                first = json.loads(f.readline() or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            first = {}
        _verdicts[key] = ({"path": path, "mtime": mtime, "size": size, "gold_mapped": "gold_mapped" in first,
                           "spans": '"start"' in json.dumps(first.get("output"))}
                          if isinstance(first, dict) and {"input", "output", "gold"} <= first.keys() else None)
    return _verdicts[key]


def list_files(root: str) -> List[Dict[str, Any]]:
    """Every infer.py output file under ``root``, newest first."""
    out = []
    for p in Path(root).rglob("*.jsonl"):
        st = p.stat()
        entry = _infer_output(str(p), st.st_mtime, st.st_size)
        if entry:
            out.append(dict(entry, name=str(p.relative_to(root))))
    return sorted(out, key=lambda e: -e["mtime"])


def gold_key(row: Dict[str, Any]) -> str:
    """``gold_mapped`` (gold in the model's spellings) when present, as the scorer reads it."""
    return "gold_mapped" if "gold_mapped" in row else "gold"


def locate(text: str, surface: str) -> List[Tuple[int, int]]:
    """Character spans of every token-run match of ``surface``, as training finds them."""
    words = list(_split(text, lower=True))
    sub = [t for t, _, _ in _split(surface, lower=True)]
    n = len(sub)
    if not n:
        return []
    return [(words[i][1], words[i + n - 1][2]) for i in range(len(words) - n + 1)
            if [t for t, _, _ in words[i:i + n]] == sub]


def _surface(x: Any) -> str:
    return (x.get("text") if isinstance(x, dict) else x or "").strip()


def _spans(text: str, surface: str, missed: List[str]) -> List[Dict[str, Any]]:
    found = locate(text, surface)
    if not found:
        missed.append(surface)
    return [{"text": text[s:e], "start": s, "end": e} for s, e in found]


def gold_entities(text: str, gold: Dict[str, Any], missed: List[str]) -> Dict[str, list]:
    return {label: [sp for s in surfaces if isinstance(s, str) and s.strip()
                    for sp in _spans(text, s.strip(), missed)]
            for label, surfaces in (gold.get("entities") or {}).items()}


def gold_events(text: str, gold: Dict[str, Any], missed: List[str]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for ev in gold.get("events") or []:
        triggers = [sp for t in ev.get("triggers") or [] if isinstance(t, str) and t.strip()
                    for sp in _spans(text, t.strip(), missed)]
        args = [{"role": a["role"], "entity": sp} for a in ev.get("arguments") or []
                for sp in _spans(text, _surface(a.get("entity")), missed)]
        out.setdefault(ev["event_type"].strip(), []).append(
            {"triggers": triggers, "arguments": args,
             "trigger_key": sorted(t.strip() for t in ev.get("triggers") or [] if isinstance(t, str) and t.strip())})
    return out


def gold_relations(text: str, gold: Dict[str, Any], missed: List[str]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for rel in gold.get("relations") or []:
        for name, f in rel.items():
            for h in _spans(text, _surface(f.get("head")), missed):
                for t in _spans(text, _surface(f.get("tail")), missed):
                    out.setdefault(name, []).append({"head": h, "tail": t})
    return out


def _mark_entities(block: Dict[str, list], other: set) -> None:
    for label, spans in block.items():
        for sp in spans:
            if isinstance(sp, dict):
                sp["match"] = (label, _surface(sp)) in other


def _mark_events(block: Dict[str, list], triggers: set, args: set) -> None:
    for etype, mentions in block.items():
        for m in mentions:
            key = tuple(m.pop("trigger_key", None) or sorted(_surface(t) for t in m.get("triggers") or [] if _surface(t)))
            for t in m.get("triggers") or []:
                if isinstance(t, dict):
                    t["match"] = (etype, _surface(t)) in triggers
            for a in m.get("arguments") or []:
                if isinstance(a.get("entity"), dict):
                    a["entity"]["match"] = (etype, a["role"].strip(), _surface(a["entity"]), key) in args


def _mark_relations(block: Dict[str, list], other: set) -> None:
    for name, items in block.items():
        for r in items:
            if isinstance(r, dict) and isinstance(r.get("head"), dict):
                hit = (name, _surface(r["head"]), _surface(r.get("tail"))) in other
                r["head"]["match"] = hit
                if isinstance(r.get("tail"), dict):
                    r["tail"]["match"] = hit


def _classifications(gold: Dict[str, Any], pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits = M._pred_classification_pairs(pred, gold)
    rows = []
    for c in gold.get("classifications") or []:
        p = pred.get(c["task"])
        p = p.get("label") if isinstance(p, dict) else p
        gl = c.get("true_label")
        gl = gl if isinstance(gl, list) else [gl]
        rows.append({"task": c["task"], "gold": gl, "pred": p,
                     "match": all((c["task"], g.strip()) in hits for g in gl if isinstance(g, str))})
    return rows


def record_view(row: Dict[str, Any]) -> Dict[str, Any]:
    """One row as {text, pred, gold, classifications, unlocated, f1, diffs}: both sides in the
    prediction shape, every span flagged ``match`` under the strict keys, every unmatched span
    given its ``why`` (diffs.py)."""
    text, pred, gold = row["input"], row.get("output") or {}, row.get(gold_key(row)) or {}
    missed: List[str] = []
    g = {"entities": gold_entities(text, gold, missed),
         "event_extraction": gold_events(text, gold, missed),
         "relation_extraction": gold_relations(text, gold, missed)}
    _mark_entities(pred.get("entities") or {}, M._gold_entity_set(gold))
    _mark_entities(g["entities"], M._pred_entity_set(pred))
    _mark_events(pred.get("event_extraction") or {}, M._gold_event_trigger_set(gold), M._gold_event_argument_set(gold))
    _mark_events(g["event_extraction"], M._pred_event_trigger_set(pred), M._pred_event_argument_set(pred))
    _mark_relations(pred.get("relation_extraction") or {}, M._gold_relation_set(gold))
    _mark_relations(g["relation_extraction"], M._pred_relation_set(pred))
    view = {"text": text, "pred": pred, "gold": g, "has_gold": bool(gold),
            "classifications": _classifications(gold, pred),
            "unlocated": sorted(set(missed)), "f1": _f1(gold, pred)}
    view["diffs"] = diffs.annotate(view) if gold else []
    return view


def _f1(gold: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, float]:
    """Strict F1 of this one record per head, from the scorer's own sets."""
    pairs = {"entity": (M._gold_entity_set(gold), M._pred_entity_set(pred)),
             "event_trigger": (M._gold_event_trigger_set(gold), M._pred_event_trigger_set(pred)),
             "event_argument": (M._gold_event_argument_set(gold), M._pred_event_argument_set(pred)),
             "relation": (M._gold_relation_set(gold), M._pred_relation_set(pred))}
    out = {}
    for head, (g, p) in pairs.items():
        if g or p:
            tp = len(g & p)
            out[head] = round(2 * tp / (len(g) + len(p)), 4)
    return out
