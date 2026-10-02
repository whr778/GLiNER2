"""Compare a corpus's labels with what the base model was trained to read.

Labels are an INPUT to GLiNER2, so a fine-tune must present the base's own spelling of a
concept it already knows. Five outcomes per label, and only the first two are automatic:

  known       already a spelling the base trained on
  mapped      the base's own `label_map` rewrites it (replayed at train and inference)
  fold_match  same letters as a known label up to case/punctuation -- PROPOSED, a human
              confirms it by reading surfaces (the standing rule: never merge on spelling alone)
  new         no counterpart in a category the checkpoint lists IN FULL; trains as new
  unlisted    no counterpart, but the category is open-vocabulary in the checkpoint
              (`default_schema.open_vocab`), so it has no full inventory -- `label_map` holds
              only the labels training RENAMED -- and "not found" does not mean "new"

Bases with no inventory (e.g. fastino) are open-vocabulary: there is nothing to match, so the
review instead flags CODE-LIKE labels (`PER.Individual`, `ORG-AFF`) that a zero-shot model
reads as opaque strings, as candidates for a natural-language name.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/train"))
from build_label_maps import fold  # noqa: E402

CATEGORY = {"entities": "entities", "relations": "relations",
            "event_types": "events", "roles": "events"}
CODE_LIKE = re.compile(r"[._]|^[A-Z0-9-]{2,}$|[a-z][A-Z]")


def inventory(model_info: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Spellings the base trained on, per label_map category."""
    inv: Dict[str, Set[str]] = {"entities": set(), "relations": set(), "events": set()}
    for cat, block in (model_info.get("label_map") or {}).items():
        if cat in inv:
            inv[cat].update((block.get("map") or {}).values())
    schema = model_info.get("default_schema") or {}
    for etype, roles in (schema.get("events") or {}).items():
        inv["events"].add(etype)
        inv["events"].update(roles if isinstance(roles, list) else roles.get("roles", []))
    inv["relations"].update(schema.get("relations") or [])
    inv["entities"].update(schema.get("entities") or [])
    return inv


def review(corpus_labels: Dict[str, Dict[str, int]], model_info: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Classify every corpus label; returns {corpus category: [rows]}."""
    inv = inventory(model_info)
    lmap = model_info.get("label_map") or {}
    out: Dict[str, List[Dict]] = {}
    for corpus_cat, cat in CATEGORY.items():
        known = inv[cat]
        by_fold = {}
        for k in known:
            by_fold.setdefault(fold(k), k)
        mapping = (lmap.get(cat) or {}).get("map") or {}
        rows = []
        for label, uses in sorted(corpus_labels.get(corpus_cat, {}).items(), key=lambda x: -x[1]):
            row = {"label": label, "uses": uses}
            if not model_info.get("has_label_inventory"):
                row["status"] = "open_vocab_code_like" if CODE_LIKE.search(label) else "open_vocab"
            elif label in known:
                row["status"] = "known"
            elif label in mapping:
                row.update(status="mapped", target=mapping[label])
            elif fold(label) in by_fold:
                row.update(status="fold_match", target=by_fold[fold(label)])
            elif cat in (model_info.get("default_schema") or {}).get("open_vocab", []):
                row["status"] = "unlisted"
            else:
                row["status"] = "new"
            rows.append(row)
        out[corpus_cat] = rows
    return out
