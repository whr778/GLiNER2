"""A small persistent model registry for the viewer.

Merges three sources into one list the frontend combobox / manager uses:
- the server default model,
- auto-discovered local checkpoints (``out/**/best`` in the repo),
- user-saved entries persisted to ``models.json`` (next to this file).

Only user-saved entries can be removed; default/discovered are always offered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
_STORE = Path(__file__).resolve().parent / "models.json"


def _load_saved() -> List[Dict[str, str]]:
    if _STORE.is_file():
        try:
            data = json.loads(_STORE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [
                    {"path": m["path"], "label": m.get("label") or m["path"]}
                    for m in data
                    if isinstance(m, dict) and m.get("path")
                ]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    return []


def _save(saved: List[Dict[str, str]]) -> None:
    _STORE.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")


def _discovered() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    root = _REPO / "out"
    if not root.is_dir():
        return out
    for best in sorted(root.glob("**/best")):
        if (best / "model.safetensors").is_file() or (best / "config.json").is_file():
            out.append({"path": str(best), "label": best.parent.name, "source": "discovered"})
    return out


def list_models(default_model: str) -> List[Dict[str, str]]:
    """Merged, de-duplicated model list (default -> saved -> discovered)."""
    entries: List[Dict[str, str]] = [
        {"path": default_model, "label": default_model, "source": "default"}
    ]
    entries += [{**m, "source": "saved"} for m in _load_saved()]
    entries += _discovered()

    result: List[Dict[str, str]] = []
    seen: set = set()
    for e in entries:
        if e["path"] in seen:
            continue
        seen.add(e["path"])
        result.append(e)
    return result


def add_model(path: str, label: Optional[str]) -> None:
    path = path.strip()
    if not path:
        raise ValueError("path is empty")
    saved = _load_saved()
    if not any(m["path"] == path for m in saved):
        saved.append({"path": path, "label": (label or path).strip()})
        _save(saved)


def remove_model(path: str) -> None:
    _save([m for m in _load_saved() if m["path"] != path.strip()])
