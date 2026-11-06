"""A small persistent model registry for the viewer.

Merges three sources into one list the frontend combobox / manager uses:
- the server default model,
- auto-discovered local checkpoints (``**/best`` under the configured
  ``models_root``; see ``config.py``),
- user-saved entries persisted to ``models.json`` (next to this file).

Only user-saved entries can be removed; default/discovered are always offered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from config import REPO, models_root

_STORE = Path(__file__).resolve().parent / "models.json"
_FALLBACK_STORE = Path("/tmp/gliner2-viewer-models.json")


def resolve_local_checkpoint(path: str) -> str:
    """Resolve a local checkpoint path if it exists on disk, else return stripped string.

    Expands ``~``, checks relative to CWD, ``REPO``, and ``models_root()``, unwraps a
    direct file path (``.../config.json`` or ``.../model.safetensors``) to its parent
    directory, and descends into ``best/`` when given a run directory whose
    ``config.json`` lives under ``best/``.
    """
    raw = (path or "").strip()
    if not raw or "@" in raw:
        return raw
    expanded = Path(raw).expanduser()
    candidates = [expanded]
    if not expanded.is_absolute():
        candidates.extend([REPO / expanded, models_root() / expanded])
    for cand in candidates:
        try:
            if cand.name in ("config.json", "model.safetensors", "pytorch_model.bin") and (
                cand.is_file() or cand.parent.is_dir()
            ):
                cand = cand.parent
            if cand.is_dir():
                if not (cand / "config.json").is_file() and (cand / "best" / "config.json").is_file():
                    cand = cand / "best"
                return str(cand.resolve())
        except OSError:
            continue
    return str(expanded) if raw.startswith("~") else raw


def _looks_like_local_path(raw: str) -> bool:
    """True when ``raw`` is clearly a filesystem path rather than a Hub ``owner/repo`` id."""
    if not raw or "@" in raw:
        return False
    if raw.startswith(("/", "~", "./", "../")) or "\\" in raw:
        return True
    parts = [p for p in raw.split("/") if p]
    if len(parts) != 2:
        return True
    if parts[0] in ("out", "models", "checkpoints", "data", "tmp", "var", "opt", "home", "Users", "Volumes"):
        return True
    if raw.endswith((".safetensors", ".bin", ".json", "/best", "/final")):
        return True
    return False


def _store_path() -> Path:
    """Path to ``models.json``. Falls back to ``_FALLBACK_STORE`` if ``_STORE`` is on a read-only mount."""
    if _STORE.exists():
        if os.access(_STORE, os.W_OK):
            return _STORE
    elif os.access(_STORE.parent, os.W_OK):
        return _STORE
    return _FALLBACK_STORE


def _load_saved() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen_ids: set = set()
    paths = [_STORE]
    if _store_path() == _FALLBACK_STORE and _FALLBACK_STORE not in paths:
        paths.append(_FALLBACK_STORE)
    for path in paths:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for m in data:
                    if isinstance(m, dict) and m.get("path"):
                        entry = _saved_entry(m)
                        eid = entry_id(entry)
                        if eid not in seen_ids:
                            seen_ids.add(eid)
                            out.append(entry)
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    return out


def _saved_entry(raw: Dict[str, str]) -> Dict[str, str]:
    entry = {"path": raw["path"], "label": raw.get("label") or raw["path"]}
    subfolder = raw.get("subfolder")
    if isinstance(subfolder, str) and subfolder.strip():
        entry["subfolder"] = subfolder.strip()
    return entry


def entry_id(entry: Dict[str, str]) -> str:
    """The string the viewer sends to load one checkpoint.

    Two epochs of one repo share ``path``. The id carries the subfolder so the
    loader can tell them apart.
    """
    sub = (entry.get("subfolder") or "").strip()
    return f"{entry['path']}@{sub}" if sub else entry["path"]


def hub_load_kwargs(path: str) -> Dict[str, str]:
    """Hub kwargs for a saved repo id. Empty when the checkpoint is the repo root.

    ``repo@subfolder`` names one epoch. A bare repo id still resolves when that
    repo has exactly one saved subfolder. Two subfolders and no ``@`` do not
    pick one.
    """
    path = path.strip()
    repo, sep, sub = path.partition("@")
    if sep:
        sub = sub.strip()
        return {"subfolder": sub} if sub else {}
    matches = [e["subfolder"] for e in _load_saved() if e["path"] == path and e.get("subfolder")]
    if len(matches) == 1:
        return {"subfolder": matches[0]}
    return {}


def _save(saved: List[Dict[str, str]]) -> None:
    target = _store_path()
    try:
        target.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        if target != _FALLBACK_STORE:
            _FALLBACK_STORE.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            raise


def _discovered() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    root = models_root()
    if not root.is_dir():
        return out
    seen_paths: set = set()
    for best in sorted(root.glob("**/best")):
        if (best / "model.safetensors").is_file() or (best / "config.json").is_file():
            resolved = str(best.resolve())
            seen_paths.add(resolved)
            out.append({"path": str(best), "label": best.parent.name, "source": "discovered"})
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "model.safetensors").is_file() or (child / "config.json").is_file():
            resolved = str(child.resolve())
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                out.append({"path": str(child), "label": child.name, "source": "discovered"})
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
        e = {**e, "id": entry_id(e)}
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        result.append(e)
    return result


def add_model(path: str, label: Optional[str]) -> None:
    raw = path.strip()
    if not raw:
        raise ValueError("path is empty")
    resolved = resolve_local_checkpoint(raw)
    repo, sep, sub = resolved.partition("@")
    repo = resolve_local_checkpoint(repo) if sep else resolved
    sub = sub.strip() if sep else ""
    if _looks_like_local_path(repo):
        p = Path(repo)
        target = p / sub if sub else p
        if not target.is_dir():
            raise ValueError(
                f"local checkpoint directory not found on the server: '{raw}' "
                f"(checked {target}, {REPO / raw}, and {models_root() / raw}; "
                "if running in Docker, pass the container path under /app/out/...)"
            )
        if not ((target / "config.json").is_file() or (target / "model.safetensors").is_file() or (target / "pytorch_model.bin").is_file()):
            raise ValueError(
                f"directory '{target}' does not contain config.json or model.safetensors"
            )
    saved = _load_saved()
    entry: Dict[str, str] = {"path": repo, "label": (label or raw).strip()}
    if sub:
        entry["subfolder"] = sub
    key = entry_id(entry)
    if not any(entry_id(m) == key for m in saved):
        saved.append(entry)
        _save(saved)


def remove_model(path: str) -> None:
    key = path.strip()
    _save([m for m in _load_saved() if entry_id(m) != key])
