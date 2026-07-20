"""Viewer configuration: where the data corpora and model checkpoints live.

Loads ``viewer/config.yaml`` (or the path in ``$GLINER2_VIEWER_CONFIG``) if
present, otherwise falls back to repo-relative defaults so the viewer works out
of the box. Relative paths in the YAML resolve against the repo root; ``~`` and
absolute paths are honored.

    data_root:     data       # scanned for <name>.train.jsonl corpus presets
    models_root:   out        # scanned for **/best checkpoints
    default_model: fastino/gliner2-base-v1
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

_BACKEND = Path(__file__).resolve().parent
_REPO = _BACKEND.parents[1]  # viewer/backend -> viewer -> repo root


@lru_cache(maxsize=1)
def _raw() -> Dict[str, Any]:
    path = Path(os.environ.get("GLINER2_VIEWER_CONFIG") or (_REPO / "viewer" / "config.yaml"))
    if path.is_file():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _root(key: str, default: Path) -> Path:
    val = _raw().get(key)
    if not val:
        return default
    p = Path(os.path.expanduser(str(val)))
    return p if p.is_absolute() else (_REPO / p)


def data_root() -> Path:
    """Directory scanned for ``<name>.train.jsonl`` corpus presets."""
    return _root("data_root", _REPO / "data")


def models_root() -> Path:
    """Directory scanned for ``**/best`` checkpoints to auto-discover."""
    return _root("models_root", _REPO / "out")


def scripts_root() -> Path:
    """Directory holding ``infer_file.py`` (for schema derivation)."""
    return _root("scripts_root", _REPO / "scripts")


def default_model() -> str:
    """Server default model: env ``GLINER2_MODEL`` > config > built-in."""
    return os.environ.get("GLINER2_MODEL") or _raw().get("default_model") or "fastino/gliner2-base-v1"
