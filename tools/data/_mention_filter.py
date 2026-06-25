"""Config-driven entity mention-type filter for the data converters.

Some corpora annotate each entity mention with a *mention type* -- ACE 2005
uses NAM (named), NOM (nominal), and PRO (pronominal). This helper lets a
converter keep only the mention types you want and drop the rest. Filtering an
entity mention cascades: any relation or event argument that referenced the
dropped mention is dropped too (the converter wires that part using its own
mention->surface map).

The filter is a pure predicate; converters own their drop counts so the stats
line stays honest about what was removed.

Config file (YAML)::

    # Keep only these mention types; drop the rest. Omit `allow` (or the whole
    # file) to keep all. `allow: []` drops every typed mention.
    allow: [NAM, NOM, PRO]        # global default

    converters:                    # optional per-converter overrides
      ace2005:
        allow: [NAM, NOM]          # e.g. drop pronouns for ACE 2005

Usage in a converter::

    from _mention_filter import load_mention_filter

    mf = load_mention_filter(args.filter_config, "ace2005")
    ...
    if not mf.allows(mention_type):
        continue                   # skip this mention; its relations/args cascade
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

import yaml


class MentionFilter:
    """Decides whether an entity mention of a given type is kept.

    ``allowed=None`` keeps every type (the no-op default). An explicit set keeps
    only those types; the empty set drops every typed mention. Untyped mentions
    (``allows(None)`` / ``allows("")``) are always kept -- there is nothing to
    filter on. Matching is case-insensitive.
    """

    def __init__(self, allowed: Optional[Set[str]]) -> None:
        self.allowed = allowed

    @property
    def active(self) -> bool:
        """True when a filter is in effect (i.e. some types may be dropped)."""
        return self.allowed is not None

    def allows(self, mention_type: Optional[str]) -> bool:
        if self.allowed is None:
            return True
        if not mention_type:
            return True
        return mention_type.strip().upper() in self.allowed

    def describe(self) -> str:
        if self.allowed is None:
            return "keep all mention types"
        return f"keep mention types {sorted(self.allowed)}"


def load_mention_filter(config_path, converter: str) -> MentionFilter:
    """Build a :class:`MentionFilter` for ``converter`` from a YAML config.

    Resolution: a ``converters.<converter>.allow`` list wins; otherwise the
    top-level ``allow`` list applies; otherwise (no path, or no ``allow`` key)
    all mention types are kept.
    """
    if config_path is None:
        return MentionFilter(None)
    path = Path(config_path)
    if not path.is_file():
        raise SystemExit(f"mention-filter config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"mention-filter config must be a YAML mapping: {path}")

    section = data
    converters = data.get("converters")
    if isinstance(converters, dict) and isinstance(converters.get(converter), dict):
        section = converters[converter]

    allow = section.get("allow") if "allow" in section else data.get("allow")
    if allow is None:
        return MentionFilter(None)
    if not isinstance(allow, list):
        raise SystemExit(
            f"mention-filter 'allow' must be a list, got {type(allow).__name__}: {path}"
        )
    allowed = {str(t).strip().upper() for t in allow if str(t).strip()}
    return MentionFilter(allowed)
