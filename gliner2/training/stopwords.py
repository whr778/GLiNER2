"""Multilingual stopword builder for relaxed NER/IE metric evaluation.

Primary source: stopwordsiso (58 languages, ISO 639-1 two-letter codes).
Supplement: optional YAML file keyed by ISO 639-2 three-letter codes for
languages not covered by stopwordsiso; the two sources are merged.

Example YAML entry (tools/train/config/stopwords.yaml):
    uzb:
      - va
      - yoki
      - lekin
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Collection

logger = logging.getLogger(__name__)


def build_stopwords(
    lang_codes: Collection[str],
    extra_yaml: Path | str | None = None,
) -> frozenset:
    """Build a combined stopword set from ISO 639-2 (alpha-3) language codes.

    Merges stopwordsiso (primary, 58 languages) with entries from extra_yaml
    (supplement for languages not covered by stopwordsiso). Codes with no
    coverage in either source emit a warning and contribute nothing.

    Args:
        lang_codes: ISO 639-2 alpha-3 codes, e.g. ['eng', 'fra', 'uzb'].
        extra_yaml: Path to a YAML file keyed by ISO 639-2 codes. Optional.

    Returns:
        Frozen union of all resolved stopword sets.
    """
    try:
        import stopwordsiso
        import langcodes
    except ImportError as exc:
        raise ImportError(
            "stopwordsiso and langcodes are required. "
            "Install with: uv add 'gliner2[train]'"
        ) from exc

    yaml_data: dict = {}
    if extra_yaml is not None:
        import yaml
        with open(extra_yaml, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    words: set[str] = set()
    missing: list[str] = []

    for code in lang_codes:
        found = False
        try:
            alpha2 = langcodes.Language.get(code).language
            if stopwordsiso.has_lang(alpha2):
                words.update(stopwordsiso.stopwords(alpha2))
                found = True
        except Exception:
            pass

        if code in yaml_data:
            entry = yaml_data[code]
            if entry:
                words.update(str(w) for w in entry)
            found = True

        if not found:
            missing.append(code)

    if missing:
        logger.warning("No stopwords found for language codes: %s", missing)

    return frozenset(words)
