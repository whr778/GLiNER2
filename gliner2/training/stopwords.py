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
            "Install with: uv add 'gliner2[local]'"
        ) from exc

    yaml_data: dict = {}
    if extra_yaml is not None:
        import yaml
        with open(extra_yaml, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    words: set[str] = set()
    missing: list[str] = []

    _CC100_ISO_FALLBACK = {"fil": "tl", "tgl": "tl", "tl": "tl"}
    _CC100_YAML_ALIAS = {"ns": "nso", "zh-hans": "zho", "zh-hant": "zho"}

    for code in lang_codes:
        found = False
        raw = str(code).strip()
        norm = _CC100_YAML_ALIAS.get(raw.lower(), raw)
        alpha3 = None
        try:
            lang = langcodes.Language.get(norm.split("_")[0])
            alpha2 = lang.language
            alpha3 = lang.to_alpha3()
            iso_key = _CC100_ISO_FALLBACK.get(alpha2, _CC100_ISO_FALLBACK.get(norm.lower(), alpha2))
            if stopwordsiso.has_lang(iso_key) and "_" not in raw:
                words.update(stopwordsiso.stopwords(iso_key))
                found = True
        except Exception:
            pass

        for yk in dict.fromkeys(k for k in (raw, raw.lower(), norm, alpha3) if k):
            if yk in yaml_data:
                entry = yaml_data[yk]
                if entry:
                    words.update(str(w) for w in entry)
                found = True

        if not found:
            missing.append(code)

    if missing:
        logger.warning("No stopwords found for language codes: %s", missing)

    return frozenset(words)
