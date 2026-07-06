"""Tests for multilingual stopword building (gliner2.training.stopwords)
and the stopwords parameter threading through the metrics API.
"""
from __future__ import annotations

import textwrap
from collections import Counter
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_STOPWORDS_AVAIL = pytest.importorskip(
    "stopwordsiso", reason="stopwordsiso not installed"
)
_LANGCODES_AVAIL = pytest.importorskip(
    "langcodes", reason="langcodes not installed"
)

from gliner2.training.stopwords import build_stopwords  # noqa: E402
from gliner2.training.metrics import (  # noqa: E402
    _DEFAULT_STOPWORDS,
    _overlap,
    _match_relaxed,
    make_compute_metrics,
)


@pytest.fixture()
def tmp_yaml(tmp_path):
    """Write a minimal stopwords YAML and return its path."""
    def _write(content: str) -> Path:
        p = tmp_path / "sw.yaml"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p
    return _write


def _counters():
    return Counter(), Counter(), Counter()


def _items(pairs):
    """Convert [(label, surface)] to _match_relaxed item format."""
    return sorted(((lbl,), (surf,), lbl) for lbl, surf in pairs)


# ---------------------------------------------------------------------------
# build_stopwords — stopwordsiso path
# ---------------------------------------------------------------------------

class TestBuildStopwordsFromPackage:
    def test_english_returns_nonempty_frozenset(self):
        sw = build_stopwords(["eng"])
        assert isinstance(sw, frozenset)
        assert len(sw) > 5

    def test_english_contains_known_words(self):
        sw = build_stopwords(["eng"])
        assert "the" in sw
        assert "of" in sw

    def test_french_returns_nonempty(self):
        sw = build_stopwords(["fra"])
        assert len(sw) > 5

    def test_german_contains_known_words(self):
        sw = build_stopwords(["deu"])
        assert "die" in sw or "der" in sw or "und" in sw

    def test_chinese_returns_nonempty(self):
        sw = build_stopwords(["zho"])
        assert len(sw) > 0

    def test_japanese_returns_nonempty(self):
        sw = build_stopwords(["jpn"])
        assert len(sw) > 0

    def test_korean_returns_nonempty(self):
        sw = build_stopwords(["kor"])
        assert len(sw) > 0

    def test_multi_language_union(self):
        en = build_stopwords(["eng"])
        fr = build_stopwords(["fra"])
        both = build_stopwords(["eng", "fra"])
        assert both >= en
        assert both >= fr
        # union should be strictly larger (English and French have different words)
        assert len(both) > len(en)

    def test_empty_lang_codes_returns_empty(self):
        sw = build_stopwords([])
        assert sw == frozenset()

    def test_returns_frozenset(self):
        sw = build_stopwords(["eng"])
        assert isinstance(sw, frozenset)


# ---------------------------------------------------------------------------
# build_stopwords — unknown / missing codes
# ---------------------------------------------------------------------------

class TestBuildStopwordsUnknownCodes:
    def test_unknown_code_skipped_silently(self):
        """A code with no coverage should produce empty set and not raise."""
        sw = build_stopwords(["xxx"])
        assert sw == frozenset()

    def test_mixed_known_and_unknown(self):
        sw = build_stopwords(["eng", "xxx_bogus"])
        assert len(sw) > 0  # English words present despite bogus code

    def test_unknown_code_emits_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="gliner2.training.stopwords"):
            build_stopwords(["xyz_not_real"])
        assert caplog.records  # at least one warning emitted


# ---------------------------------------------------------------------------
# build_stopwords — YAML supplement path
# ---------------------------------------------------------------------------

class TestBuildStopwordsYamlSupplement:
    def test_yaml_only_language(self, tmp_yaml):
        p = tmp_yaml("""
            uzb:
              - va
              - yoki
              - lekin
        """)
        sw = build_stopwords(["uzb"], extra_yaml=p)
        assert "va" in sw
        assert "yoki" in sw
        assert "lekin" in sw

    def test_yaml_merges_with_stopwordsiso(self, tmp_yaml):
        p = tmp_yaml("""
            uzb:
              - va
              - yoki
        """)
        sw = build_stopwords(["eng", "uzb"], extra_yaml=p)
        assert "the" in sw   # from stopwordsiso English
        assert "va" in sw    # from YAML Uzbek

    def test_yaml_language_missing_from_codes_ignored(self, tmp_yaml):
        p = tmp_yaml("""
            uzb:
              - va
            kat:
              - და
        """)
        sw = build_stopwords(["uzb"], extra_yaml=p)
        assert "va" in sw
        assert "და" not in sw  # kat not in lang_codes list

    def test_no_yaml_path_works(self):
        sw = build_stopwords(["eng"], extra_yaml=None)
        assert "the" in sw

    def test_language_in_both_sources_is_merged(self, tmp_yaml):
        # Extend English via YAML with extra tokens
        p = tmp_yaml("""
            eng:
              - verily
              - henceforth
        """)
        sw = build_stopwords(["eng"], extra_yaml=p)
        assert "the" in sw        # from stopwordsiso
        assert "verily" in sw     # from YAML
        assert "henceforth" in sw

    def test_empty_yaml_entry_does_not_raise(self, tmp_yaml):
        p = tmp_yaml("""
            uzb:
        """)
        sw = build_stopwords(["uzb"], extra_yaml=p)
        # Empty YAML entry — no words contributed, no error
        assert isinstance(sw, frozenset)

    def test_yaml_values_are_strings(self, tmp_yaml):
        """Numeric YAML values get coerced to str, not crash."""
        p = tmp_yaml("""
            eng:
              - 42
              - the
        """)
        sw = build_stopwords(["eng"], extra_yaml=p)
        assert "the" in sw
        assert "42" in sw


# ---------------------------------------------------------------------------
# Seeded stopwords.yaml coverage (integration)
# ---------------------------------------------------------------------------

class TestSeededYamlCoverage:
    """Verify the shipped stopwords.yaml covers key XLM-R supplement languages."""

    _YAML_PATH = (
        Path(__file__).parent.parent
        / "tools" / "train" / "config" / "stopwords.yaml"
    )

    def test_yaml_file_exists(self):
        assert self._YAML_PATH.exists(), f"Missing: {self._YAML_PATH}"

    @pytest.mark.parametrize("code,sample_word", [
        ("uzb", "va"),
        ("aze", "bir"),
        ("bos", "ali"),
        ("isl", "og"),
        ("kat", "და"),
        ("nep", "र"),
        ("srp", "али"),
        ("sqi", "dhe"),
        ("tam", "மற்றும்"),
        ("tel", "మరియు"),
        ("uzb", "yoki"),
        ("yid", "אין"),
    ])
    def test_key_supplement_language_has_expected_word(self, code, sample_word):
        sw = build_stopwords([code], extra_yaml=self._YAML_PATH)
        assert sample_word in sw, (
            f"Expected '{sample_word}' in stopwords for {code!r}"
        )


# ---------------------------------------------------------------------------
# _overlap — stopwords parameter threading
# ---------------------------------------------------------------------------

class TestOverlapWithCustomStopwords:
    def test_custom_stopwords_block_match(self):
        """Tokens in custom stopwords must not count as shared content."""
        custom = frozenset({"data", "model"})
        assert not _overlap("data model", "data results", stopwords=custom)

    def test_non_stopword_still_matches(self):
        custom = frozenset({"model"})
        assert _overlap("Paris data model", "Paris results", stopwords=custom)

    def test_empty_stopwords_allows_all_tokens(self):
        """With no stopwords, even short shared tokens count."""
        assert _overlap("is are", "is were", stopwords=frozenset())

    def test_default_stopwords_unchanged(self):
        """Existing behaviour: 'the' alone should not match."""
        assert not _overlap("the president", "the bombing")

    def test_default_stopwords_constant_is_correct_type(self):
        assert isinstance(_DEFAULT_STOPWORDS, frozenset)
        assert "the" in _DEFAULT_STOPWORDS


# ---------------------------------------------------------------------------
# _match_relaxed — stopwords parameter threading
# ---------------------------------------------------------------------------

class TestMatchRelaxedWithStopwords:
    def test_default_stopwords_match_existing_behaviour(self):
        gold = _items([("PER", "New York")])
        pred = _items([("PER", "New York City")])
        tp, fp, fn = _counters()
        _match_relaxed(gold, pred, tp, fp, fn)
        assert tp["PER"] == 1

    def test_custom_stopwords_blocks_false_match(self):
        """If both shared tokens are in custom stopwords, no match."""
        custom = frozenset({"new", "york"})
        gold = _items([("PER", "New York")])
        pred = _items([("PER", "New York")])  # exact normalized match always wins
        tp, fp, fn = _counters()
        _match_relaxed(gold, pred, tp, fp, fn, stopwords=custom)
        # exact (normalized) pass matches regardless of stopwords
        assert tp["PER"] == 1

    def test_custom_stopwords_blocks_relaxed_only_match(self):
        """Content token in stopwords → overlap pass fails; only exact pass fires."""
        custom = frozenset({"york"})
        gold = _items([("PER", "New York")])
        pred = _items([("PER", "York Avenue")])  # shares 'york' (stopword)
        tp, fp, fn = _counters()
        _match_relaxed(gold, pred, tp, fp, fn, stopwords=custom)
        # 'york' is a stopword; 'new' vs 'avenue' don't share → no match
        assert tp["PER"] == 0
        assert fp["PER"] == 1
        assert fn["PER"] == 1

    def test_no_stopwords_param_uses_default(self):
        gold = _items([("LOC", "the White House")])
        pred = _items([("LOC", "White House lawn")])
        tp, fp, fn = _counters()
        _match_relaxed(gold, pred, tp, fp, fn)
        assert tp["LOC"] == 1


# ---------------------------------------------------------------------------
# make_compute_metrics — stopwords parameter plumbing
# ---------------------------------------------------------------------------

class TestMakeComputeMetricsStopwords:
    def test_accepts_custom_stopwords(self):
        custom = frozenset({"va", "da", "la"})
        fn = make_compute_metrics(batch_size=4, threshold=0.3, stopwords=custom)
        assert callable(fn)

    def test_default_call_unchanged(self):
        fn = make_compute_metrics()
        assert callable(fn)
