"""Invisible-character normalisation in the JSONL write path.

NFKC folds every space variant but leaves zero-width and bidi FORMAT characters intact.
Those break span matching: an LLM annotator copies the visible characters and drops the
invisible ones, so its surface stops being a substring of the text and the mention is
skipped by on_missing_surface. Measured on disk: ZWSP and BOM in nuner_full, ZWSP and
soft hyphen in docee -- 2 of 91 unalignable surfaces there, the rest being run-together
words in the SOURCE text, which no normalisation can repair.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data"))
from _split import clean_text, dumps_record  # noqa: E402


def test_zero_width_and_bidi_are_stripped():
    assert clean_text("Ankara\u200bda") == "Ankarada"
    assert clean_text("\ufeffBreaking") == "Breaking"
    assert clean_text("in\u00adjured") == "injured"
    assert clean_text("\u200f\u0642\u062a\u0644") == "\u0642\u062a\u0644"
    assert clean_text("a\u2060b") == "ab"
    assert clean_text("\u2066x\u2069") == "x"


def test_semantic_joiners_are_KEPT():
    """ZWNJ separates Persian words; ZWJ forms Indic conjuncts and joins emoji.

    Stripping these would change the text, so they are excluded on purpose.
    """
    persian = "\u0645\u06cc\u200c\u0631\u0648\u062f"
    assert clean_text(persian) == persian
    emoji = "\U0001f468\u200d\U0001f469"
    assert clean_text(emoji) == emoji


def test_nfkc_still_folds_space_variants():
    assert clean_text("40\u00a0dead") == "40 dead"
    assert clean_text("40\u202fdead") == "40 dead"


def test_a_surface_becomes_matchable_again():
    """The failure this fixes: annotator drops the invisible, the text keeps it."""
    text = "Ankara\u200bda 12 kisi oldu"
    surface = "Ankarada"
    assert surface not in text
    assert surface in clean_text(text)


def test_line_separators_still_handled():
    assert "\u2028" not in clean_text("a\u2028b")
    assert dumps_record({"input": "a\u2028b"}).count("\n") == 0
