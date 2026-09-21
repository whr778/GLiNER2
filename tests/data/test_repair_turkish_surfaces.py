"""Repairing lemmatised gold, and refusing the repairs that would change the annotation.

turkish_event's annotator gave dictionary forms where the text carries inflected ones:
gold `mesafe` against a document reading `mesafenin`. The model predicts spans over the
text's OWN words, so that gold is unlearnable AND unscorable. Measured: 93.02% of surfaces
aligned before, 99.83% after, 9,648 mentions recovered.

The repair is NOT unconditionally safe, which is the whole reason for the guard: extending
across a case affix keeps the entity, extending across a DERIVATIONAL one does not
(`futbol` "football" -> `futbolcunun` "of the footballer").
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "data"))
from repair_turkish_surfaces import is_derivational, find_prefix_run, repair_record  # noqa


@pytest.fixture(scope="module")
def splitter():
    from gliner2.processor import SchemaTransformer
    return SchemaTransformer(model_name="jhu-clsp/mmBERT-base",
                             word_splitter="whitespace")


def _run(splitter, text, entities):
    import collections
    rec = {"input": text, "output": {"entities": entities}}
    stats = collections.Counter(); stats["suffixes"] = collections.Counter()
    ex = {"repaired": [], "refused": []}
    out, changed = repair_record(rec, splitter.word_splitter, splitter._find_sublist,
                                 stats, ex)
    return out["output"]["entities"], changed, stats


def test_inflectional_suffix_is_repaired(splitter):
    ents, changed, stats = _run(splitter, "Bu mesafenin uzunlugu .", {"Measure": ["mesafe"]})
    assert changed and ents["Measure"] == ["mesafenin"]
    assert stats["repaired"] == 1


def test_derivational_suffix_is_refused(splitter):
    """`futbol` -> `futbolcunun` changes football into the footballer's."""
    ents, changed, stats = _run(splitter, "Bir futbolcunun adi .", {"Field": ["futbol"]})
    assert not changed and ents["Field"] == ["futbol"], "must be left for skip to drop"
    assert stats["refused_derivational"] == 1


def test_an_already_aligned_surface_is_untouched(splitter):
    ents, changed, stats = _run(splitter, "Bu mesafe uzundur .", {"Measure": ["mesafe"]})
    assert not changed and ents["Measure"] == ["mesafe"]
    assert stats["already_aligned"] == 1


def test_capitalisation_and_dotted_I_survive(splitter):
    """REGRESSION. The splitter lowercases and Turkish dotted capital I folds to TWO
    codepoints (U+0069 U+0307), so rebuilding from tokens produced
    'İnsan kaynaklı kimyasallar' -> 'i̇nsan kaynaklı kimyasalların' -- silently lowercasing
    the corpus and injecting a combining dot. The repair must slice the RAW text."""
    text = "İnsan kaynaklı kimyasalların etkisi ."
    ents, changed, _ = _run(splitter, text, {"Cause": ["İnsan kaynaklı kimyasallar"]})
    assert changed
    got = ents["Cause"][0]
    assert got == "İnsan kaynaklı kimyasalların"
    assert got.startswith("İ"), f"lost the dotted capital: {got!r}"
    assert "̇" not in got, "combining dot injected"
    assert got in text, "repaired surface must appear verbatim in the text"


def test_multi_word_surfaces_extend_every_word(splitter):
    text = "Ülkedeki turizm bölgelerinde artis var ."
    ents, changed, _ = _run(splitter, text, {"Area": ["turizm bölgeleri"]})
    assert changed and ents["Area"] == ["turizm bölgelerinde"]


def test_a_surface_absent_from_the_text_is_left_alone(splitter):
    ents, changed, stats = _run(splitter, "Baska bir cumle .", {"X": ["bulunmayan"]})
    assert not changed and ents["X"] == ["bulunmayan"]
    assert stats["no_prefix_run"] == 1


def test_non_entity_dimensions_are_not_touched(splitter):
    import collections
    rec = {"input": "Bu mesafenin uzunlugu .",
           "output": {"events": [{"event_type": "E", "triggers": ["mesafe"]}]}}
    stats = collections.Counter(); stats["suffixes"] = collections.Counter()
    out, changed = repair_record(rec, splitter.word_splitter, splitter._find_sublist,
                                 stats, {"repaired": [], "refused": []})
    assert not changed
    assert out["output"]["events"][0]["triggers"] == ["mesafe"]


@pytest.mark.parametrize("suffix,expected", [
    ("nin", False), ("nde", False), ("na", False), ("ni", False),
    ("cu", True), ("lı", True), ("lik", True), ("siz", True), ("sal", True),
])
def test_the_derivational_set(suffix, expected):
    assert is_derivational(suffix) is expected


def test_find_prefix_run_requires_every_word(splitter):
    words = ["turizm", "bolgelerinde", "artis"]
    assert find_prefix_run(["turizm", "bolgeleri"], words) == 0
    assert find_prefix_run(["turizm", "yok"], words) == -1
