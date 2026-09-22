"""A corpus declared PARTIAL must not receive injected negatives for that dimension.

`partial_annotation` marks a corpus whose gold is NOT exhaustive for a dimension.
`build_negative_pools.py` has always refused such a corpus as a SOURCE of negatives. The
other half -- refusing it as a TARGET -- was never implemented, so declaring the key
protected nothing at training time while eb17-best declared it for `cmnee_ner`.

It matters because absent-from-gold is not absent-from-text there: cmnee_ner is 9,281
records with entity gold at a measured 32.4% miss rate within an offered label, so an
unprotected run injects on the order of 3,000 labels the document may actually contain --
4.6x the orphaned-query false negatives.
"""

from __future__ import annotations

from gliner2.training.negatives import NegativeLabels

# `annotates` is load-bearing: `_usable_pool` vetoes a dimension the corpus does not
# declare it annotates, and a pool without it silently injects nothing.
POOLS = {
    "corpus_a": {"annotates": {"entities": True},
                 "entities": ["Person", "Location", "Organization", "Date", "Money"]},
    "corpus_b": {"annotates": {"entities": True},
                 "entities": ["Vessel", "Cargo", "Port"]},
}


def _injector(**kw):
    return NegativeLabels(POOLS, {"entities": 1}, seed=42, **kw)


def _schema():
    return {"entities": {"Person": ["Ada"]}}


def test_unprotected_corpus_receives_negatives():
    """The control: without a partial declaration, injection happens as normal."""
    neg = _injector()
    out = neg.inject(_schema(), 0, "cmnee_ner")
    assert len(out.get("entities") or {}) > 1, "expected an absent label to be injected"


def test_partial_corpus_is_refused_as_a_target():
    neg = _injector(partial={"cmnee_ner": ["entities"]})
    out = neg.inject(_schema(), 0, "cmnee_ner")
    assert set(out.get("entities") or {}) == {"Person"}
    assert neg.stats["partial_skips"] == 1


def test_protection_is_scoped_to_the_declared_corpus():
    """A different corpus in the same run is unaffected."""
    neg = _injector(partial={"cmnee_ner": ["entities"]})
    out = neg.inject(_schema(), 0, "docee")
    assert len(out.get("entities") or {}) > 1
    assert neg.stats["partial_skips"] == 0


def test_protection_is_scoped_to_the_declared_dimension():
    """Declaring `entities` partial must not silence other dimensions."""
    neg = NegativeLabels(POOLS, {"entities": 1, "events": 1}, seed=42,
                         partial={"cmnee_ner": ["events"]})
    out = neg.inject(_schema(), 0, "cmnee_ner")
    assert len(out.get("entities") or {}) > 1, "entities were not declared partial"


def test_unknown_corpus_is_not_protected():
    """Provenance that never arrives must not silently grant protection."""
    neg = _injector(partial={"cmnee_ner": ["entities"]})
    out = neg.inject(_schema(), 0, None)
    assert len(out.get("entities") or {}) > 1


def test_composition_line_reports_the_protection():
    """A guard nobody can see is a guard nobody can check."""
    neg = _injector(partial={"cmnee_ner": ["entities"]})
    neg.inject(_schema(), 0, "cmnee_ner")
    assert "partial-protected 1" in neg.composition_line()
