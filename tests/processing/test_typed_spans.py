"""`entity_types` joined to word spans, through the REAL tokenizer.

A hand-rolled surface reconstruction aligned 1.5% of spans; reusing the processor's own
`_tokenize_text` + `_find_sublist` aligns 99.6-100%. That difference is the reason this join
lives on the processor, so the test exercises the real object rather than simulating it.
"""

import pytest

from gliner2.processor import SchemaTransformer


@pytest.fixture(scope="module")
def whitespace():
    return SchemaTransformer(model_name="jhu-clsp/mmBERT-base", word_splitter="whitespace")


@pytest.fixture(scope="module")
def chars():
    return SchemaTransformer(model_name="jhu-clsp/mmBERT-base", word_splitter="char")


def _batch(proc, text, schema):
    proc.change_mode(is_training=True)
    return proc.collate_fn_train([(text, schema)], architecture="boundary",
                                 event_records=True)


def test_spans_are_half_open_and_round_trip(whitespace):
    text = "Acme Corp hired Jane Doe on Tuesday ."
    schema = {
        "entities": {"Organization": ["Acme Corp"], "Person": ["Jane Doe"]},
        "entity_types": {"Acme Corp": ["Organization"], "Jane Doe": ["Person"]},
    }
    spans = _batch(whitespace, text, schema).typed_spans[0]
    assert spans, "no typed spans were produced"
    words = text.split()
    for (start, end), types in spans.items():
        # half-open, matching BoundaryProposals.indices
        surface = " ".join(words[start:end])
        assert surface in ("Acme Corp", "Jane Doe"), surface
        assert types in (("Organization",), ("Person",))


def test_every_occurrence_is_recorded(whitespace):
    text = "Acme Corp met Acme Corp ."
    schema = {"entities": {"Organization": ["Acme Corp"]},
              "entity_types": {"Acme Corp": ["Organization"]}}
    spans = _batch(whitespace, text, schema).typed_spans[0]
    assert len(spans) == 2, spans


def test_absent_entity_types_gives_an_empty_map(whitespace):
    text = "Acme Corp hired Jane Doe ."
    schema = {"entities": {"Organization": ["Acme Corp"]}}
    assert _batch(whitespace, text, schema).typed_spans[0] == {}


def test_cjk_uses_the_char_splitter(chars):
    text = "俄罗斯海军完成任务。"
    org = "俄罗斯海军"
    schema = {"entities": {"Organization": [org]}, "entity_types": {org: ["Organization"]}}
    spans = _batch(chars, text, schema).typed_spans[0]
    assert spans, "CJK surface did not align"
    (start, end), types = next(iter(spans.items()))
    assert "".join(text)[start:end] == org
    assert types == ("Organization",)
