"""Offset-anchored event gold (TODO #26): `{text, start, end}` marks ONE occurrence; windows keep and re-base it."""
import random

from gliner2.training.chunking import _filter_events
from gliner2.training.trainer import ExtractorCollator
from tests.models.boundary.test_record_head_pipeline import _build_tiny_records_model

TEXT = "Police said the bomb killed Ahmed in Kabul . Kabul police were slow ."


def span(text, start):
    return {"text": text, "start": start, "end": start + len(text)}


def _place_values(gold):
    import gliner2.processing.boundary_preprocessing as BP
    captured = []
    original = BP._build_sample_records
    BP._build_sample_records = lambda *a, **k: (captured.append(original(*a, **k)) or captured[-1])
    try:
        model = _build_tiny_records_model()
        coll = ExtractorCollator(model.processor, is_training=True, architecture="boundary", max_gold_per_query=64,
                                 on_capacity_exceeded="raise", error_policy="raise", event_records=True)
        random.seed(0)
        out = coll([(TEXT, gold)])
    finally:
        BP._build_sample_records = original
    toks = out.text_tokens[0]
    return {f.query_id: [" ".join(toks[s:e]) for v in f.values for (s, e) in v] for f in captured[0][0].fields}


def test_offset_argument_marks_one_occurrence_string_marks_all():
    strings = {"events": [{"event_type": "Attack", "triggers": ["killed"], "arguments": [{"role": "Place", "entity": "Kabul"}]}]}
    offsets = {"events": [{"event_type": "Attack", "triggers": [span("killed", TEXT.index("killed"))],
                           "arguments": [{"role": "Place", "entity": span("Kabul", TEXT.index("Kabul"))}]}]}
    assert sorted(map(len, _place_values(strings).values())) == [1, 2]   # trigger x1, "Kabul" x2
    assert sorted(map(len, _place_values(offsets).values())) == [1, 1]


def test_windows_keep_and_rebase_offset_gold_inside_and_drop_it_outside():
    doc = "aaa bbb killed Kabul ccc ddd"
    ev = [{"event_type": "Attack", "triggers": [span("killed", 8)], "arguments": [{"role": "Place", "entity": span("Kabul", 15)}]}]
    inside = _filter_events(ev, doc[4:], c_start=4, c_end=len(doc))
    assert inside[0]["triggers"] == [span("killed", 4)] and inside[0]["arguments"][0]["entity"] == span("Kabul", 11)
    assert doc[4:][4:10] == "killed"
    assert _filter_events(ev, doc[:12], c_start=0, c_end=12) == []          # trigger crosses the window edge


def test_string_gold_windowing_is_unchanged():
    ev = [{"event_type": "Attack", "triggers": ["killed"], "arguments": [{"role": "Place", "entity": "Kabul"}]}]
    assert _filter_events(ev, "x killed Kabul y") == [{"event_type": "Attack", "triggers": ["killed"],
                                                      "arguments": [{"role": "Place", "entity": "Kabul"}]}]


def test_a_wrong_offset_drops_the_whole_record_under_skip():
    """Why converters must gate every span: one bad offset silently loses the record (kept_indices shows it)."""
    model = _build_tiny_records_model()
    coll = ExtractorCollator(model.processor, is_training=True, architecture="boundary", max_gold_per_query=64,
                             on_capacity_exceeded="raise", error_policy="skip", event_records=True)
    good = {"events": [{"event_type": "Die", "triggers": [span("killed", TEXT.index("killed"))]}]}
    bad = {"events": [{"event_type": "Die", "triggers": [{"text": "killed", "start": 1, "end": 7}]}]}
    random.seed(0)
    assert coll([(TEXT, good), (TEXT, bad)]).kept_indices == [0]
