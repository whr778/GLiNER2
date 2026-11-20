"""Office ACE records keep their offsets once adapted to the GLiNER2 shape."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))
import train as T  # noqa: E402

from gliner2.processing.coref import attach
from gliner2.processing.surface import SurfaceSpan
from gliner2.processor import _parse_gold
from gliner2.training.data import InputExample

_ADAPT = Path(__file__).resolve().parents[1] / "tools" / "data" / "adapt_ace2005_office.py"


def _load():
    spec = importlib.util.spec_from_file_location("adapt_ace2005_office", _ADAPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


adapt = _load()

TEXT = "Dallas hired Smith. Dallas lost."


def _office():
    return {
        "input": TEXT,
        "output": {
            "entities": {"GPE": ["Dallas"], "PER": ["Smith"]},
            "relations": [{"ORG-AFF.Employment": {"head": "Dallas", "tail": "Smith"}}],
            "events": [{
                "event_type": "Personnel.End-Position",
                "triggers": ["lost"],
                "arguments": [{"role": "Entity", "entity": "Dallas"}],
            }],
            "entity_facts": [
                {"start": 0, "end": 6, "text": "Dallas", "mention_id": "m1", "mention_type": "NAM", "entity_id": "e1", "entity_type": "GPE"},
                {"start": 13, "end": 18, "text": "Smith", "mention_id": "m2", "mention_type": "NAM", "entity_id": "e2", "entity_type": "PER"},
                {"start": 20, "end": 26, "text": "Dallas", "mention_id": "m3", "mention_type": "NAM", "entity_id": "e1", "entity_type": "GPE"},
            ],
            "relation_facts": [{
                "relation_type": "ORG-AFF.Employment",
                "head": {"start": 0, "end": 6, "text": "Dallas", "mention_id": "m1", "entity_id": "e1"},
                "tail": {"start": 13, "end": 18, "text": "Smith", "mention_id": "m2", "entity_id": "e2"},
            }],
            "event_facts": [{
                "event_type": "Personnel.End-Position",
                "event_id": "ev1",
                "triggers": [{"trigger_id": "t1", "start": 27, "end": 31, "text": "lost"}],
                "arguments": [{"role": "Entity", "start": 20, "end": 26, "text": "Dallas", "mention_id": "m3", "trigger_id": "t1"}],
            }],
            "clusters": [
                {"cluster_id": "e1", "mentions": [[0, 6], [20, 26]]},
                {"cluster_id": "e2", "mentions": [[13, 18]]},
            ],
        },
    }


def test_loader_keeps_the_string_summary_and_drops_the_facts():
    example = InputExample.from_dict(_office())
    assert example.events[0].triggers == ["lost"]
    assert example.events[0].arguments[0].entity == "Dallas"
    assert isinstance(_parse_gold(example.events[0].triggers[0]), str)
    assert attach(_office()["output"], _office().get("coref_clusters")).get("_coref_clusters") is None
    assert "event_facts" not in example.to_dict()["output"]


def test_adapter_pins_the_second_dallas_and_builds_coref():
    record = adapt.adapt_record(_office(), Counter())
    text = record["input"]
    event = record["output"]["events"][0]
    trigger = event["triggers"][0]
    argument = event["arguments"][0]["entity"]
    assert text[trigger["start"]:trigger["end"]] == "lost"
    assert argument == {"text": "Dallas", "start": 20, "end": 26}
    assert text[argument["start"]:argument["end"]] == "Dallas"
    span = _parse_gold(argument)
    assert isinstance(span, SurfaceSpan)
    assert (span.start, span.end) == (20, 26)
    assert record["coref_clusters"] == [
        [{"text": "Dallas", "start": 0, "end": 6}, {"text": "Dallas", "start": 20, "end": 26}],
        [{"text": "Smith", "start": 13, "end": 18}],
    ]
    attached = attach(record["output"], record["coref_clusters"])
    assert attached["_coref_clusters"][0][1]["start"] == 20
    assert record["output"]["relations"] == [{"ORG-AFF.Employment": {"head": "Dallas", "tail": "Smith"}}]
    assert record["output"]["entities"]["GPE"] == ["Dallas"]
    assert record["output"]["mention_types"]["Smith"] == "NAM"
    assert "entity_facts" not in record["output"]
    assert "clusters" not in record["output"]


def test_second_pass_leaves_the_gliner_record_in_place():
    once = adapt.adapt_record(_office(), Counter())
    twice = adapt.adapt_record(once, Counter())
    assert twice == once


def test_config_flag_adapts_on_read_and_leaves_a_gliner_record_alone(tmp_path):
    office = tmp_path / "ace2005_eng.train.jsonl"
    office.write_text(json.dumps(_office(), ensure_ascii=False) + "\n", encoding="utf-8")
    plain = tmp_path / "ace2005_eng.val.jsonl"
    plain.write_text(json.dumps(adapt.adapt_record(_office(), Counter()), ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        T.bind_data_flags({"data": {"office_ace": True}})
        adapted = T._read_records([str(office)])
        untouched = T._read_records([str(plain)])
    finally:
        T.bind_data_flags({})
    argument = adapted[0]["output"]["events"][0]["arguments"][0]["entity"]
    assert argument["start"] == 20 and argument["text"] == "Dallas"
    assert adapted[0]["coref_clusters"][0][1]["start"] == 20
    assert "event_facts" not in adapted[0]["output"]
    assert untouched[0]["output"]["events"][0]["triggers"][0]["start"] == 27
    T.bind_data_flags({"data": {"office_ace": False}})
    raw = T._read_records([str(office)])
    T.bind_data_flags({})
    assert raw[0]["output"]["events"][0]["triggers"] == ["lost"]
    assert "coref_clusters" not in raw[0]


def test_span_text_that_misses_the_slice_raises():
    office = _office()
    office["output"]["event_facts"][0]["arguments"][0]["text"] = "Smith"
    with pytest.raises(ValueError, match="covers"):
        adapt.adapt_record(office, Counter())
