"""MENU_SPEC s4 negatives refinements: gold-free dimensions, log-uniform K, the whole-record cap."""
import json
import sys
from pathlib import Path

from gliner2.training.negatives import NegativeLabels

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "train"))
import train as T  # noqa: E402

POOLS = {"news": {"annotates": {"entities": True, "events": True},
                  "entities": [f"E{i}" for i in range(40)],
                  "events": {f"T{i}": ["Agent"] for i in range(30)}},
         "part": {"annotates": {"entities": True, "events": True},
                  "entities": [f"E{i}" for i in range(40)], "events": {"T1": ["Agent"]}}}
WITH_GOLD = {"entities": {"E0": ["Ann"]}, "events": [{"event_type": "T0", "triggers": ["x"], "arguments": []}]}
EVENT_FREE = {"entities": {"E0": ["Ann"]}}


def neg(**kw):
    return NegativeLabels(POOLS, {"entities": 20, "events": 20}, seed=1, **kw)


def test_event_free_record_gets_event_absents_only_when_enabled():
    assert not neg().inject(EVENT_FREE, 0, corpus="news").get("absent_events")
    out = neg(gold_free_dims=True).inject(EVENT_FREE, 0, corpus="news")
    assert len(out["absent_events"]) == 20 and set(out["absent_events"]) <= set(POOLS["news"]["events"])
    assert all(v == ["Agent"] for v in out["absent_events"].values())


def test_gold_free_respects_partial_annotation():
    out = neg(gold_free_dims=True, partial={"part": ["events"]}).inject(EVENT_FREE, 0, corpus="part")
    assert not out.get("absent_events")


def test_gold_free_needs_a_known_corpus():
    assert not neg(gold_free_dims=True).inject(EVENT_FREE, 0, corpus=None).get("absent_events")


def test_loguniform_k_varies_within_1_to_k():
    n = neg(k_sampling="loguniform")
    ks = {len(n.inject(WITH_GOLD, i, corpus="news")["entities"]) - 1 for i in range(200)}
    assert min(ks) >= 1 and max(ks) <= 20 and len(ks) > 5


def test_max_per_record_caps_the_total():
    out = neg(max_per_record=12).inject(WITH_GOLD, 0, corpus="news")
    assert len(out["entities"]) - 1 + len(out.get("absent_events") or {}) == 12


def test_unknown_k_sampling_refused():
    import pytest
    with pytest.raises(ValueError):
        neg(k_sampling="uniform")


def test_processor_emits_absent_events_for_a_record_without_gold_events():
    from gliner2.processor import SchemaTransformer
    proc = SchemaTransformer.__new__(SchemaTransformer)
    proc.V_TOKEN = "[V]"
    proc._transform_schema = lambda name, fields, tok: (name, tuple(fields))
    schemas, labels, types = [], [], []
    proc._process_events({"absent_events": {"T1": ["Agent"]}}, schemas, labels, types, None)
    assert types == ["events"] and labels == [[0, []]] and schemas == [("T1", ("trigger", "Agent"))]
    schemas, labels, types = [], [], []
    proc._process_events({}, schemas, labels, types, None)
    assert types == []


def test_read_transformed_tags_the_corpus(tmp_path):
    f = tmp_path / "mycorp.train.jsonl"
    f.write_text(json.dumps({"input": "x", "output": {}}) + "\n")
    assert T.read_transformed([str(f)], {}, set())[0]["_corpus"] == "mycorp"


def test_validation_keeps_the_corpus_tag():
    """DataLoader_Factory.load(validate=True) rebuilt records via to_dict and dropped `_corpus`, so the
    injector saw corpus=None in every validated run and gold-free negatives never fired."""
    from gliner2.training.trainer import DataLoader_Factory
    rec = {"input": "Ann met Bob.", "output": {"entities": {"person": ["Ann"]}}, "_corpus": "news"}
    out = DataLoader_Factory.load(data=[rec], max_samples=-1, shuffle=False, seed=1, validate=True)
    assert out[0]["_corpus"] == "news"
