"""Wojood Hadath rows become offset-anchored events, and leaked sentences leave train."""

import importlib.util
from collections import Counter
from pathlib import Path

_CONV = Path(__file__).resolve().parents[1] / "tools" / "data" / "convert_wojood_hadath.py"


def _load():
    spec = importlib.util.spec_from_file_location("convert_wojood_hadath", _CONV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv = _load()


def test_nested_argument_is_placed_inside_the_trigger():
    row = {
        "sentence_id": 1,
        "sentence": "بعد هزيمة داعش في الرقة جاء داعش",
        "events": [{
            "event_id": "1EVENT",
            "text": "هزيمة داعش في الرقة",
            "Arguments": [
                {"arg_id": "2ORG", "arg_text": "داعش", "relation": "hasAgent"},
                {"arg_id": "3GPE", "arg_text": "الرقة", "relation": "hasLocation"},
            ],
        }],
    }
    rec = conv.convert_row(row, Counter())
    text = rec["input"]
    trigger = rec["output"]["events"][0]["triggers"][0]
    assert text[trigger["start"]:trigger["end"]] == "هزيمة داعش في الرقة"
    agent = rec["output"]["events"][0]["arguments"][0]["entity"]
    assert text[agent["start"]:agent["end"]] == "داعش"
    assert trigger["start"] <= agent["start"] and agent["end"] <= trigger["end"]
    assert rec["output"]["events"][0]["arguments"][0]["role"] == "hasAgent"


def test_absent_event_and_empty_argument_are_dropped():
    row = {
        "sentence": "جملة لا تحتوي الذكر",
        "events": [
            {"text": "حطين", "Arguments": [{"relation": "hasDate", "arg_text": "عام 2019"}]},
            {"text": "الذكر", "Arguments": [{"relation": "hasDate", "arg_text": "  "}]},
        ],
    }
    rec = conv.convert_row(row, Counter())
    assert len(rec["output"]["events"]) == 1
    assert rec["output"]["events"][0]["arguments"] == []
    assert rec["output"]["events"][0]["event_type"] == "Event"


def test_ambiguous_argument_outside_the_trigger_is_dropped():
    row = {
        "sentence": "العام شهد النزاع السوري ثم العام",
        "events": [{
            "text": "النزاع السوري",
            "Arguments": [{"relation": "hasDate", "arg_text": "العام"}],
        }],
    }
    rec = conv.convert_row(row, Counter())
    assert rec["output"]["events"][0]["arguments"] == []


def test_held_out_sentence_leaves_train_and_val():
    shared = {"input": "نفس الجملة", "output": {"events": []}}
    other = {"input": "جملة أخرى", "output": {"events": []}}
    kept, dropped = conv.assign_splits({
        "train": [shared, other, other],
        "val": [shared],
        "test": [shared],
    })
    assert [r["input"] for r in kept["train"]] == ["جملة أخرى"]
    assert kept["val"] == []
    assert [r["input"] for r in kept["test"]] == ["نفس الجملة"]
    assert dropped["train held in test"] == 1
    assert dropped["val held in test"] == 1
    assert dropped["train duplicate"] == 1
