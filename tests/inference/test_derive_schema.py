"""Contract for derive_schema: valid multi-task union + open-vocab guard."""

from __future__ import annotations

from gliner2.inference.schema import Schema, _OPEN_VOCAB_LIMIT, derive_schema


def _record():
    return {
        "output": {
            "entities": {"person": [], "company": [], "location": []},
            "events": [
                {"event_type": "Hire", "arguments": [
                    {"role": "Employee"}, {"role": "Employer"}]},
                {"event_type": "Trigger-Only", "arguments": []},  # dropped (no roles)
            ],
            "relations": [{"works_at": {"head": "person", "tail": "company"}}],
            "classifications": [
                {"task": "topic", "labels": ["business", "sports"]},
                {"task": "flag", "labels": ["only_one"]},  # dropped (< 2 labels)
            ],
        }
    }


def test_union_shapes_are_schema_valid():
    s = derive_schema([_record()])
    assert s["entities"] == ["company", "location", "person"]
    assert s["events"] == {"Hire": ["Employee", "Employer"]}  # trigger-only dropped
    assert s["relations"] == ["works_at"]  # list of names, not head/tail dicts
    assert s["classifications"] == [{"task": "topic", "labels": ["business", "sports"]}]
    Schema.from_dict(s)  # must not raise


def test_single_label_classification_dropped():
    s = derive_schema([_record()])
    tasks = {c["task"] for c in s["classifications"]}
    assert "flag" not in tasks


def test_open_vocab_entities_marked_not_dropped():
    rec = {"output": {
        "entities": {f"ent{i}": [] for i in range(_OPEN_VOCAB_LIMIT + 1)},
        "events": [{"event_type": "Hire", "arguments": [{"role": "Employee"}]}],
    }}
    s = derive_schema([rec])
    assert "entities" not in s  # concrete labels omitted, not truncated
    assert s["open_vocab"] == ["entities"]  # capability still advertised
    assert s["events"] == {"Hire": ["Employee"]}
    Schema.from_dict(s)  # open_vocab marker is ignored, concrete dims stay valid


def test_fully_open_vocab_model_is_not_empty():
    rec = {"output": {"entities": {f"ent{i}": [] for i in range(_OPEN_VOCAB_LIMIT + 1)}}}
    s = derive_schema([rec])
    assert s == {"open_vocab": ["entities"]}  # non-empty -> train.py stores it, not None


def test_at_limit_entities_kept():
    rec = {"output": {"entities": {f"ent{i}": [] for i in range(_OPEN_VOCAB_LIMIT)}}}
    s = derive_schema([rec])
    assert len(s["entities"]) == _OPEN_VOCAB_LIMIT
    assert "open_vocab" not in s


def test_empty_records_give_empty_schema():
    assert derive_schema([]) == {}
    assert derive_schema([{"output": {}}]) == {}
