"""Contract for derive_schema: valid multi-task union + open-vocab guard."""

from __future__ import annotations

from gliner2.inference.schema import Schema, _OPEN_VOCAB_LIMIT, derive_schema, merge_schemas


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


def test_derive_schema_handles_list_entities_dict_args_absent_events_and_head_tail_relations():
    rec = {
        "output": {
            "entities": [{"Person": ["Alice"], "Organization": ["Acme"]}],
            "events": [{"event_type": "Attack", "arguments": {"Attacker": ["Alice"], "Target": ["Acme"]}}],
            "absent_events": {"Databreach": ["Compromised-Data"]},
            "relations": [{"head": "Alice", "tail": "Acme", "relation": "works_for"}],
        }
    }
    s = derive_schema([rec])
    assert s == {
        "entities": ["Organization", "Person"],
        "events": {"Attack": ["Attacker", "Target"], "Databreach": ["Compromised-Data"]},
        "relations": ["works_for"],
    }


def test_derive_schema_collects_structures_and_majority_anchor():
    recs = [
        {
            "output": {
                "json_structures": [{"casualty_report": {"location": "Austin"}}],
                "record_metadata": {"casualty_report": {"mode": "natural", "anchor": "location"}},
            }
        },
        {
            "output": {
                "json_structures": [{"casualty_report": {"dead": "4", "location": "Austin"}}],
                "record_metadata": {"casualty_report": {"mode": "natural", "anchor": "dead"}},
            }
        },
        {
            "output": {
                "json_structures": [{"casualty_report": {"dead": "9", "injured": "12"}}],
                "record_metadata": {"casualty_report": {"mode": "natural", "anchor": "dead"}},
            }
        },
    ]
    s = derive_schema(recs)
    assert s["structures"]["casualty_report"] == {
        "fields": [
            {"name": "location", "dtype": "str"},
            {"name": "dead", "dtype": "str"},
            {"name": "injured", "dtype": "str"},
        ],
        "mode": "natural",
        "anchor": "dead",
    }
    Schema.from_dict(s)


def test_merge_schemas_unions_base_and_dataset_preserving_open_vocab():
    base = {
        "events": {"Attack": ["Attacker"], "Hire": ["Employee"]},
        "relations": ["works_at"],
        "classifications": [{"task": "topic", "labels": ["news", "tech"]}],
        "open_vocab": ["entities"],
    }
    added = {
        "entities": ["Person", "Weapon"],
        "events": {"Attack": ["Target"], "Deploy": ["Subject"]},
    }
    merged = merge_schemas(base, added)
    assert "entities" not in merged
    assert merged["open_vocab"] == ["entities"]
    assert merged["events"] == {
        "Attack": ["Attacker", "Target"],
        "Deploy": ["Subject"],
        "Hire": ["Employee"],
    }
    assert merged["relations"] == ["works_at"]
    Schema.from_dict(merged)
