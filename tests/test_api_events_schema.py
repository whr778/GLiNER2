"""SchemaAPI event-extraction builder (client-side, no network)."""

import inspect

import pytest

from gliner2.api_client import GLiNER2API, SchemaAPI


def test_events_only_build():
    s = SchemaAPI().events({"Attack": ["Attacker", "Target"]}).build()
    assert s == {"events": {"Attack": ["Attacker", "Target"]}}


def test_events_thresholds_and_rich_form():
    s = SchemaAPI().events(
        {"Attack": {"roles": ["Attacker"], "description": "an attack"}},
        trigger_threshold=0.6,
        argument_threshold=0.4,
    ).build()
    assert s["events"] == {"Attack": {"roles": ["Attacker"], "description": "an attack"}}
    assert s["event_trigger_threshold"] == 0.6
    assert s["event_argument_threshold"] == 0.4


def test_events_list_form_normalized_to_dict():
    # The list form is normalized to the {type: config} dict the server parses.
    s = SchemaAPI().events([{"name": "Attack", "roles": ["Attacker"]}]).build()
    assert s["events"] == {"Attack": {"roles": ["Attacker"]}}


def test_events_compose_with_other_tasks():
    s = SchemaAPI().entities(["person"]).events({"Attack": ["Attacker"]}).build()
    assert "entities" in s and "events" in s


def test_empty_or_bad_events_rejected():
    for bad in ({}, []):
        with pytest.raises(ValueError):
            SchemaAPI().events(bad)
    with pytest.raises(ValueError):
        SchemaAPI().events("not a dict or list")


def test_events_counts_as_a_valid_task():
    # An events-only schema must pass extract()'s "at least one task" check.
    assert '"events"' in inspect.getsource(GLiNER2API.extract)


def test_extract_events_convenience_exists():
    assert hasattr(GLiNER2API, "extract_events")
    assert hasattr(GLiNER2API, "batch_extract_events")
