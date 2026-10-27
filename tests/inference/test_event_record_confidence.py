"""An instance confidence score is not an event argument."""

from gliner2.models.boundary.engine import _record_instances_to_events


def test_instance_confidence_is_not_emitted_as_a_role():
    inst = {
        "trigger": {"text": "left", "start": 0, "end": 4, "confidence": 0.9},
        "Person": {"text": "Ann", "start": 5, "end": 8, "confidence": 0.7},
        "confidence": 0.81,
    }
    events = _record_instances_to_events([inst], "trigger")
    assert [a["role"] for a in events[0]["arguments"]] == ["Person"]
    assert events[0]["arguments"][0]["entity"]["text"] == "Ann"
