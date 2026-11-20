"""Gold arguments contained in a gold trigger of the same event."""
import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "data" / "measure_argument_inside_trigger.py"


def _load():
    spec = importlib.util.spec_from_file_location("measure_argument_inside_trigger", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()


def _event(trigger, arguments):
    return {"output": {"events": [{"event_type": "Event", "triggers": trigger, "arguments": arguments}]}}


def test_offset_argument_inside_trigger_counts():
    record = _event(
        [{"text": "battle", "start": 10, "end": 40}],
        [{"role": "hasAgent", "entity": {"text": "army", "start": 16, "end": 20}}],
    )
    totals, _ = tool.measure_record(record)
    assert totals["inside"] == 1
    assert totals["outside"] == 0


def test_equal_span_counts_as_inside():
    record = _event(
        [{"text": "battle", "start": 10, "end": 16}],
        [{"role": "hasAgent", "entity": {"text": "battle", "start": 10, "end": 16}}],
    )
    totals, _ = tool.measure_record(record)
    assert totals["inside"] == 1


def test_outside_offset_argument_is_not_inside():
    record = _event(
        [{"text": "battle", "start": 10, "end": 16}],
        [{"role": "hasDate", "entity": {"text": "Tuesday", "start": 20, "end": 27}}],
    )
    totals, _ = tool.measure_record(record)
    assert totals["inside"] == 0
    assert totals["outside"] == 1


def test_string_gold_is_unmeasured():
    record = _event(["attack"], [{"role": "Victim", "entity": "Terry Duffield"}])
    totals, roles = tool.measure_record(record)
    assert totals["unmeasured"] == 1
    assert totals["inside"] + totals["outside"] == 0
    assert roles["Victim"]["unmeasured"] == 1


def test_argument_inside_any_trigger_counts():
    record = _event(
        [{"text": "a", "start": 0, "end": 4}, {"text": "battle", "start": 10, "end": 30}],
        [{"role": "hasLocation", "entity": {"text": "city", "start": 12, "end": 16}}],
    )
    totals, _ = tool.measure_record(record)
    assert totals["inside"] == 1
