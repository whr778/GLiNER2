"""Gold arguments contained in a gold trigger of the same event."""
import importlib.util
import json
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


def _office(text, trigger, argument):
    role, surface, start, end = argument
    t_surface, t_start, t_end = trigger
    return {
        "input": text,
        "output": {
            "events": [{
                "event_type": "Event",
                "triggers": [t_surface],
                "arguments": [{"role": role, "entity": surface}],
            }],
            "event_facts": [{
                "event_type": "Event",
                "triggers": [{"start": t_start, "end": t_end, "text": text[t_start:t_end]}],
                "arguments": [{"role": role, "start": start, "end": end, "text": text[start:end]}],
            }],
        },
    }


def test_office_file_counts_the_fact_spans(tmp_path):
    outside = _office(
        "Dallas hired Smith. Dallas lost.",
        ("lost", 27, 31),
        ("Entity", "Dallas", 20, 26),
    )
    inside = _office(
        "the United Nations agreement",
        ("United Nations agreement", 4, 28),
        ("hasAgent", "United Nations", 4, 18),
    )
    plain = _event(
        [{"text": "battle", "start": 10, "end": 40}],
        [{"role": "hasAgent", "entity": {"text": "army", "start": 16, "end": 20}}],
    )
    path = tmp_path / "ace.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in (outside, inside, plain)) + "\n",
        encoding="utf-8",
    )
    text = tool.report([path])
    assert "[office_ace]" in text
    assert "records=3 adapted=2" in text
    assert "inside 2/3 (66.67%)" in text
    assert "outside 1" in text
    assert "unmeasured 0" in text


def test_office_json_object_is_read(tmp_path):
    path = tmp_path / "ace.json"
    path.write_text(json.dumps(_office(
        "Dallas hired Smith. Dallas lost.",
        ("lost", 27, 31),
        ("Entity", "Dallas", 20, 26),
    ), ensure_ascii=False), encoding="utf-8")
    text = tool.report([path])
    assert "records=1 adapted=1" in text
    assert "outside 1" in text
    assert "unmeasured 0" in text


def test_string_gold_file_stays_unmeasured(tmp_path):
    path = tmp_path / "wiki.jsonl"
    record = _event(["attack"], [{"role": "Victim", "entity": "Terry Duffield"}])
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    text = tool.report([path])
    assert "[office_ace]" not in text
    assert "inside 0/0 (0.00%)" in text
    assert "unmeasured 1" in text
