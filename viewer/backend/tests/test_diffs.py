"""diffs.py: every unmatched span named by its reason, on the two shapes the viewer showed ambiguously."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gold  # noqa: E402

TEXT = "Systems were knocked out by a complex ransomware virus. The virus demanded four bitcoins."
ROW = {
    "input": TEXT,
    "output": {
        "entities": {"Malware": [{"text": "complex ransomware virus", "start": 30, "end": 54}]},
        "event_extraction": {"Cyber.Phishing": [
            {"triggers": [{"text": "demanded", "start": 66, "end": 74}], "arguments": []}]},
    },
    "gold": {
        "entities": {"Malware": ["a complex ransomware virus"]},
        "events": [{"event_type": "Cyber.Ransom", "triggers": ["demanded"],
                    "arguments": [{"role": "Price", "entity": "four bitcoins"}]}],
    },
}


def view(row=ROW):
    return gold.record_view(json.loads(json.dumps(row)))


def reasons(v, layer):
    return sorted((d["reason"], (d["gold"] or {}).get("text"), (d["pred"] or {}).get("text"))
                  for d in v["diffs"] if d["layer"] == layer)


def test_a_boundary_miss_reads_span_not_label():
    v = view()
    assert reasons(v, "entities") == [("span", "a complex ransomware virus", "complex ransomware virus")]
    why = v["pred"]["entities"]["Malware"][0]["why"]
    assert why == ['span differs: gold "a complex ransomware virus" (Malware) vs pred "complex ransomware virus" (Malware)']


def test_a_trigger_under_another_type_reads_type_and_its_arguments_read_missed():
    v = view()
    assert reasons(v, "events") == [("missed", "four bitcoins", None), ("type", "demanded", "demanded")]
    d = next(d for d in v["diffs"] if d["reason"] == "type")
    # eids follow the frontend's collectMarks numbering, so a table row can pin the mention.
    assert (d["gold"]["eid"], d["pred"]["eid"]) == ("g-evt-0", "evt-0")


def test_matched_spans_carry_no_reason_and_unrelated_predictions_are_spurious():
    row = json.loads(json.dumps(ROW))
    row["output"]["entities"]["Malware"][0].update(text="a complex ransomware virus", start=28)
    row["output"]["entities"]["Person"] = [{"text": "Systems", "start": 0, "end": 7}]
    v = view(row)
    assert "why" not in v["pred"]["entities"]["Malware"][0]
    assert reasons(v, "entities") == [("spurious", None, "Systems")]


def test_same_text_same_role_under_another_trigger_reads_attachment():
    row = json.loads(json.dumps(ROW))
    row["output"]["event_extraction"] = {"Cyber.Ransom": [
        {"triggers": [{"text": "knocked", "start": 13, "end": 20}],
         "arguments": [{"role": "Price", "entity": {"text": "four bitcoins", "start": 75, "end": 88}}]}]}
    assert ("attachment", "four bitcoins", "four bitcoins") in reasons(view(row), "events")
