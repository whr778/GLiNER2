"""Unit tests for the paraloq schema-extraction converter's row -> GLiNER2 logic.

Loads convert_paraloq_json.py by path (it's a script) and exercises convert_row
on synthetic (text, item) rows -- no network / dataset download.
"""

import importlib.util
from pathlib import Path

_CONV = Path(__file__).resolve().parents[1] / "tools" / "data" / "convert_paraloq_json.py"


def _load():
    spec = importlib.util.spec_from_file_location("convert_paraloq_json", _CONV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv = _load()

TEXT = ("Patient ID: PT123456. Name: Jane Doe. Conditions: Asthma, Hypertension. "
        "Prescribed Aspirin 100mg.")
ITEM = {
    "prescription": {
        "prescriptionId": "RX999",                       # NOT in text -> dropped
        "patient": {
            "patientId": "PT123456",
            "firstName": "Jane",
            "medicalConditions": ["Asthma", "Hypertension"],
        },
        "medication": {"name": "Aspirin", "dosage": "100mg"},
    }
}


def test_convert_row_flattens_nested_to_verbatim_fields():
    rec = conv.convert_row({"text": TEXT, "item": ITEM})
    ents = rec["output"]["entities"]
    # leaf field name -> values, only those appearing verbatim in the text
    assert ents["patientId"] == ["PT123456"]
    assert ents["firstName"] == ["Jane"]
    assert ents["medicalConditions"] == ["Asthma", "Hypertension"]  # list under parent label
    assert ents["name"] == ["Aspirin"]
    assert ents["dosage"] == ["100mg"]
    # value not present verbatim is dropped, so its field never appears
    assert "prescriptionId" not in ents
    # every emitted surface is verbatim in the text
    for surfaces in ents.values():
        for s in surfaces:
            assert s in TEXT


def test_convert_row_accepts_json_string_item():
    import json
    rec = conv.convert_row({"text": TEXT, "item": json.dumps(ITEM)})
    assert rec["output"]["entities"]["firstName"] == ["Jane"]


def test_convert_row_none_when_nothing_verbatim():
    rec = conv.convert_row({"text": "unrelated text", "item": {"a": {"b": "zzz"}}})
    assert rec is None


def test_convert_row_none_without_text_or_item():
    assert conv.convert_row({"text": "", "item": ITEM}) is None
    assert conv.convert_row({"text": TEXT, "item": None}) is None
