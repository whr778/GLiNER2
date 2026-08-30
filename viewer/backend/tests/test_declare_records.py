"""A structure schema with no record metadata is undecodable on the BOUNDARY path.

The failure is SILENT -- extraction returns {} with no error, which in the viewer reads
as "this model found nothing" rather than "this schema could not be decoded". Measured on
whr778/gliner2-tr-dose-15000, same text and threshold:

    without mode/anchor -> {}
    with mode/anchor    -> {'casualty_report': [{'dead': '22', 'injured': '40', ...}]}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import _declare_records  # noqa: E402


def test_missing_mode_is_declared():
    out = _declare_records({"structures": {"r": {"fields": [{"name": "a"}, {"name": "b"}]}}})
    assert out["structures"]["r"]["mode"] == "natural"
    assert out["structures"]["r"]["anchor"] == "a"


def test_explicit_mode_is_left_alone():
    """A user who declares a mode means it; anchorless must not be rewritten to natural."""
    schema = {"structures": {"r": {"mode": "anchorless", "fields": [{"name": "a"}]}}}
    assert _declare_records(schema)["structures"]["r"]["mode"] == "anchorless"


def test_explicit_anchor_is_left_alone():
    schema = {"structures": {"r": {"mode": "natural", "anchor": "b",
                                   "fields": [{"name": "a"}, {"name": "b"}]}}}
    assert _declare_records(schema)["structures"]["r"]["anchor"] == "b"


def test_schema_without_structures_is_untouched():
    schema = {"entities": ["person"], "classifications": [{"task": "t", "labels": ["x", "y"]}]}
    assert _declare_records(schema) == schema


def test_fieldless_structure_gets_mode_but_no_anchor():
    """No field to anchor on: declare the mode, do not invent an anchor."""
    out = _declare_records({"structures": {"r": {"fields": []}}})
    assert out["structures"]["r"]["mode"] == "natural"
    assert "anchor" not in out["structures"]["r"]


def test_input_is_not_mutated():
    """The request object is reused by the caller; rewriting it in place would leak."""
    schema = {"structures": {"r": {"fields": [{"name": "a"}]}}}
    _declare_records(schema)
    assert "mode" not in schema["structures"]["r"]
