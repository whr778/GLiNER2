"""Unit tests for the DocRED converter's row -> GLiNER2 record logic.

Loads convert_docred.py by path (it's a script) and exercises convert_row on
synthetic DocRED-shaped rows -- no network / dataset download.
"""

import importlib.util
from pathlib import Path

_CONV = Path(__file__).resolve().parents[1] / "tools" / "data" / "convert_docred.py"


def _load():
    spec = importlib.util.spec_from_file_location("convert_docred", _CONV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv = _load()

ROW = {
    "title": "t",
    "sents": [["Zest", "Airways", ",", "Inc.", "is", "based", "in", "Pasay", "City", "."]],
    "vertexSet": [
        [{"name": "Zest Airways", "sent_id": 0, "pos": [0, 4], "type": "ORG"}],
        [{"name": "Pasay City", "sent_id": 0, "pos": [7, 9], "type": "LOC"}],
    ],
    "labels": {
        "head": [0], "tail": [1],
        "relation_id": ["P159"], "relation_text": ["headquarters location"],
        "evidence": [[0]],
    },
}


def test_convert_row_entities_relations_and_verbatim_surfaces():
    rec = conv.convert_row(ROW, set())
    # surface reconstructed from tokens at pos -> appears verbatim in the joined text
    assert rec["input"].startswith("Zest Airways , Inc.")
    assert rec["output"]["entities"] == {"ORG": ["Zest Airways , Inc."], "LOC": ["Pasay City"]}
    for surfaces in rec["output"]["entities"].values():
        for s in surfaces:
            assert s in rec["input"]
    # relation uses the human-readable relation_text with first-mention head/tail
    assert rec["output"]["relations"] == [
        {"headquarters location": {"head": "Zest Airways , Inc.", "tail": "Pasay City"}}
    ]


def test_convert_row_skip_types_drops_bucket():
    rec = conv.convert_row(ROW, {"LOC"})
    assert "LOC" not in rec["output"]["entities"]
    # the relation needs the dropped tail entity, so it is not emitted
    assert "relations" not in rec["output"]


def test_convert_row_none_without_entities():
    empty = {"sents": [["just", "tokens"]], "vertexSet": [],
             "labels": {"head": [], "tail": [], "relation_text": []}}
    assert conv.convert_row(empty, set()) is None


def test_convert_row_entities_only_when_no_labels():
    row = dict(ROW, labels={"head": [], "tail": [], "relation_text": []})
    rec = conv.convert_row(row, set())
    assert rec["output"]["entities"]
    assert "relations" not in rec["output"]
