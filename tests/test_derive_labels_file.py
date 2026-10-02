"""tools/derive/labels_file.py: base spelling wins, folds are only proposed, the map is closed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "derive"))
from labels_file import build_labels  # noqa: E402

BASE = {"has_label_inventory": True, "default_schema": {"open_vocab": ["entities"]},
        "label_map": {"entities": {"rollup": False, "separator": ".", "map": {"loc": "Location"}}}}


def _corpus(tmp_path, ents_per_record):
    f = tmp_path / "c.train.jsonl"
    f.write_text("\n".join(json.dumps({"input": "x", "output": {"entities": e}}) for e in ents_per_record) + "\n",
                 encoding="utf-8")
    return [str(f)]


def test_base_spelling_beats_corpus_majority(tmp_path):
    paths = _corpus(tmp_path, [{"location": ["a"]}] * 10 + [{"Location": ["b"]}])
    block, _ = build_labels(paths, BASE, {})
    m = block["entities"]["map"]
    assert m["location"] == "Location" and "Location" not in m


def test_unpinned_fold_match_is_proposed_not_active(tmp_path):
    rows = {"entities": [{"label": "QUANTITY_X", "uses": 3, "status": "fold_match", "target": "Quantity X"}]}
    block, proposals = build_labels(_corpus(tmp_path, [{"QUANTITY_X": ["1"]}]), BASE, rows)
    assert "QUANTITY_X" not in block["entities"]["map"]
    assert proposals == {"entities": {"QUANTITY_X": "Quantity X"}}


def test_map_is_closed_and_base_map_kept(tmp_path):
    paths = _corpus(tmp_path, [{"LOC": ["a"]}, {"loc": ["b"]}, {"Location": ["c"]}])
    block, _ = build_labels(paths, BASE, {})
    m = block["entities"]["map"]
    assert m["loc"] == "Location"
    assert not any(v in m for v in m.values())


def test_base_without_map_uses_only_corpus_clusters(tmp_path):
    bare = {"has_label_inventory": False, "label_map": None, "default_schema": {}}
    block, proposals = build_labels(_corpus(tmp_path, [{"job title": ["a"]}] * 3 + [{"Job Title": ["b"]}]), bare, {})
    assert block["entities"]["map"] == {"Job Title": "job title"}
    assert proposals == {}
