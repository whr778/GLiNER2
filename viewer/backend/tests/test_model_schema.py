"""model-schema reads coreference switches from config.json and does not load weights."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402


def _write(path: Path, boundary_head: dict | None) -> None:
    body = {
        "model_type": "extractor",
        "architecture": "boundary",
        "model_name": "bert-base-uncased",
    }
    if boundary_head is not None:
        body["boundary_head"] = boundary_head
    path.mkdir()
    (path / "config.json").write_text(json.dumps(body), encoding="utf-8")


def test_absent_coref_keys_read_as_off(tmp_path):
    folder = tmp_path / "bare"
    _write(folder, None)
    coref = app.model_schema(str(folder))["coref"]
    assert coref == {
        "record_merge_coreferent": "off",
        "record_coref_link": False,
        "record_coref_link_threshold": 0.5,
        "entity_coref_link": False,
    }


def test_link_threshold_is_read_from_config(tmp_path):
    folder = tmp_path / "linked"
    _write(folder, {
        "record_coref_link": True,
        "record_merge_coreferent": "link",
        "record_coref_link_threshold": 0.7,
        "entity_coref_link": True,
    })
    coref = app.model_schema(str(folder))["coref"]
    assert coref["record_merge_coreferent"] == "link"
    assert coref["record_coref_link"] is True
    assert coref["record_coref_link_threshold"] == 0.7
    assert coref["entity_coref_link"] is True
