"""The model-card route returns the shared formatter and does not load weights."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402


def test_model_card_route_uses_the_formatter(tmp_path, monkeypatch):
    folder = tmp_path / "ckpt"
    folder.mkdir()
    (folder / "val_metrics.json").write_text(json.dumps({
        "eval_entity_strict_micro_precision": 0.5,
        "eval_entity_strict_micro_recall": 0.25,
        "eval_entity_strict_micro_f1": 0.3333,
        "eval_provenance": {"split": "val", "records": 4, "threshold": 0.3},
    }), encoding="utf-8")

    def fail_load(*_args, **_kwargs):
        raise AssertionError("model card must not load weights")

    monkeypatch.setattr("gliner2.AutoExtractor.from_pretrained", fail_load)
    body = app.model_card(str(folder))
    assert "strict micro F1 0.3333" in body["card"]
    assert body["model"] == str(folder)
    row = body["view"]["sections"][0]["rows"][0]
    assert row["category"] == "entity"
    assert row["precision"] == "0.5000"
    assert row["recall"] == "0.2500"
    assert row["f1"] == "0.3333"
    provenance = {item["key"]: item["value"] for item in body["view"]["sections"][0]["provenance"]}
    assert provenance["split"] == "val"
    assert provenance["records"] == "4"
    assert provenance["threshold"] == "0.3000"
    assert body["view"]["empty"] is False

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "config.json").write_text(json.dumps({
        "inference_defaults": {"threshold": 0.5},
    }), encoding="utf-8")
    absent = app.model_card(str(bare))
    assert "No metrics file" in absent["card"]
    assert "threshold: 0.5000" in absent["card"]
    assert absent["view"]["empty"] is True
    assert absent["view"]["sections"] == []
    defaults = {item["key"]: item["value"] for item in absent["view"]["inference_defaults"]}
    assert defaults["threshold"] == "0.5000"
