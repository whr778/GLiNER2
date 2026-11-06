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


def test_local_checkpoint_variants_and_missing_paths(tmp_path, monkeypatch):
    import models
    from fastapi import HTTPException

    out_dir = tmp_path / "out"
    run_dir = out_dir / "my_run"
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True)
    (best_dir / "config.json").write_text(json.dumps({
        "model_type": "extractor",
        "architecture": "boundary",
        "model_name": "bert-base-uncased",
    }), encoding="utf-8")

    monkeypatch.setattr(models, "REPO", tmp_path)
    monkeypatch.setattr(models, "models_root", lambda: out_dir)
    monkeypatch.setattr(models, "_STORE", tmp_path / "models.json")

    for variant in (
        "out/my_run/best",
        "my_run/best",
        "out/my_run",
        str(run_dir),
        str(best_dir / "config.json"),
        str(best_dir / "model.safetensors"),
    ):
        models._STORE.unlink(missing_ok=True)
        res = app.create_model(app.ModelEntry(path=variant, label="local"))
        saved = [m for m in res["models"] if m["label"] == "local"]
        assert len(saved) == 1
        assert Path(saved[0]["path"]) == best_dir.resolve()
        schema = app.model_schema(variant)
        assert schema["record_defaults"]["boundary"] is True

    # Nonexistent local path returns 400 on POST /models and 404 on GET /model-schema (never 500)
    try:
        app.create_model(app.ModelEntry(path="/nonexistent/local/checkpoint", label="bad"))
        raise AssertionError("expected 400 HTTPException")
    except HTTPException as exc:
        assert exc.status_code == 400

    try:
        app.model_schema("/nonexistent/local/checkpoint")
        raise AssertionError("expected 404 HTTPException")
    except HTTPException as exc:
        assert exc.status_code == 404
