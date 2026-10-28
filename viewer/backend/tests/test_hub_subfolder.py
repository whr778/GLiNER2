"""A saved Hub id can name the checkpoint subfolder the loader must pass."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402
import models  # noqa: E402

REPO = "whr778/gliner2-eb22d-trace-epoch"


def _store(tmp_path, monkeypatch):
    store = tmp_path / "models.json"
    store.write_text(json.dumps([{
        "path": REPO,
        "label": "eb22d epoch 1",
        "subfolder": "checkpoint-epoch-1",
    }]), encoding="utf-8")
    monkeypatch.setattr(models, "_STORE", store)
    return store


def test_two_epochs_of_one_repo_load_apart(tmp_path, monkeypatch):
    store = tmp_path / "models.json"
    store.write_text(json.dumps([
        {"path": REPO, "label": "eb22d epoch 1", "subfolder": "checkpoint-epoch-1"},
        {"path": REPO, "label": "eb22d epoch 2", "subfolder": "checkpoint-epoch-2"},
    ]), encoding="utf-8")
    monkeypatch.setattr(models, "_STORE", store)
    listed = [m for m in models.list_models("fastino/gliner2-base-v1") if m["path"] == REPO]
    assert [m["id"] for m in listed] == [
        f"{REPO}@checkpoint-epoch-1",
        f"{REPO}@checkpoint-epoch-2",
    ]
    assert models.hub_load_kwargs(f"{REPO}@checkpoint-epoch-2") == {"subfolder": "checkpoint-epoch-2"}
    assert models.hub_load_kwargs(f"{REPO}@checkpoint-epoch-1") == {"subfolder": "checkpoint-epoch-1"}
    assert models.hub_load_kwargs(REPO) == {}
    models.remove_model(f"{REPO}@checkpoint-epoch-2")
    kept = json.loads(store.read_text(encoding="utf-8"))
    assert [m["subfolder"] for m in kept] == ["checkpoint-epoch-1"]

    # put epoch 2 back and prove the loader strips the id before the Hub call
    store.write_text(json.dumps([
        {"path": REPO, "label": "eb22d epoch 1", "subfolder": "checkpoint-epoch-1"},
        {"path": REPO, "label": "eb22d epoch 2", "subfolder": "checkpoint-epoch-2"},
    ]), encoding="utf-8")
    seen = {}

    def fake_from_pretrained(repo, **kwargs):
        seen["repo"] = repo
        seen["kwargs"] = kwargs

        class _Weights:
            def parameters(self):
                import torch
                yield torch.nn.Parameter(torch.zeros(1))

        return _Weights()

    monkeypatch.setattr("gliner2.AutoExtractor.from_pretrained", fake_from_pretrained)
    app._models.clear()
    app.get_model(f"{REPO}@checkpoint-epoch-2", device="cpu")
    assert seen["repo"] == REPO
    assert seen["kwargs"]["subfolder"] == "checkpoint-epoch-2"


def test_saved_subfolder_is_the_load_kwarg(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    assert models.hub_load_kwargs(REPO) == {"subfolder": "checkpoint-epoch-1"}
    assert models.hub_load_kwargs("fastino/gliner2-base-v1") == {}
    listed = models.list_models("fastino/gliner2-base-v1")
    hit = next(m for m in listed if m["path"] == REPO)
    assert hit["label"] == "eb22d epoch 1"
    assert hit["subfolder"] == "checkpoint-epoch-1"

    models.add_model("org/other", "other")
    models.remove_model("org/other")
    kept = json.loads(store.read_text(encoding="utf-8"))
    assert kept[0]["subfolder"] == "checkpoint-epoch-1"
    assert kept[0]["label"] == "eb22d epoch 1"


def test_extract_loader_passes_subfolder_and_does_not_fetch(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    seen = {}

    def fake_from_pretrained(repo, **kwargs):
        seen["repo"] = repo
        seen["kwargs"] = kwargs

        class _Weights:
            def parameters(self):
                import torch
                yield torch.nn.Parameter(torch.zeros(1))

        return _Weights()

    monkeypatch.setattr("gliner2.AutoExtractor.from_pretrained", fake_from_pretrained)
    app._models.clear()
    app.get_model(REPO, device="cpu")
    assert seen == {
        "repo": REPO,
        "kwargs": {"map_location": "cpu", "subfolder": "checkpoint-epoch-1"},
    }


def test_model_schema_passes_subfolder_and_does_not_fetch(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    seen = {}

    def fake_from_pretrained(model, **kwargs):
        seen["model"] = model
        seen["kwargs"] = kwargs

        class _Cfg:
            boundary_head = {}
            default_schema = {"entities": ["person"]}
            inference_defaults = None
            label_map = None

        return _Cfg()

    monkeypatch.setattr(
        "gliner2.configuration.ExtractorConfig.from_pretrained", fake_from_pretrained)
    body = app.model_schema(REPO)
    assert seen == {
        "model": REPO,
        "kwargs": {"subfolder": "checkpoint-epoch-1"},
    }
    assert body["schema"] == {"entities": ["person"]}
