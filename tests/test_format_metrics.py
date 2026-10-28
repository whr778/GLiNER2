"""format_metrics prints stored strict micro F1 and survives a checkpoint with none."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))
import format_metrics  # noqa: E402


def test_strict_micro_f1_and_absent_metrics(tmp_path, monkeypatch):
    metrics = {
        "eval_entity_strict_micro_precision": 0.5,
        "eval_entity_strict_micro_recall": 0.25,
        "eval_entity_strict_micro_f1": 0.3333,
        "eval_entity_strict_support": 4,
        "eval_provenance": {"threshold": 0.3, "split": "val", "records": 4},
    }
    root = tmp_path / "run"
    best = root / "best"
    best.mkdir(parents=True)
    (best / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    card = format_metrics.render(str(root))
    assert "entity  strict micro precision 0.5000  strict micro recall 0.2500  strict micro F1 0.3333  support 4" in card
    assert "split: val" in card
    assert "records: 4" in card
    assert "threshold: 0.3000" in card

    empty = tmp_path / "bare"
    empty.mkdir()
    (empty / "config.json").write_text(json.dumps({
        "inference_defaults": {"threshold": 0.5, "global_decode": True},
    }), encoding="utf-8")
    bare = format_metrics.render(str(empty))
    assert "No metrics file (val_metrics.json, test_metrics.json, eval_metrics.json)." in bare
    assert "threshold: 0.5000" in bare
    assert "global_decode: true" in bare

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        if (kwargs.get("repo_type") == "model" and kwargs["filename"] == "config.json"
                and kwargs.get("subfolder") == "checkpoint-epoch-2"):
            return str(empty / "config.json")
        from huggingface_hub.errors import EntryNotFoundError
        raise EntryNotFoundError("missing")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    hub = format_metrics.render("whr778/gliner2-eb22d-trace-epoch@checkpoint-epoch-2")
    flagged = format_metrics.render(
        "whr778/gliner2-eb22d-trace-epoch", subfolder="checkpoint-epoch-2",
    )
    assert "No metrics file" in hub
    assert "threshold: 0.5000" in hub
    assert "No metrics file" in flagged
    names = {call["filename"] for call in calls}
    assert "model.safetensors" not in names
    assert names <= {
        "val_metrics.json", "test_metrics.json", "eval_metrics.json",
        "train_results.json", "config.json", "eval_history.jsonl",
    }
    assert all(call.get("subfolder") == "checkpoint-epoch-2" for call in calls)

    def fake_dataset(**kwargs):
        calls.append(kwargs)
        if kwargs.get("repo_type") == "model":
            from huggingface_hub.errors import LocalEntryNotFoundError
            raise LocalEntryNotFoundError("not cached as a model")
        if kwargs["filename"] == "val_metrics.json" and kwargs.get("subfolder") == "eb22d-epoch2-read":
            return str(best / "val_metrics.json")
        from huggingface_hub.errors import EntryNotFoundError
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dataset)
    logged = format_metrics.render("whr778/gliner2-run-logs@eb22d-epoch2-read")
    assert "strict micro F1 0.3333" in logged
    assert "records: 4" in logged


def test_structure_card_matches_text_fields(tmp_path):
    metrics = {
        "eval_entity_strict_micro_precision": 0.5,
        "eval_entity_strict_micro_recall": 0.25,
        "eval_entity_strict_micro_f1": 0.3333,
        "eval_entity_strict_support": 1200,
        "eval_entity_relaxed_micro_f1": 0.4,
        "eval_fullmenu_entity_strict_micro_precision": 0.2,
        "eval_fullmenu_entity_strict_micro_recall": 0.1,
        "eval_fullmenu_entity_strict_micro_f1": 0.1333,
        "eval_overall_strict_head_min_f1": 0.1,
        "eval_provenance": {
            "threshold": 0.3, "split": "val", "records": 4000, "full_menu": True,
        },
        "by_language": {"eng": {}, "zho": {}},
        "per_head": {
            "identical_to_global": True,
            "thresholds": {"entity": 0.4},
            "metrics": {"eval_relation_strict_micro_f1": 0.5},
        },
    }
    root = tmp_path / "run"
    root.mkdir()
    (root / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (root / "config.json").write_text(json.dumps({
        "inference_defaults": {"threshold": 0.5, "chunk_size": 4096},
    }), encoding="utf-8")

    title, files = format_metrics.load(str(root))
    text = format_metrics.format_card(title, files)
    view = format_metrics.structure_card(title, files)
    section = view["sections"][0]
    row = section["rows"][0]
    assert row == {
        "category": "entity",
        "precision": "0.5000",
        "recall": "0.2500",
        "f1": "0.3333",
        "support": "1,200",
        "relaxed_f1": "0.4000",
    }
    assert "support 1,200" in text
    assert section["full_menu"][0]["f1"] == "0.1333"
    assert "full menu" in text
    provenance = {item["key"]: item["value"] for item in section["provenance"]}
    assert provenance["records"] == "4,000"
    assert provenance["threshold"] == "0.3000"
    assert provenance["split"] == "val"
    assert provenance["full_menu"] == "true"
    assert section["notes"] == [{"key": "overall strict head-min F1", "value": "0.1000"}]
    assert section["by_language"] == "eng, zho"
    assert section["per_head"]["identical_to_global"] is True
    assert section["per_head"]["thresholds"] == "entity=0.4000"
    assert section["per_head"]["rows"][0]["category"] == "relation"
    defaults = {item["key"]: item["value"] for item in view["inference_defaults"]}
    assert defaults["threshold"] == "0.5000"
    assert defaults["chunk_size"] == "4,096"
    assert view["empty"] is False
