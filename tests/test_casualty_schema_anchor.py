"""Tests for the EKF pipeline schema, Title_Snake mapping, offset windows, scope agreement, and replay hygiene."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from gliner2 import Schema

ROOT = Path(__file__).resolve().parents[1]


def _load_showcase_module(name: str, filename: str):
    showcase = str(ROOT / "tools" / "ekf_showcase")
    if showcase not in sys.path:
        sys.path.insert(0, showcase)
    path = ROOT / "tools" / "ekf_showcase" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pipeline():
    return _load_showcase_module("ekf_run_pipeline", "run_pipeline.py")


def test_default_anchor_is_dead():
    built = _pipeline().build_casualty_schema(with_location=True).build()
    report = built["record_metadata"]["casualty_report"]
    assert report["anchor"] == "dead"
    assert report["mode"] == "natural"


def test_location_anchor_requires_the_location_field():
    built = _pipeline().build_casualty_schema(with_location=True, anchor="location").build()
    assert built["record_metadata"]["casualty_report"]["anchor"] == "location"
    with pytest.raises(ValueError, match="anchor"):
        _pipeline().build_casualty_schema(with_location=False, anchor="location").build()


def test_legacy_record_mode_none_omits_record_metadata():
    built = _pipeline().build_casualty_schema(with_location=True, record_mode=None).build()
    assert "record_metadata" not in built


def test_eb22g_finetune_configs_use_the_v4_label_file_and_name_replay():
    expected = {
        ROOT / "tools/train/config/gate/gate4-eb22g.yaml": (
            "data/gate3_eb22g",
            "data/replay_eb22g_gate4",
        ),
        ROOT / "tools/train/config/casualty/casualty-ml-eb22g.yaml": (
            "data/casualty_ml_eb22g",
            "data/replay_eb22g_casualty",
        ),
    }
    for path, (corpus, replay) in expected.items():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["model"]["pretrained"] == "whr778/gliner2-eb22g"
        assert "boundary_head" not in cfg["model"]
        assert "labels" not in cfg
        assert cfg["labels_file"] == "labels/unified-full-v4.yaml"
        assert corpus in cfg["data"]["corpora"]
        assert replay in cfg["data"]["corpora"]


def test_eb23a_casualty_config_uses_the_same_slices_without_copying_the_head():
    path = ROOT / "tools/train/config/casualty/casualty-ml-eb23a.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["model"]["pretrained"] == "whr778/gliner2-eb23a"
    assert "boundary_head" not in cfg["model"]
    assert "strip_bare_space_subword" not in cfg["model"]
    assert "labels" not in cfg
    assert cfg["labels_file"] == "labels/unified-full-v4.yaml"
    assert cfg["data"]["corpora"] == [
        "data/casualty_ml_eb22g",
        "data/casualty_events",
        "data/replay_eb22g_casualty",
    ]
    assert cfg["training"]["schema_dropout"]["remove_events_prob"] == 0.0
    assert cfg["training"]["relation_gold_pairs"] == "nearest"
    assert cfg["training"]["error_policy"] == "skip"
    assert cfg["training"]["on_capacity_exceeded"] == "truncate_with_warning"
    assert cfg["eval"]["global_decode"] is True
    assert cfg["eval"]["threshold"] == 0.3


def test_gate4_configs_keep_short_text_and_inherit_the_head():
    """Gate text stays short. The head comes from the checkpoint, both bases."""
    shared = {
        "error_policy": "skip",
        "on_capacity_exceeded": "truncate_with_warning",
        "relation_gold_pairs": "nearest",
    }
    expected = {
        ROOT / "tools/train/config/gate/gate4-eb22g.yaml": "whr778/gliner2-eb22g",
        ROOT / "tools/train/config/gate/gate4-eb23a.yaml": "whr778/gliner2-eb23a",
    }
    for path, pretrained in expected.items():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["model"]["pretrained"] == pretrained
        assert "boundary_head" not in cfg["model"]
        assert "strip_bare_space_subword" not in cfg["model"]
        assert "labels" not in cfg
        assert cfg["labels_file"] == "labels/unified-full-v4.yaml"
        assert cfg["data"]["corpora"] == [
            "data/gate3_eb22g",
            "data/replay_eb22g_gate4",
        ]
        assert cfg["training"]["sliding_window"] is False
        assert cfg["training"]["max_len"] == 2048
        assert cfg["training"]["metric_for_best"] == "eval_classification_strict_micro_f1"
        assert cfg["training"]["schema_dropout"]["remove_events_prob"] == 0.0
        for key, value in shared.items():
            assert cfg["training"][key] == value
        assert cfg["eval"]["global_decode"] is True
        assert cfg["eval"]["threshold"] == 0.3
        assert cfg["eval"]["eval_by_language"] is True


def test_pipeline_title_snake_round_trip_and_offset_windows():
    rp = _pipeline()
    text = (
        "February 2, 1999 -- Archive note from 1999. "
        "Today in February 2024, officials confirmed at least 2 people died in Florida."
    )

    class MockTitleSnakeExtractor:
        def __init__(self):
            self.config = types.SimpleNamespace(
                label_map=None,
                label_style={"name": "title_snake"},
            )
            self.thresholds = []

        def create_schema(self):
            return Schema()

        def extract(self, txt, schema, threshold=0.5, include_confidence=False, include_spans=False):
            self.thresholds.append(threshold)
            cls_list = schema.get("classifications") or []
            tasks = {c["task"]: c["labels"] for c in cls_list}
            if "relevance" in tasks:
                assert tasks["relevance"] == ["Mass_Casualty", "Other"]
                return {"relevance": {"label": "Mass_Casualty", "confidence": 0.97}}
            if "disaster_type" in tasks:
                return {"disaster_type": {"label": "Flood", "confidence": 0.89}}
            if "qualifier" in tasks:
                assert "At_Least" in tasks["qualifier"]
                assert "Official" in tasks["source"]
                return {
                    "qualifier": {"label": "At_Least", "confidence": 0.91},
                    "source": {"label": "Official", "confidence": 0.93},
                }
            if rp.DOCEE_TASK in tasks:
                assert "Casualties_And_Losses" in schema["entities"]
                return {
                    rp.DOCEE_TASK: {"label": "Air_Crash", "confidence": 0.92},
                    "entities": {
                        "Casualties_And_Losses": [{"text": "at least 2 people died", "start": 88, "end": 110}],
                        "Location": [{"text": "Florida", "start": 114, "end": 121}],
                        "Date": [],
                        "Cause": [],
                    },
                }
            if schema.get("json_structures"):
                assert "Casualty_Report" in schema["json_structures"][0]
                assert schema["record_metadata"]["Casualty_Report"]["anchor"] == "Dead"
                i = txt.rfind("2 people")
                loc_i = txt.rfind("Florida")
                return {
                    "Casualty_Report": [{
                        "Location": {"text": "Florida", "confidence": 0.94, "start": loc_i, "end": loc_i + 7},
                        "Dead": {"text": "2", "confidence": 0.91, "start": i, "end": i + 1},
                    }]
                }
            raise AssertionError(f"Unhandled schema: {schema}")

    m = MockTitleSnakeExtractor()
    gates = rp.gate(m, [text], threshold=0.9)
    assert gates[0]["relevant"] is True
    assert gates[0]["relevance"] == "mass_casualty"
    assert gates[0]["disaster_type"] == "flood"

    s1 = rp.extract_stage1(m, [text], threshold=0.3)
    assert s1[0]["event_type"] == "Air Crash"
    assert len(s1[0]["casualties"]) == 1

    cas_sc = rp.build_casualty_schema(with_location=True)
    recs = rp.extract_casualty_records(m, text, cas_sc, long=False, threshold=0.25)
    assert m.thresholds[-1] == 0.25
    assert recs[0]["location"]["text"] == "Florida"
    assert recs[0]["dead"]["text"] == "2"

    # Without offsets, "2" hits "February 2, 1999" (1999 < 2023 -> rejected); with offsets it hits 2024 (kept).
    assert rp.out_of_window(text, "2", {}, event_year=2024) == 1999
    assert (
        rp.out_of_window(
            text, "2", {}, event_year=2024,
            start=recs[0]["dead"]["start"], end=recs[0]["dead"]["end"],
        )
        is None
    )

    cls_sc = rp.build_normalizer_schema(m)
    per_mode = {"heuristic": [], "classify": []}
    entry = {"observations": []}
    rejected = []
    rp._emit(
        recs[0], text, {"t_hours": 12.0, "text": text}, entry,
        ["heuristic", "classify"], per_mode, m, cls_sc,
        event_key="Flood|florida", events_block={}, event_year=2024, rejected=rejected,
    )
    assert rejected == []
    assert per_mode["heuristic"][0]["qualifier"] == "at_least"
    assert per_mode["classify"][0]["qualifier"] == "at_least"
    assert per_mode["classify"][0]["source"] == "official"


def test_scope_agreement_handles_duplicate_values_across_places():
    sg = _load_showcase_module("ekf_scope_gate", "scope_gate.py")
    states = {"florida": "florida", "georgia": "georgia"}
    obs = [
        {"t_hours": 1.0, "event_key": "__aggregate__", "value": 100, "scope": "national"},
        {"t_hours": 2.0, "event_key": "florida", "value": 5, "scope": "place"},
        {"t_hours": 3.0, "event_key": "florida", "value": 8, "scope": "place"},
        {"t_hours": 10.0, "event_key": "florida", "value": 90, "scope": "national"},
        {"t_hours": 10.0, "event_key": "georgia", "value": 90, "scope": "place"},
    ]
    agr = sg.scope_agreement(obs, states, ratio=2.0, warmup=2, reference="aggregate")
    assert agr["national/national"] == 1
    assert agr["place/place"] == 3
    assert agr["agreement"] == 1.0
    assert all("_sa_id" not in o for o in obs)


def test_threshold_coupled_decode_detection():
    fg = _load_showcase_module("ekf_frontend_gates", "frontend_gates.py")
    ba = _load_showcase_module("ekf_binding_accuracy", "binding_accuracy.py")
    plain = types.SimpleNamespace(boundary_settings=types.SimpleNamespace(
        record_argument_coref_boost=0.0, entity_coref_threshold=None,
        record_argument_coref_copy="all", record_merge_coreferent="off",
    ))
    boosted = types.SimpleNamespace(boundary_settings=types.SimpleNamespace(
        record_argument_coref_boost=0.25, entity_coref_threshold=0.5,
        record_argument_coref_copy="canonical", record_merge_coreferent="off",
    ))
    assert fg._has_threshold_coupled_decode(plain) is False
    assert fg._has_threshold_coupled_decode(boosted) is True
    assert ba._has_threshold_coupled_decode(plain) is False
    assert ba._has_threshold_coupled_decode(boosted) is True


def test_build_137k_replay_dedupes_paths_and_excludes_eval_docs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    p_train = tmp_path / "data" / "corp_a.train.jsonl"
    p_val = tmp_path / "data" / "corp_a.val.jsonl"
    p_test = tmp_path / "data" / "corp_a.test.jsonl"
    target_val = tmp_path / "data" / "target.val.jsonl"
    target_test = tmp_path / "data" / "target.test.jsonl"

    p_train.write_text(
        "\n".join([
            json.dumps({"input": "Clean doc 1", "output": {}}),
            json.dumps({"input": "Clean doc 2", "output": {}}),
            json.dumps({"input": "Base val leak", "output": {}}),
            json.dumps({"input": "Target test leak", "output": {}}),
        ]) + "\n",
        encoding="utf-8",
    )
    p_val.write_text(json.dumps({"input": "Base val leak", "output": {}}) + "\n", encoding="utf-8")
    p_test.write_text("", encoding="utf-8")
    target_val.write_text("", encoding="utf-8")
    target_test.write_text(json.dumps({"input": "Target test leak", "output": {}}) + "\n", encoding="utf-8")

    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "data": {
                "corpora": ["data/corp_a"],
                # Same train path referenced again under event_files must be deduplicated
                "event_files": {"corp_a_alias": {"train": "data/corp_a.train.jsonl"}},
            }
        }),
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location(
        "build_137k_replay_mod", ROOT / "tools" / "train" / "build_137k_replay.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    files = mod.pool_files(cfg_path)
    assert len(files) == 1

    monkeypatch.setattr(
        sys, "argv",
        [
            "build_137k_replay.py",
            "--config", str(cfg_path),
            "--new-records", "2",
            "--replay-frac", "0.5",
            "--exclude-corpora", "data/target",
            "--out", str(tmp_path / "data" / "replay_out"),
        ],
    )
    assert mod.main() == 0
    written = [
        json.loads(l)["input"]
        for l in (tmp_path / "data" / "replay_out.train.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert set(written) == {"Clean doc 1", "Clean doc 2"}


def test_normalize_group_key_strips_invisibles_matching_dumps_record():
    data_dir = str(ROOT / "tools" / "data")
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    from _split import dumps_record, normalize_group_key

    raw = "Photographs of two Allegheny\u00ad County men"
    written = json.loads(dumps_record({"input": raw}))["input"]
    assert "\u00ad" not in written
    assert normalize_group_key(raw) == normalize_group_key(written)

