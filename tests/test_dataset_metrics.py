"""Tests for scripts/dataset_metrics.py."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "dataset_metrics", Path(__file__).resolve().parent.parent / "scripts" / "dataset_metrics.py")
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)


def _write(path, records):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                    encoding="utf-8")


EVENT_RECORD = {
    "input": "Hackers breached Acme and stole data .",
    "output": {
        "entities": {"Organization": ["Acme"]},
        "events": [
            {"event_type": "Databreach", "triggers": ["breached"],
             "arguments": [{"role": "Victim", "entity": "Acme"},
                           {"role": "Attacker", "entity": "Hackers"}]},
            {"event_type": "Theft", "triggers": [],
             "arguments": [{"role": "Victim", "entity": "Acme"}]},
        ],
    },
}


def test_triggerless_event_counts_as_instance_but_not_argument_gold(tmp_path):
    f = tmp_path / "x.train.jsonl"
    _write(f, [EVENT_RECORD])
    m = dm.scan_file(f)
    assert m["ev_instances"] == 2
    assert m["ev_triggerless"] == 1
    assert m["trigger_gold"] == 1
    assert m["argument_gold"] == 2
    assert m["ent_records"] == 1


def test_trigger_only_corpus_is_flagged(tmp_path):
    f = tmp_path / "x.train.jsonl"
    _write(f, [{"input": "It exploded .", "output": {
        "events": [{"event_type": "Attack", "triggers": ["exploded"], "arguments": []}]}}])
    lines = dm._event_lines(dm.scan_file(f), top=0)
    assert any("TRIGGER-ONLY" in line for line in lines)


def test_classification_majority_share(tmp_path):
    f = tmp_path / "x.train.jsonl"
    recs = [{"input": f"doc {i}", "output": {"classifications": [
        {"task": "topic", "labels": ["a", "b"], "true_label": ["a" if i < 3 else "b"]}]}}
        for i in range(4)]
    _write(f, recs)
    lines = dm._classification_lines(dm.scan_file(f), top=0)
    assert any("majority class 75.0% of 4" in line for line in lines)


def test_dominant_language_skips_und():
    m = dm.new_metrics()
    m["languages"] = Counter({"und": 60, "zho": 40})
    assert dm.dominant_language(m) == "zho"


def test_provenance_per_head_and_unknown():
    reg = {"cmnee_ner": {"provenance": {"events": "human", "entities": "llm_real"}},
           "casie": {"provenance": "human"}}
    assert dm.provenance(reg, "cmnee_ner", "events") == "human"
    assert dm.provenance(reg, "cmnee_ner", "entities") == "llm_real"
    assert dm.provenance(reg, "scaling_joint/casie", "events") == "human"
    assert dm.provenance(reg, "text2json", "structures") == "unknown"


def test_uniqueness_reports_duplicates_and_overlap():
    def m(texts):
        x = dm.new_metrics()
        for t in texts:
            dm.add_record(x, {"input": t, "output": {}}, None)
        return x
    data = {"a": {"train": m(["one", "two", "two"]), "test": m(["two"])}}
    u = dm.uniqueness(data)
    assert u["train_records"] == 3 and u["train_unique_inputs"] == 2
    assert u["train&test"] == 1


def test_corpus_key_keeps_subdirectory():
    assert dm.corpus_key(Path("data/cmnee.test.jsonl")) == "cmnee"
    assert dm.corpus_key(Path("data/scaling_joint/cmnee.val.jsonl")) == "scaling_joint/cmnee"


def test_config_paths_honours_train_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    for split in ("train", "val", "test"):
        _write(tmp_path / "data" / f"a.{split}.jsonl", [EVENT_RECORD])
        _write(tmp_path / "data" / f"b.{split}.jsonl", [EVENT_RECORD])
    cfg = {"data": {"corpora": ["data/a", "data/b"], "train_only": ["b"]}}
    paths = dm.config_paths(cfg)
    assert paths["train"] == ["data/a.train.jsonl", "data/b.train.jsonl"]
    assert paths["test"] == ["data/a.test.jsonl"]


def test_labels_option_maps_before_counting(tmp_path):
    f = tmp_path / "x.train.jsonl"
    _write(f, [{"input": "Ada .", "output": {"entities": {"person": ["Ada"]}}}])
    m = tmp_path / "map.yaml"
    m.write_text("labels:\n  entities:\n    map:\n      person: Person\n", encoding="utf-8")
    assert list(dm.collect(tmp_path, [], fns=dm.load_label_fns(str(m)))["x"]["train"]["entities"]) == ["Person"]
    assert list(dm.collect(tmp_path, [])["x"]["train"]["entities"]) == ["person"]
