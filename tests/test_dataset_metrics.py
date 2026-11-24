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


# "John left. He waved." — John 0:4, left 5:9, He 11:13, waved 14:19.
COREF_RECORD = {
    "input": "John left. He waved.",
    "output": {"events": [
        {"event_type": "Leave",
         "triggers": [{"text": "left", "start": 5, "end": 9},
                      {"text": "waved", "start": 14, "end": 19}],
         "arguments": [{"role": "Person", "entity": {"text": "John", "start": 0, "end": 4}}]},
        {"event_type": "Leave",
         "triggers": [{"text": "left", "start": 5, "end": 9}],
         "arguments": [{"role": "Person", "entity": {"text": "he", "start": 11, "end": 13}}]},
    ]},
    "coref_clusters": [[
        {"text": "John", "start": 0, "end": 4},
        {"text": "He", "start": 11, "end": 13},
    ]],
}


def test_coref_counts_trigger_pairs_and_argument_expansion(tmp_path):
    f = tmp_path / "x.train.jsonl"
    _write(f, [COREF_RECORD])
    m = dm.scan_file(f)
    assert m["coref_docs"] == 1
    assert m["coref_clusters"] == 1
    assert m["coref_mentions"] == 2
    assert m["coref_mention_pairs"] == 1
    assert m["coref_pronouns"] == 1
    # One event lists both triggers: one positive pair. The third trigger is a second
    # event of the same type, so the other two pairs are cross-event.
    assert (m["trig_coref_events"], m["trig_coref_pos"], m["trig_coref_cross"]) == (1, 1, 2)
    assert m["id_multi"] == m["trig_coref_events"]
    # "John" is in one cluster and gains "He". "he" does not match the mention text "He".
    assert (m["arg_coref_added"], m["arg_coref_skipped"]) == (1, 0)
    # Both argument spans sit in the cluster, and each has the other mention.
    assert (m["arg_span_coref"], m["arg_span_others"], m["arg_span_ambiguous"]) == (2, 2, 0)


def test_span_geometry_partitions_inside_and_outside(tmp_path):
    f = tmp_path / "x.train.jsonl"
    _write(f, [{"input": "xxxx yyyy zzzz", "output": {"events": [{
        "event_type": "E",
        "triggers": [{"text": "yyyy", "start": 5, "end": 9}],
        "arguments": [
            {"role": "eq", "entity": {"text": "yyyy", "start": 5, "end": 9}},
            {"role": "in", "entity": {"text": "yy", "start": 5, "end": 7}},
            {"role": "over", "entity": {"text": "yy z", "start": 7, "end": 11}},
            {"role": "out", "entity": {"text": "xxxx", "start": 0, "end": 4}},
            {"role": "str", "entity": "somewhere"},
        ],
    }]}}])
    m = dm.scan_file(f)
    assert m["arg_place"]["inside"] == m["arg_equal"] + m["arg_nested"] == 2
    assert m["arg_place"]["outside"] == (
        m["arg_trigger_inside"] + m["arg_partial"] + m["arg_disjoint"]) == 2
    assert (m["arg_equal"], m["arg_nested"], m["arg_partial"], m["arg_disjoint"]) == (1, 1, 1, 1)
    assert m["arg_place"]["unmeasured"] == 1


def test_markdown_names_tvd_and_coref(tmp_path):
    _write(tmp_path / "x.train.jsonl", [
        COREF_RECORD,
        {"input": "aaa", "output": {"entities": {"A": ["a", "a", "a"]}}},
    ])
    _write(tmp_path / "x.val.jsonl", [{
        "input": "bbb", "output": {"entities": {"A": ["a"], "B": ["b", "b", "b"]}},
    }])
    text = dm.markdown_report(dm.collect(tmp_path, ["x"]), {}, top=0)
    # Train is all A. Val is 1/4 A and 3/4 B. TVD = 0.75.
    assert "0.7500 STRONG" in text
    assert "### Coreference" in text
    assert "positive pairs" in text
    assert "mentions the string rule adds" in text
    bare = dm.markdown_report({"z": {"train": dm.new_metrics()}}, {}, top=0)
    assert "No `coref_clusters`." in bare


def test_language_section_lists_every_language_and_its_records():
    train, val = dm.new_metrics(), dm.new_metrics()
    train["records"], val["records"] = 10, 4
    train["languages"] = Counter({"eng": 7, "fra": 2, "deu": 1})
    val["languages"] = Counter({"eng": 1, "zho": 3})
    text = dm.markdown_report({"x": {"train": train, "val": val}}, {}, top=0)
    assert "| eng | 7 | 1 | 8 | 57.14% |" in text
    assert "| zho | 0 | 3 | 3 | 21.43% |" in text
    assert "| fra | 2 | 0 | 2 | 14.29% |" in text
    assert "| deu | 1 | 0 | 1 | 7.14% |" in text
    bare = dm.markdown_report({"z": {"train": dm.new_metrics()}}, {}, top=0)
    assert "Language was not measured." in bare


def test_tvd_gates_are_the_data_health_gates():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "derive"))
    import data_health
    assert (dm.TVD_WARN, dm.TVD_STRONG, dm.THIN_SUPPORT) == (
        data_health.TVD_WARN, data_health.TVD_STRONG, data_health.THIN_SUPPORT)
