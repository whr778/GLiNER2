"""Regression tests: label folding must never merge across scripts."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
import build_label_maps as B  # noqa: E402


def test_fold_keeps_non_latin_scripts_distinct():
    folds = {B.fold(s) for s in ("학교", "イベント名", "Портал", "ø")}
    assert "" not in folds and len(folds) == 4


def test_fold_still_unifies_case_and_separators():
    assert B.fold("jobTitle") == B.fold("job_title") == B.fold("Job Title")


def test_follower_labels_never_map_onto_an_unrelated_script(tmp_path):
    voter, follower = tmp_path / "v.train.jsonl", tmp_path / "f.train.jsonl"
    voter.write_text(json.dumps({"input": "x", "output": {"entities": {"イベント名": ["x"]}}}) + "\n",
                     encoding="utf-8")
    follower.write_text(json.dumps({"input": "y", "output": {"entities": {"학교": ["y"], "?": ["y"]}}}) + "\n",
                        encoding="utf-8")
    maps, _ = B.build([str(voter)], None, [str(follower)])
    assert "학교" not in maps["entities"] and "?" not in maps["entities"]


def test_labels_by_category_handles_list_entities_dict_args_absent_events_and_head_tail_relations():
    rec = {
        "input": "Alice works for Acme.",
        "output": {
            "entities": [{"Person": ["Alice"], "Organization": ["Acme"]}],
            "events": [{"event_type": "Attack", "arguments": {"Attacker": ["Alice"], "Target": ["Acme"]}}],
            "absent_events": {"Databreach": ["Compromised-Data"]},
            "relations": [{"head": "Alice", "tail": "Acme", "relation": "works_for"}],
            "record_metadata": {"casualty_report": {"mode": "natural", "anchor": "location"}},
        },
    }
    got = B.labels_by_category(rec)
    assert ("entities", "Person") in got
    assert ("entities", "Organization") in got
    assert ("relations", "works_for") in got
    assert ("relations", "head") not in got
    assert ("events", "Attack") in got
    assert ("events", "Attacker") in got
    assert ("events", "Target") in got
    assert ("events", "Databreach") in got
    assert ("events", "Compromised-Data") in got
    assert ("structures", "casualty_report") in got
    assert ("structures", "location") in got
