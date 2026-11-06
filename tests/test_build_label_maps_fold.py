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


def test_canonical_config_includes_event_files_and_refuses_missing(tmp_path):
    ev_val = tmp_path / "ev.val.jsonl"
    ev_val.write_text(
        json.dumps({"input": "x", "output": {"events": [{"event_type": "CyberAttack", "arguments": [{"role": "Victim"}]}]}}) + "\n",
        encoding="utf-8",
    )
    corp_train = tmp_path / "corp.train.jsonl"
    corp_train.write_text(
        json.dumps({"input": "y", "output": {"entities": {"Person": ["y"]}}}) + "\n",
        encoding="utf-8",
    )
    downstream = tmp_path / "down.train.jsonl"
    downstream.write_text(
        json.dumps({"input": "z", "output": {"events": [
            {"event_type": "cyber_attack", "arguments": [{"role": "victim"}]},
            {"event_type": "cyber_attack", "arguments": [{"role": "victim"}]},
        ]}}) + "\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        f"data:\n  corpora:\n    - {tmp_path / 'corp'}\n  event_files:\n    ev:\n      val: {ev_val}\n",
        encoding="utf-8",
    )
    # Without --canonical, downstream's 2x 'cyber_attack' / 'victim' would outvote ev_val's 1x 'CyberAttack' / 'Victim'.
    # With --canonical base.yaml, event_files is scanned as canonical and 'CyberAttack' / 'Victim' win.
    maps, _ = B.build([str(cfg), str(downstream)], canonical=[str(cfg)])
    assert maps["events"]["cyber_attack"] == "CyberAttack"
    assert maps["events"]["victim"] == "Victim"


def test_config_files_respects_train_only_and_validates_partial_annotation(tmp_path):
    import pytest
    for split, label in (("train", "Person"), ("val", "person"), ("test", "person")):
        (tmp_path / f"to_corp.{split}.jsonl").write_text(
            json.dumps({"input": "x", "output": {"entities": {label: ["x"]}}}) + "\n",
            encoding="utf-8",
        )
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"data:\n  corpora:\n    - {tmp_path / 'to_corp'}\n  train_only:\n    - to_corp\n"
        f"  partial_annotation:\n    to_corp:\n      - entities\n",
        encoding="utf-8",
    )
    files = B.config_files(str(cfg))
    assert files == [str(tmp_path / "to_corp.train.jsonl")]

    bad_partial = tmp_path / "bad_partial.yaml"
    bad_partial.write_text(
        f"data:\n  corpora:\n    - {tmp_path / 'to_corp'}\n  partial_annotation:\n    typo_corp:\n      - entities\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="partial_annotation"):
        B.config_files(str(bad_partial))
