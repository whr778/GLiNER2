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
