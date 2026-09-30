"""Tests for data.labels_passthrough: zero-shot corpora bypass the label map."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
import train as T  # noqa: E402

FNS = T._category_fns({"entities": {"map": {"person": "Person"}}})


def _write(path, labels):
    path.write_text(json.dumps({"input": "Ada met Bob .", "output": {
        "entities": {lab: ["Ada"] for lab in labels}}}) + "\n", encoding="utf-8")


def test_passthrough_corpus_keeps_its_labels_while_others_are_mapped(tmp_path):
    zs, tax = tmp_path / "zeroshot.train.jsonl", tmp_path / "taxo.train.jsonl"
    _write(zs, ["person"]); _write(tax, ["person"])
    out = T.read_transformed([str(zs), str(tax)], FNS, {"zeroshot"})
    assert list(out[0]["output"]["entities"]) == ["person"]
    assert list(out[1]["output"]["entities"]) == ["Person"]


def test_without_passthrough_every_corpus_is_mapped(tmp_path):
    zs = tmp_path / "zeroshot.train.jsonl"
    _write(zs, ["person"])
    assert list(T.read_transformed([str(zs)], FNS, set())[0]["output"]["entities"]) == ["Person"]


def test_a_passthrough_name_the_config_does_not_load_is_refused():
    cfg = {"data": {"corpora": ["data/nuner_full"], "labels_passthrough": ["nuner_ful"]}}
    with pytest.raises(SystemExit, match="nuner_ful"):
        T.labels_passthrough(cfg)
