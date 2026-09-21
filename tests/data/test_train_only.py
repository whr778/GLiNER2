"""`data.train_only` keeps a corpus in TRAIN and out of val and test.

An A/B whose treatment ADDS a corpus otherwise scores the two arms on different gold: roles2
ran 20,602 records against 23,326, with entity support 79,912 against 96,306, because the
added corpus brought its own test split. Comparing F1 across different gold is void, and it
survived two runs unnoticed -- the first Option 4 run's entity row was void for the same
reason -- until eval_provenance let compare_runs.py refuse it.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _train_mod():
    spec = importlib.util.spec_from_file_location("_t_only", ROOT / "tools" / "train" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _train_mod()


def test_train_only_corpus_is_kept_out_of_val_and_test(mod, monkeypatch, capsys):
    monkeypatch.setattr(mod, "_fetch_if_missing", lambda p: None)
    corpora = ["data/casie", "data/cmnee_roles_ner"]
    train = mod._split_files(corpora, "train", {"cmnee_roles_ner"})
    val = mod._split_files(corpora, "val", {"cmnee_roles_ner"})
    test = mod._split_files(corpora, "test", {"cmnee_roles_ner"})
    assert "data/cmnee_roles_ner.train.jsonl" in train, "must still TRAIN on it"
    assert not any("cmnee_roles_ner" in p for p in val), "must not reach val"
    assert not any("cmnee_roles_ner" in p for p in test), "must not reach test"
    assert "data/casie.val.jsonl" in val, "other corpora are untouched"
    assert "is train_only" in capsys.readouterr().out, "the exclusion must be NAMED, not silent"


def test_without_it_the_corpus_reaches_every_split(mod, monkeypatch):
    """The defect this prevents: both arms' splits differ when the treatment adds a corpus."""
    monkeypatch.setattr(mod, "_fetch_if_missing", lambda p: None)
    corpora = ["data/casie", "data/cmnee_roles_ner"]
    for suffix in ("train", "val", "test"):
        got = mod._split_files(corpora, suffix)
        assert any("cmnee_roles_ner" in p for p in got), f"{suffix} should include it without train_only"


def test_empty_train_only_changes_nothing(mod, monkeypatch):
    monkeypatch.setattr(mod, "_fetch_if_missing", lambda p: None)
    corpora = ["data/casie"]
    assert mod._split_files(corpora, "test") == mod._split_files(corpora, "test", set())
