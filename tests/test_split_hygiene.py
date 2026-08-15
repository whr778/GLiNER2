"""The pre-training gate: train/val/test must be mutually disjoint.

Both live configs failed this when it was written -- joint-boundary-mmbert-137k
shared 1,080 documents between train and test (7.03% of the blind set).
"""
from __future__ import annotations

import pytest

from gliner2.training.split_hygiene import check_and_clean, enforce_disjoint_splits


def rec(text, target="a"):
    return {"input": text, "output": {"entities": {"T": [target]}}}


def test_test_set_is_authoritative_and_never_modified():
    """Removal always comes out of the LOWER-priority split, so the blind set is
    exactly what the corpus said it was."""
    train = [rec("shared doc"), rec("train only")]
    val = [rec("shared doc"), rec("val only")]
    test = [rec("shared doc"), rec("test only")]

    cleaned, report = enforce_disjoint_splits({"train": train, "val": val, "test": test})

    assert [r["input"] for r in cleaned["test"]] == ["shared doc", "test only"]
    assert [r["input"] for r in cleaned["val"]] == ["val only"]
    assert [r["input"] for r in cleaned["train"]] == ["train only"]
    assert report.contaminated[("val", "test")] == 1
    assert report.contaminated[("train", "test")] == 1


def test_val_yields_to_test_and_train_yields_to_both():
    train = [rec("in val"), rec("in test"), rec("clean")]
    val = [rec("in val")]
    test = [rec("in test")]

    cleaned, report = enforce_disjoint_splits({"train": train, "val": val, "test": test})

    assert [r["input"] for r in cleaned["train"]] == ["clean"]
    assert report.contaminated[("train", "val")] == 1
    assert report.contaminated[("train", "test")] == 1


def test_same_document_with_a_different_target_is_kept_within_a_split():
    """text2json emits one document up to 10x with 8 distinct schemas -- that is
    the schema-conditioning signal, not duplication. Dropping on text alone would
    delete it."""
    train = [rec("doc", "first"), rec("doc", "second"), rec("doc", "third")]

    cleaned, report = enforce_disjoint_splits({"train": train})

    assert len(cleaned["train"]) == 3
    assert not report.exact_duplicates


def test_exact_repeats_within_a_split_are_dropped():
    train = [rec("doc", "same"), rec("doc", "same"), rec("doc", "other")]

    cleaned, report = enforce_disjoint_splits({"train": train})

    assert len(cleaned["train"]) == 2
    assert report.exact_duplicates["train"] == 1


def test_cross_split_uses_TEXT_not_the_target():
    """A document in both train and test is contamination even when the targets
    differ -- the model has still seen the text."""
    train = [rec("doc", "train target")]
    test = [rec("doc", "test target")]

    cleaned, report = enforce_disjoint_splits({"train": train, "test": test})

    assert cleaned["train"] == []
    assert len(cleaned["test"]) == 1
    assert report.contaminated[("train", "test")] == 1


def test_matching_ignores_case_and_whitespace():
    train = [rec("The  Quick   Brown Fox")]
    test = [rec("the quick brown fox")]

    cleaned, _ = enforce_disjoint_splits({"train": train, "test": test})

    assert cleaned["train"] == []


def test_clean_splits_pass_through_untouched():
    train, val, test = [rec("a")], [rec("b")], [rec("c")]

    cleaned, report = enforce_disjoint_splits(
        {"train": train, "val": val, "test": test}
    )

    assert report.clean
    assert [len(cleaned[s]) for s in ("train", "val", "test")] == [1, 1, 1]


def test_policy_raise_refuses_to_train():
    with pytest.raises(ValueError, match="not disjoint"):
        check_and_clean([rec("doc")], None, [rec("doc")], policy="raise")


def test_policy_warn_reports_but_changes_nothing():
    train, _, test, report = check_and_clean(
        [rec("doc"), rec("other")], None, [rec("doc")], policy="warn"
    )

    assert len(train) == 2 and len(test) == 1
    assert not report.clean


def test_missing_splits_are_skipped():
    train, val, test, report = check_and_clean([rec("a")], None, None)

    assert len(train) == 1 and val is None and test is None
    assert report.clean
