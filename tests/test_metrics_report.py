"""Tests for the per-eval micro precision/recall/F1 summary printer."""

from gliner2.training.metrics import _print_micro_report


def test_micro_report_prints_strict_and_relaxed(capsys):
    metrics = {
        "eval_entity_strict_micro_precision": 0.50,
        "eval_entity_strict_micro_recall": 0.25,
        "eval_entity_strict_micro_f1": 0.3333,
        "eval_entity_relaxed_micro_precision": 0.80,
        "eval_entity_relaxed_micro_recall": 0.40,
        "eval_entity_relaxed_micro_f1": 0.5333,
    }
    _print_micro_report(metrics)
    out = capsys.readouterr().out

    assert "strict -> relaxed" in out
    assert "entity" in out
    assert "P=0.5000->0.8000" in out
    assert "F1=0.3333->0.5333" in out
    # a category with no keys is not printed
    assert "relation" not in out


def test_micro_report_prints_nothing_when_empty(capsys):
    _print_micro_report({})
    assert capsys.readouterr().out == ""
