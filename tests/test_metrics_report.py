"""Tests for the per-eval micro precision/recall/F1 summary printer."""

from gliner2.training.metrics import _print_micro_report


def test_micro_report_prints_only_present_categories(capsys):
    metrics = {
        "eval_entity_micro_precision": 0.5,
        "eval_entity_micro_recall": 0.25,
        "eval_entity_micro_f1": 0.3333,
        "eval_relation_micro_precision": 0.1,
        "eval_relation_micro_recall": 0.2,
        "eval_relation_micro_f1": 0.1333,
    }
    _print_micro_report(metrics)
    out = capsys.readouterr().out

    assert "micro precision / recall / f1" in out
    assert "entity" in out and "relation" in out
    # values are formatted to 4 dp
    assert "P=0.5000" in out and "R=0.2500" in out and "F1=0.3333" in out
    # categories absent from the dict must not be printed
    assert "classification" not in out
    assert "event_trigger" not in out
    assert "event_argument" not in out


def test_micro_report_prints_nothing_when_empty(capsys):
    _print_micro_report({})
    assert capsys.readouterr().out == ""
