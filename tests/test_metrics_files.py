"""The trainer writes eval metrics to output_dir and the best/ checkpoint folder."""

import json
import types

from gliner2.training.trainer import GLiNER2Trainer

METRICS = {
    "eval_loss": 0.5,
    "eval_event_strict_micro_f1": 0.9,
    "eval_entity_strict_classification_report": "line1\nline2",
    "step": 100,
    "epoch": 2,
}


def test_eval_metrics_written_to_output_dir_and_best(tmp_path):
    best = tmp_path / "best"
    best.mkdir()
    GLiNER2Trainer._write_eval_metrics(
        types.SimpleNamespace(output_dir=tmp_path), METRICS
    )
    top = json.loads((tmp_path / "eval_metrics.json").read_text())
    in_best = json.loads((best / "eval_metrics.json").read_text())
    assert top == METRICS and in_best == METRICS
    # multi-line classification report survives the JSON round-trip
    assert "\n" in in_best["eval_entity_strict_classification_report"]


def test_eval_metrics_output_dir_only_when_no_best(tmp_path):
    GLiNER2Trainer._write_eval_metrics(
        types.SimpleNamespace(output_dir=tmp_path), METRICS
    )
    assert (tmp_path / "eval_metrics.json").is_file()
    assert not (tmp_path / "best" / "eval_metrics.json").exists()
