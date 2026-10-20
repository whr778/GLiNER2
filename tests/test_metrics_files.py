"""The trainer writes eval metrics to output_dir and the best/ checkpoint folder.

These called `_write_eval_metrics` directly and passed while NOTHING in the trainer called it (3e05e97 to 2026-10-20);
tests/training/test_run_records.py now proves a real train() writes the files."""

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


def _trainer(tmp_path):
    return types.SimpleNamespace(output_dir=tmp_path, is_main_process=True, epoch=1, global_step=100)


def test_eval_metrics_written_to_output_dir_and_best(tmp_path):
    best = tmp_path / "best"
    best.mkdir()
    GLiNER2Trainer._write_eval_metrics(_trainer(tmp_path), METRICS, True)
    top = json.loads((tmp_path / "eval_metrics.json").read_text())
    in_best = json.loads((best / "eval_metrics.json").read_text())
    expected = {**METRICS, "eval_epoch": 2, "eval_step": 100, "is_best": True}
    assert top == expected and in_best == expected
    # multi-line classification report survives the JSON round-trip
    assert "\n" in in_best["eval_entity_strict_classification_report"]


def test_eval_metrics_output_dir_only_when_no_best(tmp_path):
    GLiNER2Trainer._write_eval_metrics(_trainer(tmp_path), METRICS, True)
    assert (tmp_path / "eval_metrics.json").is_file()
    assert not (tmp_path / "best" / "eval_metrics.json").exists()


def test_a_non_best_eval_goes_to_the_history_only(tmp_path):
    GLiNER2Trainer._write_eval_metrics(_trainer(tmp_path), METRICS, False)
    assert not (tmp_path / "eval_metrics.json").exists()
    rows = [json.loads(line) for line in open(tmp_path / "eval_history.jsonl")]
    assert len(rows) == 1 and rows[0]["is_best"] is False and rows[0]["eval_loss"] == 0.5
