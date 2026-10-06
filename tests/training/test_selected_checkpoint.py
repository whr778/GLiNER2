"""best/ records WHICH checkpoint it is: epoch, step, metric, value -- in config.json and the card.

Before 2026-10-06 the selected epoch lived only in the training log. Real 3-epoch run on the
tiny span model; greater_is_better=True on eval_loss makes epoch 1 the best, so "records the
best" and "records the latest" disagree and the test can fail.
"""
import json
import sys
from pathlib import Path

from gliner2.configuration import ExtractorConfig
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "train"))
from model_card import selected_checkpoint_text  # noqa: E402

RECS = [{"input": "Alice works at Acme in Paris.", "output": {"entities": {"person": ["Alice"], "organization": ["Acme"]}}},
        {"input": "Bob joined Globex last year.", "output": {"entities": {"person": ["Bob"], "organization": ["Globex"]}}}] * 4


def test_best_records_its_own_epoch_not_the_last(tmp_path):
    cfg = TrainingConfig(output_dir=str(tmp_path), num_workers=0, fp16=False, num_epochs=3, batch_size=4,
                         eval_strategy="epoch", save_best=True, metric_for_best="eval_loss",
                         greater_is_better=True, logging_steps=1)
    res = ExtractorTrainer(model=build_tiny_span_model(), config=cfg).train(RECS, RECS[:2])
    losses = [m["eval_loss"] for m in res["eval_metrics_history"]]
    want = losses.index(max(losses)) + 1
    assert want != len(losses), "fixture must make an epoch other than the last the best"
    assert res["best_epoch"] == want
    saved = json.loads((tmp_path / "best" / "config.json").read_text())["selected_checkpoint"]
    assert saved["epoch"] == want and saved["step"] == res["best_step"] and saved["metric"] == "eval_loss"
    assert ExtractorConfig.from_pretrained(tmp_path / "best").selected_checkpoint == saved
    assert selected_checkpoint_text(res, cfg).startswith(f"epoch {want} of 3 (step ")


def test_card_says_nothing_when_no_selection_was_recorded():
    assert selected_checkpoint_text({}, TrainingConfig()) == "—"
