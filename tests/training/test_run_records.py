"""A training run leaves its record on disk and in the log: every eval, the best eval, and every per-head loss.

Until 2026-10-20 `_write_eval_metrics` had no caller (since 3e05e97, 2026-08-05), so no run wrote eval_metrics.json,
and TrainingMetrics dropped every per-head loss, so whether record_field_loss fell could not be read from any log.
A hand-stopped eb22c run at the office left three checkpoints and no numbers.
"""
import json
import logging

from gliner2.training.data import create_entity_example
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def _train(tmp_path, caplog):
    examples = [create_entity_example("apple released iphone .", {"company": ["apple"], "product": ["iphone"]},
                                      {"company": "a company", "product": "a product"})] * 2
    cfg = TrainingConfig(output_dir=str(tmp_path / "out"), batch_size=1, num_epochs=2, eval_strategy="epoch",
                         fp16=False, bf16=False, num_workers=0, logging_steps=1, warmup_ratio=0.0,
                         scheduler_type="constant", local_rank=-1)
    with caplog.at_level(logging.INFO):
        GLiNER2Trainer(build_tiny_boundary_model(), cfg, eval_data=examples).train(train_data=examples)
    return tmp_path / "out"


def test_every_eval_is_kept_and_the_best_one_is_marked(tmp_path, caplog):
    out = _train(tmp_path, caplog)
    rows = [json.loads(line) for line in open(out / "eval_history.jsonl", encoding="utf-8")]
    assert [r["eval_epoch"] for r in rows] == [1, 2]
    assert all("eval_loss" in r for r in rows)
    best = json.loads((out / "eval_metrics.json").read_text(encoding="utf-8"))
    assert best["is_best"] and best["eval_epoch"] in (1, 2)
    assert best == next(r for r in reversed(rows) if r["is_best"])
    assert json.loads((out / "best" / "eval_metrics.json").read_text(encoding="utf-8")) == best


def test_every_per_head_loss_reaches_the_log(tmp_path, caplog):
    _train(tmp_path, caplog)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[losses] step")]
    assert len(lines) == 4                                   # 2 epochs x 2 steps, logging every step
    # start/end are per-head losses TrainingMetrics never carried; record_field_loss appears only on batches with
    # events or structures (it is on every line of a real eb22c run), and these examples are entity-only.
    assert all(k in lines[0] for k in ("start_loss=", "end_loss=", "total_loss="))
