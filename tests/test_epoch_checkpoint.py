"""save_epoch_checkpoint: with eval_strategy "steps", an epoch-end checkpoint-epoch-N (with resume state) is written.

Resume restarts at the NEXT epoch with no mid-epoch batch skip, so only an epoch-end checkpoint resumes without
silently skipping data. The real trainer, the tiny boundary model, one epoch.
"""
import torch
import pytest

from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

RECORDS = [{"input": f"Ann met Bob in Paris number {i}.", "output": {"entities": {"person": ["Ann", "Bob"]}}}
           for i in range(3)]


def _train(tmp_path, flag):
    model = build_tiny_boundary_model().train()
    cfg = TrainingConfig(output_dir=str(tmp_path), num_epochs=1, batch_size=1, fp16=False, bf16=False,
                         eval_strategy="steps", eval_steps=10_000, save_total_limit=5, num_workers=0,
                         pin_memory=False, validate_data=False, report_to_wandb=False, logging_steps=1,
                         save_epoch_checkpoint=flag)
    GLiNER2Trainer(model, cfg).train(train_data=RECORDS)
    return tmp_path / "checkpoint-epoch-1"


def test_epoch_checkpoint_written_with_resume_state(tmp_path):
    ckpt = _train(tmp_path, True)
    assert ckpt.is_dir()
    state = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["epoch"] == 0 and state["param_names"] and all(state["param_names"])


@pytest.mark.parametrize("flag", [False])
def test_no_epoch_checkpoint_by_default(tmp_path, flag):
    assert not _train(tmp_path, flag).exists()
