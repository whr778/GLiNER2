"""checkpoint_restart: resumable-checkpoint discovery + training-state round-trip.

These exercise the pure pieces (filesystem discovery and optimizer-state
save/restore) on a stub + a tiny CPU optimizer. The full in-loop resume is
exercised end-to-end only in a real training run.
"""

import os
import types

import torch
from torch.optim import AdamW

from gliner2.training.trainer import GLiNER2Trainer as T


def _stub(output_dir, mode):
    return types.SimpleNamespace(
        output_dir=output_dir, config=types.SimpleNamespace(checkpoint_restart=mode)
    )


def test_find_resume_checkpoint_highest_and_last(tmp_path):
    for n in ("checkpoint-epoch-1", "checkpoint-epoch-3", "checkpoint-epoch-2", "best", "final"):
        (tmp_path / n).mkdir()
    for n in ("checkpoint-epoch-1", "checkpoint-epoch-3", "checkpoint-epoch-2"):
        (tmp_path / n / "training_state.pt").write_bytes(b"x")  # mark resumable
    # mtimes: epoch-2 newest, epoch-3 highest number
    os.utime(tmp_path / "checkpoint-epoch-1", (1000, 1000))
    os.utime(tmp_path / "checkpoint-epoch-3", (2000, 2000))
    os.utime(tmp_path / "checkpoint-epoch-2", (3000, 3000))

    assert T._find_resume_checkpoint(_stub(tmp_path, "highest")).name == "checkpoint-epoch-3"
    assert T._find_resume_checkpoint(_stub(tmp_path, "last")).name == "checkpoint-epoch-2"


def test_find_resume_checkpoint_none_and_empty(tmp_path):
    assert T._find_resume_checkpoint(_stub(tmp_path, None)) is None          # disabled
    assert T._find_resume_checkpoint(_stub(tmp_path, "highest")) is None     # no candidates -> warn -> None
    # best/final without training_state.pt are not resumable
    (tmp_path / "best").mkdir()
    assert T._find_resume_checkpoint(_stub(tmp_path, "last")) is None


def test_training_state_round_trip(tmp_path):
    model = torch.nn.Linear(8, 8)
    opt = AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(4, 8)).sum().backward()
    opt.step()  # populate optimizer state

    saver = types.SimpleNamespace(
        optimizer=opt, scheduler=None, epoch=2, global_step=44,
        best_metric=0.42, patience_counter=1,
    )
    T._save_training_state(saver, tmp_path)
    assert (tmp_path / "training_state.pt").is_file()

    loaded = torch.load(tmp_path / "training_state.pt", map_location="cpu", weights_only=False)
    opt2 = AdamW(torch.nn.Linear(8, 8).parameters(), lr=1e-3)
    restorer = types.SimpleNamespace(
        optimizer=opt2, scheduler=None, device=torch.device("cpu"),
        best_metric=float("inf"), global_step=0, patience_counter=0,
    )
    start_epoch = T._restore_training_state(restorer, loaded)

    assert start_epoch == 3                       # saved epoch 2 -> resume at index 3
    assert restorer.global_step == 44
    assert abs(restorer.best_metric - 0.42) < 1e-9
    assert restorer.patience_counter == 1
    assert len(opt2.state) > 0                     # optimizer moments restored


def test_save_training_state_noop_without_optimizer(tmp_path):
    T._save_training_state(types.SimpleNamespace(optimizer=None), tmp_path)
    assert not (tmp_path / "training_state.pt").exists()


def test_load_checkpoint_reapplies_gradient_checkpointing(tmp_path):
    """from_pretrained returns a fresh model with GC off; load_checkpoint must
    re-enable it, else a resume silently loses the activation-memory fix."""
    class FakeEncoder:
        def __init__(self):
            self.gc = False
            self.config = types.SimpleNamespace(use_cache=True)
        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.gc = True

    class FakeModel:
        def __init__(self):
            self.encoder = FakeEncoder()
        @classmethod
        def from_pretrained(cls, d):
            return cls()
        def to(self, dev):
            return self

    ckpt = tmp_path / "checkpoint-epoch-1"
    ckpt.mkdir()  # no adapter_config.json -> the full (non-LoRA) load path
    stub = types.SimpleNamespace(
        model=FakeModel(),
        device=torch.device("cpu"),
        config=types.SimpleNamespace(use_lora=False, gradient_checkpointing=True),
        _setup_parallel=lambda: None,
        lora_layers={},
    )
    stub._setup_gradient_checkpointing = lambda: T._setup_gradient_checkpointing(stub)

    T.load_checkpoint(stub, str(ckpt))
    assert stub.model.encoder.gc is True
    assert stub.model.encoder.config.use_cache is False
