"""checkpoint_restart: resumable-checkpoint discovery + training-state round-trip.

These exercise the pure pieces (filesystem discovery and optimizer-state
save/restore) on a stub + a tiny CPU optimizer. The full in-loop resume is
exercised end-to-end only in a real training run.
"""

import os
import types

import pytest
import torch
from torch.optim import AdamW

from gliner2.training.trainer import GLiNER2Trainer as T


def _stub(output_dir, mode):
    return types.SimpleNamespace(
        output_dir=output_dir,
        # bf16/fp16: upstream's reload path consults them before deciding whether
        # to move the model to float32.
        config=types.SimpleNamespace(checkpoint_restart=mode, bf16=False, fp16=False),
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
        model=model, optimizer=opt, scheduler=None, epoch=2, global_step=44,
        best_metric=0.42, best_epoch=2, best_step=40, patience_counter=1,
    )
    saver._optimizer_param_names = types.MethodType(T._optimizer_param_names, saver)
    T._save_training_state(saver, tmp_path)
    assert (tmp_path / "training_state.pt").is_file()

    loaded = torch.load(tmp_path / "training_state.pt", map_location="cpu", weights_only=False)
    model2 = torch.nn.Linear(8, 8)
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    restorer = types.SimpleNamespace(
        model=model2, optimizer=opt2, scheduler=None, device=torch.device("cpu"),
        best_metric=float("inf"), best_epoch=None, best_step=None, global_step=0, patience_counter=0,
    )
    restorer._optimizer_param_names = types.MethodType(T._optimizer_param_names, restorer)
    restorer._load_optimizer_state = types.MethodType(T._load_optimizer_state, restorer)
    start_epoch = T._restore_training_state(restorer, loaded)

    assert start_epoch == 3                       # saved epoch 2 -> resume at index 3
    assert restorer.global_step == 44
    assert abs(restorer.best_metric - 0.42) < 1e-9
    assert restorer.patience_counter == 1
    assert (restorer.best_epoch, restorer.best_step) == (2, 40)
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
        def float(self):          # upstream's reload calls this when not bf16/fp16
            return self
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
        # bf16/fp16: upstream's reload consults them before deciding whether to
        # move the model to float32.
        config=types.SimpleNamespace(
            use_lora=False, gradient_checkpointing=True, bf16=False, fp16=False
        ),
        _setup_distributed=lambda: None,
        lora_layers={},
    )
    stub._setup_gradient_checkpointing = lambda: T._setup_gradient_checkpointing(stub)

    T.load_checkpoint(stub, str(ckpt))
    assert stub.model.encoder.gc is True
    assert stub.model.encoder.config.use_cache is False


class _Net(torch.nn.Module):
    """`late=True` registers `b` AFTER construction, the way train.py adds the junction / links to a warm start;
    `late=False` builds it in __init__, the way a reload of that checkpoint does. Same names, different order."""

    def __init__(self, late=False):
        super().__init__()
        if not late:
            self.b = torch.nn.Linear(3, 5)
        self.a = torch.nn.Linear(4, 4)
        if late:
            self.b = torch.nn.Linear(3, 5)


def _named_stub(model, opt):
    s = types.SimpleNamespace(model=model, optimizer=opt, scheduler=None, device=torch.device("cpu"), epoch=0,
                              global_step=1, best_metric=0.0, best_epoch=0, best_step=0, patience_counter=0)
    s._optimizer_param_names = types.MethodType(T._optimizer_param_names, s)
    s._load_optimizer_state = types.MethodType(T._load_optimizer_state, s)
    return s


def _stepped(model):
    opt = AdamW(model.parameters(), lr=1e-3)
    for i, (n, p) in enumerate(model.named_parameters()):
        p.grad = torch.full_like(p, float(i + 1))
    opt.step()
    return opt


def test_restore_maps_moments_by_name_across_a_reordered_model(tmp_path):
    late = _Net(late=True)
    opt = _stepped(late)
    T._save_training_state(_named_stub(late, opt), tmp_path)
    want = {n: opt.state[p]["exp_avg"].clone() for n, p in late.named_parameters()}
    state = torch.load(tmp_path / "training_state.pt", map_location="cpu", weights_only=False)

    built = _Net(late=False)
    assert [n for n, _ in built.named_parameters()] != [n for n, _ in late.named_parameters()]
    opt2 = AdamW(built.parameters(), lr=1e-3)
    T._restore_training_state(_named_stub(built, opt2), state)
    for n, p in built.named_parameters():
        assert torch.equal(opt2.state[p]["exp_avg"], want[n]), n


def test_nameless_state_with_reordered_shapes_is_refused():
    late = _Net(late=True)
    opt = _stepped(late)
    state = {"optimizer": opt.state_dict(), "epoch": 0}          # an old checkpoint: no param_names
    built = _Net(late=False)
    with pytest.raises(RuntimeError, match="do not line up"):
        T._restore_training_state(_named_stub(built, AdamW(built.parameters(), lr=1e-3)), state)


def test_nameless_state_in_the_same_order_still_restores():
    net = _Net(late=False)
    opt = _stepped(net)
    state = {"optimizer": opt.state_dict(), "epoch": 0}
    net2 = _Net(late=False)
    opt2 = AdamW(net2.parameters(), lr=1e-3)
    T._restore_training_state(_named_stub(net2, opt2), state)
    assert len(opt2.state) == len(list(net2.parameters()))
