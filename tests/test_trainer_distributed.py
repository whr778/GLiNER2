"""Unit tests for the trainer's DistributedDataParallel setup (mocked -- no real
process group or GPUs).

Targets this branch's DDP design: the raw module stays on ``trainer.model``
(optimizer, grad-clipping, and checkpointing all use it) and only the forward
pass is wrapped in ``trainer._fwd_model``. DDP is auto-detected from
``LOCAL_RANK`` (falling back to ``config.local_rank``) and picks nccl on CUDA /
gloo on CPU. See ``_setup_device`` and ``_setup_parallel`` in the trainer.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest

from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


class _StubModel:
    """Minimal stand-in: ``.to()`` is a no-op, exposes a processor + params."""

    def __init__(self):
        self.to = Mock(return_value=self)
        self.processor = Mock()
        self.save_pretrained = Mock()

    def parameters(self):
        return iter(())


def _mock_cuda_ddp(monkeypatch, *, rank: int, world_size: int = 2):
    """Drive the trainer down the CUDA/nccl DDP branch without real GPUs.

    ``dist.is_initialized()`` tracks state so it reads False during setup (so the
    group is initialised) and True afterwards (so ``is_main_process`` uses rank).
    """
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("gliner2.training.trainer.torch.backends.cuda.enable_cudnn_sdp", Mock())
    set_device = Mock()
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.set_device", set_device)

    state = {"inited": False}
    init_pg = Mock(side_effect=lambda **kw: state.__setitem__("inited", True))
    monkeypatch.setattr("gliner2.training.trainer.dist.init_process_group", init_pg)
    monkeypatch.setattr("gliner2.training.trainer.dist.is_initialized", lambda: state["inited"])
    monkeypatch.setattr("gliner2.training.trainer.dist.is_available", lambda: True)
    monkeypatch.setattr("gliner2.training.trainer.dist.get_rank", lambda: rank)
    monkeypatch.setattr("gliner2.training.trainer.dist.get_world_size", lambda: world_size)

    # DDP wrapper stand-in exposing ``.module`` so we can assert the raw model
    # is preserved on ``trainer.model``.
    ddp_cls = Mock(side_effect=lambda model, **kw: type("DDP", (), {"module": model, "kw": kw})())
    monkeypatch.setattr("gliner2.training.trainer.DistributedDataParallel", ddp_cls)
    return Mock(set_device=set_device, init_pg=init_pg, ddp_cls=ddp_cls)


@pytest.fixture
def base_config(tmp_path):
    return TrainingConfig(output_dir=str(tmp_path / "out"), fp16=False, bf16=False)


def _cfg(config: TrainingConfig, rank: int) -> TrainingConfig:
    return replace(config, local_rank=rank)


def test_ddp_wraps_fwd_model_and_keeps_model_raw(base_config, monkeypatch):
    m = _mock_cuda_ddp(monkeypatch, rank=0)
    stub = _StubModel()

    trainer = GLiNER2Trainer(model=stub, config=_cfg(base_config, 0))

    m.init_pg.assert_called_once_with(backend="nccl")
    m.set_device.assert_called_once_with(0)
    m.ddp_cls.assert_called_once()
    assert m.ddp_cls.call_args.kwargs["device_ids"] == [0]
    assert m.ddp_cls.call_args.kwargs["find_unused_parameters"] is True

    # Our design: the wrapper is _fwd_model; trainer.model stays the raw stub.
    assert trainer.model is stub
    assert trainer._fwd_model is not stub
    assert trainer._fwd_model.module is stub
    assert trainer.is_distributed is True
    assert trainer.is_main_process is True


def test_non_zero_rank_is_not_main(base_config, monkeypatch):
    _mock_cuda_ddp(monkeypatch, rank=1)

    trainer = GLiNER2Trainer(model=_StubModel(), config=_cfg(base_config, 1))

    assert trainer.is_distributed is True
    assert trainer.is_main_process is False


def test_init_process_group_failure_propagates(base_config, monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("gliner2.training.trainer.torch.backends.cuda.enable_cudnn_sdp", Mock())
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.set_device", Mock())
    monkeypatch.setattr("gliner2.training.trainer.dist.is_initialized", lambda: False)
    monkeypatch.setattr(
        "gliner2.training.trainer.dist.init_process_group",
        Mock(side_effect=RuntimeError("pg init failed")),
    )

    with pytest.raises(RuntimeError, match="pg init failed"):
        GLiNER2Trainer(model=_StubModel(), config=_cfg(base_config, 0))


def test_single_device_is_not_distributed(base_config, monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("gliner2.training.trainer.torch.backends.cuda.enable_cudnn_sdp", Mock())

    trainer = GLiNER2Trainer(model=_StubModel(), config=_cfg(base_config, -1))

    assert trainer.is_distributed is False
    assert trainer._fwd_model is trainer.model  # no wrap on a single device
