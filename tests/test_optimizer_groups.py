"""encoder_lr_backbone_only: only the backbone (`encoder.*`) gets encoder_lr; the head's boundary_encoder and
candidate_encoder -- caught by upstream's `"encoder" in name` rule -- move to task_lr. Default unchanged."""
import tempfile

import pytest

from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def _groups(flag):
    model = build_tiny_boundary_model()
    names = {id(p): n for n, p in model.named_parameters()}
    cfg = TrainingConfig(output_dir=tempfile.mkdtemp(), encoder_lr=2e-5, task_lr=5e-4, fp16=False, bf16=False,
                         encoder_lr_backbone_only=flag)
    opt = GLiNER2Trainer(model, cfg)._create_optimizer()
    return [[names[id(p)] for p in g["params"]] for g in opt.param_groups]


def _head_encoder(n):
    return "boundary_encoder" in n or "candidate_encoder" in n


@pytest.mark.parametrize("flag, head_on_encoder_lr", [(False, True), (True, False)])
def test_head_encoders_follow_the_flag(flag, head_on_encoder_lr):
    enc, task = _groups(flag)
    assert any(_head_encoder(n) for n in enc + task), "fixture has no head encoder to route"
    assert any(_head_encoder(n) for n in enc) == head_on_encoder_lr
    assert all(n.startswith("encoder.") for n in enc) == (not head_on_encoder_lr)
    assert not any(n.startswith("encoder.") for n in task), "a backbone tensor left encoder_lr"
