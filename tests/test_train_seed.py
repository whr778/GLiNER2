"""train.py seeds torch with the config's training seed BEFORE building the model.

The trainer's own _setup_seed runs after train.py has built the model, so until 2026-10-21 every head
initialisation was a fresh random draw and `seed` never reached the starting weights.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))
from train import _seed_model_init  # noqa: E402


def _draw(cfg):
    _seed_model_init(cfg)
    return torch.randn(4)


def test_same_seed_gives_the_same_init_draw_whatever_came_before():
    torch.manual_seed(12345)
    a = _draw({"training": {"seed": 42}})
    torch.manual_seed(999)
    b = _draw({"training": {"seed": 42}})
    assert torch.equal(a, b)


def test_a_different_seed_changes_the_draw():
    assert not torch.equal(_draw({"training": {"seed": 42}}), _draw({"training": {"seed": 43}}))


def test_missing_seed_uses_the_trainer_default():
    assert _seed_model_init({}) == 42
