"""Numeric unit tests for the structure-loss variants in gliner2/model.py.

The loss methods (``_struct_loss_term`` / ``_dice_struct_loss``) read only
``self.config`` and operate on tensors, so we exercise them via a tiny stub
holding an :class:`ExtractorConfig` -- no encoder download, runs on CPU.

Each variant must produce a finite, positive, differentiable loss; the
non-BCE variants must actually differ from plain BCE (i.e. they are not
silently no-ops).
"""

from types import SimpleNamespace

import pytest
import torch

from gliner2.model import Extractor, ExtractorConfig

# scores grid: (gold_count, fields, starts, widths) -- same layout as
# compute_struct_loss builds it.
G, F, S, W = 2, 3, 8, 5


def _stub(struct_loss, **knobs):
    cfg = ExtractorConfig(model_name="x", max_width=W, max_len=64, struct_loss=struct_loss, **knobs)
    return SimpleNamespace(config=cfg)


@pytest.fixture
def grid():
    """Fresh (scores, labs) each test so backward() doesn't leak across cases."""
    torch.manual_seed(0)
    scores = torch.randn(G, F, S, W, requires_grad=True)
    labs = torch.zeros(G, F, S, W)
    labs[0, 0, 1, 2] = 1.0
    labs[1, 2, 3, 1] = 1.0
    labs[0, 1, 5, 0] = 1.0
    return scores, labs


# (name, knobs) for the per-cell variants routed through _struct_loss_term.
TERM_VARIANTS = [
    ("bce", {}),
    ("bce_posweight", {"struct_pos_weight": 8.0}),
    ("focal", {"focal_gamma": 2.0, "focal_alpha": 0.25}),
    ("asl", {"asl_gamma_pos": 1.0, "asl_gamma_neg": 4.0, "asl_clip": 0.05}),
]


@pytest.mark.parametrize("name,knobs", TERM_VARIANTS, ids=[v[0] for v in TERM_VARIANTS])
def test_struct_loss_term_is_finite_and_differentiable(name, knobs, grid):
    scores, labs = grid
    loss = Extractor._struct_loss_term(_stub(name, **knobs), scores, labs)

    assert loss.shape == scores.shape, "term loss should be unreduced (per-cell)"
    assert torch.isfinite(loss).all()
    total = loss.sum()
    assert total.item() > 0.0

    total.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0.0, "no gradient flowed to scores"


@pytest.mark.parametrize("name,knobs", TERM_VARIANTS[1:], ids=[v[0] for v in TERM_VARIANTS[1:]])
def test_nonbce_term_variants_differ_from_bce(name, knobs, grid):
    scores, labs = grid
    bce = Extractor._struct_loss_term(_stub("bce"), scores, labs)
    variant = Extractor._struct_loss_term(_stub(name, **knobs), scores, labs)
    assert not torch.allclose(bce, variant), f"{name} produced the same per-cell loss as bce"


@pytest.mark.parametrize("include_bce,name", [(False, "dice"), (True, "bce_dice")])
def test_dice_variants_are_finite_and_differentiable(include_bce, name, grid):
    scores, labs = grid
    span_mask = torch.zeros(1, S * W, dtype=torch.bool)  # all spans valid
    loss = Extractor._dice_struct_loss(_stub(name, dice_smooth=1.0), scores, labs, span_mask, include_bce)

    assert loss.dim() == 0, "dice loss should be reduced to a scalar"
    assert torch.isfinite(loss).all()
    assert loss.item() > 0.0

    loss.backward()
    assert scores.grad is not None and scores.grad.abs().sum().item() > 0.0


def test_bce_dice_exceeds_plain_dice(grid):
    scores, labs = grid
    span_mask = torch.zeros(1, S * W, dtype=torch.bool)
    dice = Extractor._dice_struct_loss(_stub("dice", dice_smooth=1.0), scores, labs, span_mask, False)
    bce_dice = Extractor._dice_struct_loss(_stub("bce_dice", dice_smooth=1.0), scores, labs, span_mask, True)
    assert bce_dice.item() > dice.item(), "bce_dice should add a positive BCE term on top of dice"
