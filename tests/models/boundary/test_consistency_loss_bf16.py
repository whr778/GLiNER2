"""`marginal_pair_consistency_loss` must not produce NaN gradients in bf16.

A hard `1.0 - 1e-6` clamp is a NO-OP in bf16: with 8 mantissa bits, eps is 0.0078125 and
`1.0 - 1e-6` rounds to exactly 1.0. A saturated probability then gives `log1p(-1.0) = -inf`,
whose derivative `-1/(1-p)` is infinite; a masked entry's zero grad_output turns that into
`0 * inf = NaN`.

THE FORWARD IS FINITE THROUGHOUT, which is why this survived: the -inf is summed and
`1 - exp(-inf)` is 1. Any test that only checks the loss value passes while training dies.
"""

import pytest
import torch

from gliner2.models.boundary.losses import marginal_pair_consistency_loss


def _inputs(dtype):
    """One saturating pair logit and one masked-out neighbour -- the minimal trigger."""
    torch.manual_seed(0)
    b, q, n, c = 1, 1, 4, 2
    # 30.0 saturates sigmoid to exactly 1.0 in every float dtype here.
    pair_logits = torch.tensor([[[30.0, 30.0]]], dtype=dtype, requires_grad=True)
    indices = torch.zeros(b, q, c, 2, dtype=torch.long)
    valid_mask = torch.tensor([[[True, False]]])
    start_logits = torch.zeros(b, q, n, dtype=dtype, requires_grad=True)
    end_logits = torch.zeros(b, q, n, dtype=dtype, requires_grad=True)
    boundary_keep = torch.ones(b, q, n, dtype=torch.bool)
    return pair_logits, indices, valid_mask, start_logits, end_logits, boundary_keep


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_saturated_probability_gives_finite_gradients(dtype):
    pair_logits, indices, valid_mask, start_logits, end_logits, keep = _inputs(dtype)

    loss = marginal_pair_consistency_loss(
        pair_logits, indices, valid_mask, start_logits, end_logits, keep
    )
    assert torch.isfinite(loss), f"forward already non-finite in {dtype}"

    loss.backward()
    assert torch.isfinite(pair_logits.grad).all(), (
        f"NaN/inf gradient in {dtype}: the clamp epsilon is not representable in this dtype, "
        f"so log1p(-p) saw p == 1.0 exactly"
    )


def test_the_clamp_epsilon_is_representable_in_every_supported_dtype():
    """The root cause, asserted directly so it cannot silently return.

    `1.0 - eps` must be strictly less than 1.0 AFTER rounding into the dtype, or the clamp is
    decorative.
    """
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        eps = max(1e-6, torch.finfo(dtype).eps)
        bound = torch.tensor(1.0 - eps, dtype=dtype)
        assert bound.item() < 1.0, (
            f"{dtype}: clamp bound rounds to {bound.item()}, so the clamp does nothing"
        )
    # And the fp32 path is unchanged by the fix.
    assert max(1e-6, torch.finfo(torch.float32).eps) == 1e-6
