"""fg-GEGLU: training draws disjoint random activated and skipped chunks; activated slots hold the product of the
GELU'd activated chunks, skipped chunks pass through, the rest get GELU; inference is plain GELU or the mean of K
random gate patterns."""
import pytest
import torch
import torch.nn.functional as F

from gliner2.models.boundary.flipped_gate import FlippedGateGELU


def test_eval_mode_gelu_is_exactly_gelu():
    m = FlippedGateGELU(chunks=4, gates=2, skipped=1, eval_mode="gelu").eval()
    x = torch.randn(3, 5, 16)
    assert torch.equal(m(x), F.gelu(x))


def test_eval_mode_mc_is_the_mean_of_k_gate_patterns():
    m = FlippedGateGELU(chunks=6, gates=2, skipped=2, eval_mode="mc", mc_samples=4).eval()
    x = torch.randn(5, 12)
    torch.manual_seed(7)
    y = m(x)
    torch.manual_seed(7)
    expected = torch.stack([m.gated(x) for _ in range(4)]).mean(0)
    assert torch.allclose(y, expected)
    assert not torch.equal(y, F.gelu(x))                       # it is not the plain-GELU rule


def test_training_keeps_width():
    m = FlippedGateGELU(chunks=4, gates=2, skipped=1).train()
    x = torch.randn(2, 7, 12)
    assert m(x).shape == x.shape


def test_draws_are_disjoint_random_and_cover_every_chunk():
    m = FlippedGateGELU(chunks=6, gates=2, skipped=2).train()
    seen_active, seen_skipped = set(), set()
    for _ in range(300):
        a, s = m.draw(torch.device("cpu"))
        assert len(a) == 2 and len(s) == 2 and not set(a) & set(s) and len(set(a)) == 2 and len(set(s)) == 2
        seen_active |= set(a); seen_skipped |= set(s)
    assert seen_active == set(range(6)) and seen_skipped == set(range(6))     # no fixed passive chunk


def test_slots_hold_product_passthrough_or_gelu_by_role():
    m = FlippedGateGELU(chunks=4, gates=2, skipped=1).train()
    x = torch.tensor([[1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.25, 4.0]])   # 4 chunks of 2
    torch.manual_seed(3)
    activated, skipped = m.draw(x.device)
    torch.manual_seed(3)
    y = m(x)
    pieces = x.chunk(4, dim=-1)
    product = F.gelu(pieces[activated[0]]) * F.gelu(pieces[activated[1]])
    for i in range(4):
        expected = product if i in activated else pieces[i] if i in skipped else F.gelu(pieces[i])
        assert torch.allclose(y[..., 2 * i:2 * i + 2], expected), (i, activated, skipped)
    assert sum(1 for i in range(4) if i not in activated and i not in skipped) == 1   # exactly one plain-GELU chunk


def test_zero_skipped_means_only_the_product_differs_from_gelu():
    m = FlippedGateGELU(chunks=4, gates=2, skipped=0).train()
    x = torch.randn(5, 8)
    torch.manual_seed(1)
    activated, skipped = m.draw(x.device)
    torch.manual_seed(1)
    y = m(x)
    assert skipped == []
    for i in range(4):
        if i not in activated:
            assert torch.equal(y[..., 2 * i:2 * i + 2], F.gelu(x[..., 2 * i:2 * i + 2]))


def test_gradient_reaches_every_chunk():
    m = FlippedGateGELU(chunks=4, gates=2, skipped=1).train()
    x = torch.randn(4, 8, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and float(x.grad.abs().sum()) > 0


@pytest.mark.parametrize("kwargs", [dict(chunks=3), dict(chunks=4, gates=0), dict(chunks=4, gates=3, skipped=2), dict(chunks=4, skipped=-1),
                                    dict(eval_mode="mean"), dict(mc_samples=0)])
def test_invalid_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        FlippedGateGELU(**kwargs)


def test_width_not_divisible_is_refused():
    with pytest.raises(ValueError):
        FlippedGateGELU(chunks=4, gates=2, skipped=1).train()(torch.randn(2, 10))
