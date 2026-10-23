"""fg-GEGLU: eval is GELU, training gates random chunks except the passive one, gated slots hold the product."""
import pytest
import torch
import torch.nn.functional as F

from gliner2.models.boundary.flipped_gate import FlippedGateGELU


def test_eval_is_exactly_gelu():
    m = FlippedGateGELU(chunks=4, gates=2).eval()
    x = torch.randn(3, 5, 16)
    assert torch.equal(m(x), F.gelu(x))


def test_training_keeps_width_and_passive_chunk():
    m = FlippedGateGELU(chunks=4, gates=3, passive=0).train()
    x = torch.randn(2, 7, 12)
    y = m(x)
    assert y.shape == x.shape
    assert torch.equal(y[..., :3], x[..., :3])          # passive chunk 0 (width 12 / 4) untouched


def test_passive_never_drawn_and_every_other_chunk_is():
    m = FlippedGateGELU(chunks=6, gates=2, passive=4).train()
    drawn = [i for _ in range(500) for i in m.draw_gates(torch.device("cpu")).tolist()]
    assert 4 not in drawn
    assert set(drawn) == {0, 1, 2, 3, 5}
    assert all(len(set(m.draw_gates(torch.device("cpu")).tolist())) == 2 for _ in range(50))   # without replacement


def test_gated_slots_hold_the_product_of_gelu_chunks():
    m = FlippedGateGELU(chunks=4, gates=2, passive=0).train()
    x = torch.tensor([[1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.25, 4.0]])   # 4 chunks of 2
    torch.manual_seed(3)
    gated = m.draw_gates(x.device).tolist()
    torch.manual_seed(3)
    y = m(x)
    pieces = x.chunk(4, dim=-1)
    product = F.gelu(pieces[gated[0]]) * F.gelu(pieces[gated[1]])
    for i in range(4):
        expected = product if i in gated else pieces[i]
        assert torch.allclose(y[..., 2 * i:2 * i + 2], expected), (i, gated)


def test_gradient_reaches_gated_and_passive_chunks():
    m = FlippedGateGELU(chunks=4, gates=2).train()
    x = torch.randn(4, 8, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0


@pytest.mark.parametrize("kwargs", [dict(chunks=3), dict(chunks=4, gates=4), dict(chunks=4, gates=0), dict(chunks=4, passive=4)])
def test_invalid_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        FlippedGateGELU(**kwargs)


def test_width_not_divisible_is_refused():
    with pytest.raises(ValueError):
        FlippedGateGELU(chunks=4, gates=2).train()(torch.randn(2, 10))
