"""record_link_geom_activation: a GELU inside the junction's geometry channel (none = the historical affine term)."""
import pytest
import torch

import gliner2.models.boundary.records as R
from gliner2.configuration import validate_boundary_head
from gliner2.training.trainer import describe_links
from tests.models.boundary.test_link_geom_dim import build


def test_setting_parses_and_validates():
    assert validate_boundary_head({})["record_link_geom_activation"] == "none"
    assert validate_boundary_head({"record_link_geom_activation": "gelu"})["record_link_geom_activation"] == "gelu"
    with pytest.raises(ValueError, match="record_link_geom_activation"):
        validate_boundary_head({"record_link_geom_activation": "relu"})


def test_activation_adds_no_tensors_and_changes_the_geometry_term_only_once_it_is_nonzero():
    torch.manual_seed(0)
    lin, act = R.LinkJunction(32, 128, 64), R.LinkJunction(32, 128, 64, geom_activation="gelu")
    act.load_state_dict(lin.state_dict())   # same tensors: the activation is forward-only
    inst, cand, q = torch.randn(3, 32), torch.randn(5, 32), torch.randn(32)
    inst_spans, cand_spans = torch.tensor([[1, 2], [10, 11], [20, 21]]), torch.tensor([[3, 4], [8, 9], [15, 16], [25, 26], [30, 31]])
    zero_l, zero_a = lin(inst, [q], [cand], inst_spans, [cand_spans])[0], act(inst, [q], [cand], inst_spans, [cand_spans])[0]
    assert torch.count_nonzero(zero_l) == 0 and torch.count_nonzero(zero_a) == 0   # geom_q is zero: 0 at init either way
    with torch.no_grad():
        lin.geom_q.weight.normal_(); lin.geom_q.bias.normal_()
    act.load_state_dict(lin.state_dict())   # SAME weights: the activation is the only difference
    out_l, out_a = lin(inst, [q], [cand], inst_spans, [cand_spans])[0], act(inst, [q], [cand], inst_spans, [cand_spans])[0]
    assert not torch.allclose(out_l, out_a)


def test_built_from_config_and_described():
    model = build(64, record_link_geom_activation="gelu")
    assert model.record_decoder.link.geom_activation == "gelu"
    lines = describe_links(model)
    assert any("junction" in l and "width 64" in l and "activation gelu" in l for l in lines), lines
