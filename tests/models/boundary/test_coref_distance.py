"""coref_link_distance: TriggerLink's and MentionLink's distance bias as a zero-at-init MLP (scalar = historical)."""
import sys
from pathlib import Path

import pytest
import torch

from gliner2.configuration import validate_boundary_head
from gliner2.models.boundary.distance_bias import build_distance_bias
from gliner2.training.trainer import describe_links
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model
from tests.models.boundary.test_link_geom_dim import build

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "train"))
import train as T  # noqa: E402

LINKS = dict(record_coref_link=True, record_coref_link_weight=0.3, entity_coref_link=True, entity_coref_link_weight=0.3)


def test_setting_parses_and_validates():
    d = validate_boundary_head({})
    assert (d["coref_link_distance"], d["coref_link_distance_dim"]) == ("scalar", 16)
    assert validate_boundary_head({"coref_link_distance": "mlp", "coref_link_distance_dim": 64})["coref_link_distance_dim"] == 64
    with pytest.raises(ValueError, match="coref_link_distance"):
        validate_boundary_head({"coref_link_distance": "none"})
    with pytest.raises(ValueError, match="coref_link_distance_dim"):
        validate_boundary_head({"coref_link_distance_dim": 0})


def test_both_kinds_are_zero_at_init_and_keep_the_historical_names():
    torch.manual_seed(0)
    scalar, mlp = build_distance_bias("scalar", 16), build_distance_bias("mlp", 64)
    x = torch.log1p(torch.arange(0, 50, dtype=torch.float32)).unsqueeze(-1)
    assert torch.count_nonzero(scalar(x)) == 0 and torch.count_nonzero(mlp(x)) == 0
    assert set(scalar.state_dict()) == {"weight", "bias"}
    assert set(mlp.state_dict()) == {"0.weight", "0.bias", "2.weight", "2.bias"} and mlp[0].weight.shape == (64, 1)
    with torch.no_grad():
        mlp[2].weight.normal_()
    assert torch.count_nonzero(mlp(x)) > 0


def test_mlp_links_leave_every_other_weight_identical_at_init():
    a, b = build(**LINKS), build(coref_link_distance="mlp", coref_link_distance_dim=64, **LINKS)
    assert set(a.record_decoder.coref.dist.state_dict()) == {"weight", "bias"}
    assert set(b.record_decoder.coref.dist.state_dict()) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    assert set(b.mention_link.dist.state_dict()) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    shared = [n for n in pa if ".dist." not in n]
    assert shared and all(n in pb and torch.equal(pa[n], pb[n]) for n in shared), [n for n in shared if n not in pb or not torch.equal(pa[n], pb[n])][:5]
    assert any("coref" in l and "distance mlp" in l and "width 64" in l for l in describe_links(b)), describe_links(b)


def test_kind_is_structural_on_a_built_model():
    with pytest.raises(SystemExit, match="structural"):
        T._apply_boundary_head_overrides(build_tiny_boundary_model(), {**LINKS, "coref_link_distance": "mlp"})
