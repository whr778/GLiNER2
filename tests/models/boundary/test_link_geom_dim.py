"""record_link_geom_dim: the junction's geometry channel width is a YAML setting; structural (reshapes 4 tensors);
a non-default width leaves every other weight of the head bit-identical at init (the RNG parity the A/B needs)."""
import sys
from pathlib import Path

import pytest
import torch

from gliner2 import ExtractorConfig
from gliner2.configuration import validate_boundary_head
from gliner2.inference.engine import BoundaryExtractor
from tests.fixtures.tiny_boundary_checkpoint import (TINY_BOUNDARY_HEAD, build_tiny_boundary_model, build_tiny_encoder_config,
                                                     build_tiny_tokenizer)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "train"))
import train as T  # noqa: E402

JUNCTION = dict(record_link_mode="junction", record_link_column_weight=0.3)


def build(geom_dim=None, **extra):
    tok = build_tiny_tokenizer()
    head = {**TINY_BOUNDARY_HEAD, **JUNCTION, **({"record_link_geom_dim": geom_dim} if geom_dim else {}), **extra}
    cfg = ExtractorConfig(model_name="tiny-bert-fixture", architecture="boundary", boundary_head=head, token_pooling="first")
    torch.manual_seed(17)
    return BoundaryExtractor(cfg, encoder_config=build_tiny_encoder_config(vocab_size=len(tok)), tokenizer=tok)


def test_setting_parses_and_validates():
    assert validate_boundary_head({})["record_link_geom_dim"] == 16
    assert validate_boundary_head({"record_link_geom_dim": 64})["record_link_geom_dim"] == 64
    with pytest.raises(ValueError, match="record_link_geom_dim"):
        validate_boundary_head({"record_link_geom_dim": 0})


def test_width_reshapes_only_the_geometry_tensors_and_leaves_the_rest_of_the_init_identical():
    a, b = build(), build(64)
    la, lb = a.record_decoder.link, b.record_decoder.link
    assert la.geom.weight.shape == (16, 2) and lb.geom.weight.shape == (64, 2)
    assert la.geom_q.weight.shape[0] == 16 and lb.geom_q.weight.shape[0] == 64
    assert torch.count_nonzero(lb.geom_q.weight) == 0 and torch.count_nonzero(lb.v.weight) == 0
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert set(pa) == set(pb)
    reshaped = {n for n in pa if pa[n].shape != pb[n].shape}
    assert reshaped == {f"record_decoder.link.{n}" for n in ("geom.weight", "geom.bias", "geom_q.weight", "geom_q.bias")}
    same = [n for n in pa if n not in reshaped and torch.equal(pa[n], pb[n])]
    assert len(same) == len(pa) - 4, [n for n in pa if n not in reshaped and not torch.equal(pa[n], pb[n])][:5]


def test_width_is_structural_on_a_built_model():
    model = build_tiny_boundary_model()
    with pytest.raises(SystemExit, match="structural"):
        T._apply_boundary_head_overrides(model, {**JUNCTION, "record_link_geom_dim": 64})
