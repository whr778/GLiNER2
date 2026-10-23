"""relation_scorer_activation: GELU by default (bit-identical to before), fg-GEGLU on request, proven as built."""
import pytest
import torch
from torch import nn

from gliner2.configuration import validate_boundary_head
from gliner2.models.boundary.flipped_gate import FlippedGateGELU, build_activation
from gliner2.models.boundary.relations import SparseRelationScorer
from gliner2.training.trainer import describe_links


def test_config_defaults_and_validation():
    d = validate_boundary_head({})
    assert (d["relation_scorer_activation"], d["fg_geglu_chunks"], d["fg_geglu_gates"], d["fg_geglu_skipped"], d["fg_geglu_mc_samples"]) == ("gelu", 12, 3, 3, 8)
    assert validate_boundary_head({"relation_scorer_activation": "fg_geglu", "fg_geglu_chunks": 24})["fg_geglu_chunks"] == 24
    with pytest.raises(ValueError, match="relation_scorer_activation"):
        validate_boundary_head({"relation_scorer_activation": "relu"})


def test_build_activation():
    assert isinstance(build_activation("gelu"), nn.GELU)
    act = build_activation("fg_geglu", 24, 4, 4, 6)
    assert isinstance(act, FlippedGateGELU) and (act.chunks, act.gates, act.skipped, act.eval_mode, act.mc_samples) == (24, 4, 4, "mc", 6)
    with pytest.raises(ValueError):
        build_activation("swish")


def test_scorer_default_is_gelu_and_unchanged():
    torch.manual_seed(0)
    a = SparseRelationScorer(32, dropout=0.0)
    torch.manual_seed(0)
    b = SparseRelationScorer(32, dropout=0.0, activation=None)
    assert isinstance(a.mlp[1], nn.GELU) and isinstance(a.mlp[2], nn.Dropout)      # positions callers rely on
    x = torch.randn(5, 4 * 32 + 32 + 2)
    assert torch.equal(a.mlp(x), b.mlp(x))


def test_scorer_takes_fg_geglu_with_zero_extra_parameters():
    torch.manual_seed(0)
    gelu = SparseRelationScorer(48, dropout=0.0)
    torch.manual_seed(0)
    fg = SparseRelationScorer(48, dropout=0.0, activation=build_activation("fg_geglu", 12, 3, 3))
    assert isinstance(fg.mlp[1], FlippedGateGELU)
    assert sum(p.numel() for p in fg.parameters()) == sum(p.numel() for p in gelu.parameters())
    x = torch.randn(7, 4 * 48 + 48 + 2)
    fg.train(); assert fg.mlp(x).shape == (7, 1)
    fg.eval(); assert fg.mlp(x).shape == (7, 1)
    assert not torch.equal(fg.eval().mlp(x), gelu.eval().mlp(x))                      # the activation really differs


def test_describe_links_reports_the_activation_as_built():
    class Shell(nn.Module):
        def __init__(self, act):
            super().__init__()
            self.relation_scorer = SparseRelationScorer(16, activation=act)
    lines = describe_links(Shell(None))
    assert any(l.startswith("[relation scorer] activation GELU") and "width 16" in l for l in lines)
    lines = describe_links(Shell(build_activation("fg_geglu", 8, 2, 2)))
    assert any("FlippedGateGELU(chunks=8, gates=2, skipped=2" in l for l in lines)


def test_model_builds_the_scorer_with_the_configured_activation():
    from gliner2 import ExtractorConfig
    from gliner2.inference.engine import BoundaryExtractor
    from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD, build_tiny_encoder_config, build_tiny_tokenizer

    def build(**extra):
        tok = build_tiny_tokenizer()
        cfg = ExtractorConfig(model_name="tiny-bert-fixture", architecture="boundary", token_pooling="first",
                              boundary_head={**TINY_BOUNDARY_HEAD, "enable_relations": True, **extra})
        torch.manual_seed(5)
        return BoundaryExtractor(cfg, encoder_config=build_tiny_encoder_config(vocab_size=len(tok)), tokenizer=tok)

    assert isinstance(build().relation_scorer.mlp[1], nn.GELU)
    m = build(relation_scorer_activation="fg_geglu", fg_geglu_chunks=4, fg_geglu_gates=2, fg_geglu_skipped=1, fg_geglu_mc_samples=3)
    act = m.relation_scorer.mlp[1]
    assert isinstance(act, FlippedGateGELU) and (act.chunks, act.gates, act.skipped, act.mc_samples) == (4, 2, 1, 3)
