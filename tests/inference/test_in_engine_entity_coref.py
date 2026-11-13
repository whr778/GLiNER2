"""Tests for in-engine MentionLink clustering, argument coref attachment, and Lever 2A/2B overrides."""
import pytest
import torch

from gliner2.configuration import validate_boundary_head
from gliner2.training.eval_metrics import apply_boundary_overrides
from tests.models.boundary.test_record_head_pipeline import _build_tiny_records_model


def test_validate_boundary_head_accepts_coref_decode_settings_and_rejects_invalid():
    cfg = validate_boundary_head({
        "entity_coref_link": True,
        "entity_coref_threshold": 0.5,
        "record_argument_coref_match": "relaxed",
        "record_argument_coref_copy": "gated",
        "record_argument_coref_min_score": 0.25,
        "record_argument_coref_boost": 0.15,
    })
    assert cfg["entity_coref_threshold"] == 0.5
    assert cfg["record_argument_coref_match"] == "relaxed"
    assert cfg["record_argument_coref_copy"] == "gated"
    assert cfg["record_argument_coref_min_score"] == 0.25
    assert cfg["record_argument_coref_boost"] == 0.15

    with pytest.raises(ValueError, match="record_argument_coref_match"):
        validate_boundary_head({"record_argument_coref_match": "invalid"})
    with pytest.raises(ValueError, match="record_argument_coref_copy"):
        validate_boundary_head({"record_argument_coref_copy": "invalid"})
    with pytest.raises(ValueError, match="record_argument_coref_min_score"):
        validate_boundary_head({"record_argument_coref_min_score": -0.1})


def test_in_engine_entity_coref_attaches_clusters_without_reencoding():
    model = _build_tiny_records_model()
    model.enable_mention_link()
    with torch.no_grad():
        model.mention_link.w.weight.zero_()
        model.mention_link.w.bias.fill_(5.0)

    text = "Alice met Bob. She left."
    schema = {"entities": ["Person"], "events": {"Leave": ["Person"]}}

    # Off by default: coref_clusters is not emitted
    res_off = model.batch_extract([text], [schema], threshold=0.0, include_spans=True)[0]
    assert "coref_clusters" not in res_off

    # Enabled via boundary_head override: coref_clusters is emitted in the same forward pass
    apply_boundary_overrides(model, {
        "entity_coref_threshold": 0.1,
        "record_argument_coref_match": "relaxed",
        "record_argument_coref_copy": "all",
    })
    res_on = model.batch_extract([text], [schema], threshold=0.0, include_spans=True)[0]
    assert "coref_clusters" in res_on
    assert isinstance(res_on["coref_clusters"], list)

    # Also works via batch_extract_long with include_spans=False (the compute_metrics path)
    res_long = model.batch_extract_long(
        [text], [schema], threshold=0.0, include_spans=False, chunk_size=64, chunk_overlap=0
    )[0]
    assert "coref_clusters" in res_long
    assert isinstance(res_long["coref_clusters"], list)
    for c in res_long["coref_clusters"]:
        assert isinstance(c, list)
        for m in c:
            assert isinstance(m, dict) and "start" in m and "end" in m
