"""The dense record loss must not average the LIST-field term over PADDING.

`compute_dense_group_loss` reduced its list-field BCE with `.mean(-1)` across the full
padded pool width, not across the real candidate spans, so the list-field loss was
scaled by exactly ``n_real / n_padded``. On the fixture below (10 real spans in a
64-wide pool) the sparse path's list term 0.361597 became 0.056500 = 0.361597 * 10/64,
which reproduced the whole gap in `field_loss` (1.016330 -> 0.863781) to 7 decimal
places. The scalar branch was unaffected because `log_softmax`/`logsumexp` handles
padding gracefully.

WHY IT MATTERS: the list branch is where record LIST fields train, which for events is
every argument role, and the dilution scales with `pool_size` (default 384).

FIXED LOCALLY 2026-09-30, reported upstream as fastino-ai/GLiNER2#180 (upstream
2e8a44e3). The reduction now means over `group.field_membership`, the [n_fields, pool]
mask of real spans that already existed, exactly as `instance_mask` is applied to the
object loss. Only `candidate_pool: "shared"` reaches this path; the default
"per_query" routes to `compute_group_loss`, which was never affected.
"""

from __future__ import annotations

import pytest
import torch

from gliner2.models.boundary import records as R
from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.training import ExtractorCollator

from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer
from tests.models.boundary.test_record_head_pipeline import _build_tiny_records_model


def _batch():
    processor = SchemaTransformer(
        tokenizer=build_tiny_tokenizer(),
        sampling_config=SamplingConfig(
            remove_json_structure_prob=0.0,
            shuffle_json_fields=False,
            remove_json_field_prob=0.0,
            synthetic_entity_label_prob=0.0,
        ),
    )
    return ExtractorCollator(
        processor, is_training=True, architecture="boundary", max_gold_per_query=16
    )([(
        "Alice bought apples",
        {
            "json_structures": [{"purchase": {"buyer": "Alice", "item": "apples"}}],
            "record_metadata": {"purchase": {"mode": "natural", "anchor": "buyer"}},
        },
    )])


def _capture_one_group():
    """Run the real per-query path and keep one group plus the args that built it.

    `forward_group` and `forward_group_dense` take IDENTICAL arguments, so the same
    candidates can build both representations -- which is what makes this a
    comparison of the two LOSSES rather than of two different candidate sets.
    """
    model = _build_tiny_records_model(candidate_pool="per_query")
    model.train()
    captured: list[dict] = []
    decoder_cls = type(model.record_decoder)
    real_fg, real_loss = decoder_cls.forward_group, R.compute_group_loss

    def fg_spy(self, spec, query_states, candidates, sample_index):
        group = real_fg(self, spec, query_states, candidates, sample_index)
        captured.append({
            "args": (spec, query_states, candidates, sample_index),
            "decoder": self, "group": group,
        })
        return group

    def loss_spy(group, recs):
        result = real_loss(group, recs)
        for entry in captured:
            if entry["group"] is group:
                entry["recs"], entry["sparse"] = recs, result
        return result

    decoder_cls.forward_group, R.compute_group_loss = fg_spy, loss_spy
    try:
        torch.manual_seed(1234)
        with torch.no_grad():
            model(_batch())
    finally:
        decoder_cls.forward_group, R.compute_group_loss = real_fg, real_loss

    scored = [c for c in captured if "recs" in c]
    assert scored, "no record group reached the loss -- the comparison is unmeasured"
    return scored[0]


def test_the_dense_pool_is_mostly_padding():
    """Pins the premise: the pool is padded far beyond the real spans."""
    entry = _capture_one_group()
    spec, qs, cands, si = entry["args"]
    with torch.no_grad():
        dense = entry["decoder"].forward_group_dense(spec, qs, cands, si)
    real = int((dense.pool_spans.abs().sum(-1) > 0).sum())
    padded = dense.pool_spans.shape[0]
    assert real < padded, "fixture no longer pads; this test proves nothing"
    # The mask that would fix the reduction exists and counts exactly the real spans.
    assert int(dense.field_membership.sum()) == real * dense.field_membership.shape[0]


def test_dense_and_sparse_record_losses_agree_on_identical_input():
    entry = _capture_one_group()
    spec, qs, cands, si = entry["args"]
    with torch.no_grad():
        dense_group = entry["decoder"].forward_group_dense(spec, qs, cands, si)
        dense = R.compute_dense_group_loss(dense_group, entry["recs"])
    sparse = entry["sparse"]

    # The object term already agrees: both return zero in natural mode.
    assert float(dense["object_loss"]) == pytest.approx(float(sparse["object_loss"]))
    # The field term agrees only when padding columns are excluded (#180).
    assert float(dense["field_loss"]) == pytest.approx(
        float(sparse["field_loss"]), rel=1e-4
    ), "dense field_loss is diluted by padding columns"
