"""The GIST veto end to end: cache -> injected rival -> candidate column -> dropped negative.

Every stage of this path fails silently. A wrong text hash, a surface key that does not
match, a swapped axis or an un-injected rival all produce the same observable outcome --
the loss is unchanged -- so the tests here assert on the counters and on the mined mask,
not merely that a step runs.
"""

from __future__ import annotations

import json
from dataclasses import replace

import torch

from gliner2.models.boundary.guide import (
    densify_guide,
    guide_counts,
    reset_guide_counts,
)
from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.training import ExtractorCollator, ExtractorDataset
from gliner2.training.guide_scores import GuideScores, text_key
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

TEXT = "john works at apple ."


def _cache(tmp_path, *, own, rival, scores, text=TEXT):
    path = tmp_path / "guide.jsonl"
    path.write_text(json.dumps({
        "i": 0, "sha1": text_key(text), "own": own, "rival": rival, "s": scores,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return GuideScores.load(path)


def _collator(model):
    """A processor with sampling disabled: synthetic type renaming would rename the very
    queries the veto matches by name, turning a real failure into an intermittent one."""
    processor = SchemaTransformer(
        tokenizer=model.processor.tokenizer,
        sampling_config=SamplingConfig(
            shuffle_entities=False, synthetic_entity_label_prob=0.0,
            remove_entity_prob=0.0, remove_entities_prob=0.0,
        ),
    )
    return ExtractorCollator(processor, is_training=True, architecture="boundary")


def _step(model, guide, schema):
    """One training step. An injected rival has no gold, so its negatives are only mined
    at all under ``keep_all_when_absent`` -- without it an untrained tiny model decides by
    noise which candidates it proposes, and the veto has nothing to act on."""
    model.boundary_head.settings = replace(
        model.boundary_head.settings, hard_negative_keep_all_when_absent=True
    )
    reset_guide_counts()
    model.set_guide_scores(guide)
    model.train()
    batch = _collator(model)([(TEXT, schema)])
    model(batch)
    return guide_counts()


def test_injection_adds_the_cached_rival_as_an_absent_entity_query(tmp_path):
    guide = _cache(tmp_path, own=["person"], rival={"company": "a company"},
                   scores={"john": {"person": 0.3, "company": 0.9}})
    dataset = ExtractorDataset(
        [{"input": TEXT, "output": {"entities": {"person": ["john"]}}}],
        shuffle=False, guide_scores=guide,
    )
    _, schema = dataset[0]

    assert schema["entities"] == {"person": ["john"], "company": []}
    assert schema["entity_descriptions"]["company"] == "a company"


def test_injection_leaves_the_stored_record_unmutated(tmp_path):
    guide = _cache(tmp_path, own=["person"], rival={"company": "a company"},
                   scores={"john": {"person": 0.3, "company": 0.9}})
    dataset = ExtractorDataset(
        [{"input": TEXT, "output": {"entities": {"person": ["john"]}}}],
        shuffle=False, guide_scores=guide,
    )
    dataset[0]

    assert dataset.data[0]["output"] == {"entities": {"person": ["john"]}}


def test_a_rival_the_guide_prefers_removes_the_mined_negative(tmp_path):
    model = build_tiny_boundary_model()
    guide = _cache(tmp_path, own=["person"], rival={"company": "a company"},
                   scores={"john": {"person": 0.3, "company": 0.9}})
    counts = _step(model, guide, {"entities": {"person": ["john"], "company": []}})

    assert counts["samples_cached"] == 1
    assert counts["spans_matched"] >= 1
    assert counts["vetoed"] >= 1


def test_an_own_record_query_is_never_vetoed_however_high_the_guide_scores_it(tmp_path):
    """Within a record gold is authoritative -- `company` is definitively wrong on `john`,
    and that makes it a correct hard negative no guide may delete."""
    model = build_tiny_boundary_model()
    guide = _cache(tmp_path, own=["person", "company"], rival={},
                   scores={"john": {"person": 0.3, "company": 0.9}})
    counts = _step(model, guide, {"entities": {"person": ["john"], "company": []}})

    assert counts["spans_matched"] >= 1
    assert counts["vetoed"] == 0


def test_a_rival_the_guide_ranks_below_the_gold_owner_is_kept(tmp_path):
    model = build_tiny_boundary_model()
    guide = _cache(tmp_path, own=["person"], rival={"company": "a company"},
                   scores={"john": {"person": 0.9, "company": 0.3}})
    counts = _step(model, guide, {"entities": {"person": ["john"], "company": []}})

    assert counts["spans_matched"] >= 1
    assert counts["vetoed"] == 0


def test_a_text_the_cache_does_not_hold_fills_nothing(tmp_path):
    model = build_tiny_boundary_model()
    guide = _cache(tmp_path, own=["person"], rival={"company": "a company"},
                   scores={"john": {"person": 0.3, "company": 0.9}},
                   text="a different document entirely")
    counts = _step(model, guide, {"entities": {"person": ["john"], "company": []}})

    assert counts["samples"] == 1
    assert counts["samples_cached"] == 0
    assert counts["vetoed"] == 0


def test_densify_puts_the_score_on_the_matching_candidate_in_both_axis_orders():
    spans = torch.tensor([[[2, 4]]])                 # [B=1, S=1, 2]
    values = torch.tensor([[[0.0, 0.7]]])            # [B,S,Q=2]
    reference = torch.tensor([[0.4]])                # [B,S]
    mask = torch.ones(1, 1, dtype=torch.bool)
    sparse = (spans, values, reference, mask)

    per_query = torch.tensor([[[[0, 1], [2, 4]], [[2, 4], [5, 6]]]])   # [B,Q=2,C=2,2]
    dense, ref = densify_guide(sparse, per_query, 1, 2)
    assert torch.equal(dense, torch.tensor([[[0.0, 0.0], [0.7, 0.0]]]))
    # The reference follows the SPAN: (2,4) sits at column 1 for query 0 and column 0 for
    # query 1, which is exactly why it cannot be read off a fixed column.
    assert torch.equal(ref, torch.tensor([[[0.0, 0.4], [0.4, 0.0]]]))

    pooled = torch.tensor([[[0, 1], [2, 4]]])                          # [B,C=2,2]
    dense, ref = densify_guide(sparse, pooled, 2, 1)
    assert torch.equal(dense, torch.tensor([[[0.0, 0.0], [0.0, 0.7]]]))  # [B,C,Q]
    assert torch.equal(ref, torch.tensor([[[0.0, 0.0], [0.4, 0.4]]]))
