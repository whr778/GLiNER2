"""A record carrying NER + Structures + Classifications + Relations + Events.

Real training data puts all five task types in every record, so the boundary layout
must count queries consistently for every combination. If it under-counts (as it did
for events before "[V]" was added to `_EXTRACTIVE_MARKERS`), `query_states`/
`query_mask` still carry one entry per marker and gold injection dies comparing
mismatched shapes.

Asserts the invariant directly -- one layout query per emitted query marker -- rather
than a forward pass, so it stays fast and pinpoints the cause on failure.
"""

from __future__ import annotations

import random

import pytest
import torch

from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.training import ExtractorCollator

TEXT = ("Alice from Apple acquired Beats for $400 million and ordered 3 laptops "
        "while Bob from Sony sold Walkman units in Tokyo last winter")

ALL_FIVE = {
    "entities": {"Person": ["Alice", "Bob"], "Organization": ["Apple", "Sony"],
                 "Money": ["$400 million"]},
    "json_structures": [{"order": {"buyer": "Alice", "item": "laptops"}}],
    "record_metadata": {"order": {"mode": "natural", "anchor": "buyer"}},
    "classifications": [{"task": "topic", "labels": ["business", "sports"],
                         "true_label": ["business"]}],
    "relations": [{"works_for": {"head": "Alice", "tail": "Apple"}}],
    "events": [{"event_type": "business.acquire", "triggers": ["acquired"],
                "arguments": [{"role": "Acquirer", "entity": "Apple"},
                              {"role": "Price", "entity": "$400 million"}]}],
}


@pytest.mark.parametrize("seed", range(12))
def test_layout_query_count_matches_emitted_markers(tiny_tokenizer, seed):
    """One layout query per query marker, for whatever task mix the sampler draws."""
    torch.manual_seed(seed)
    random.seed(seed)
    proc = SchemaTransformer(tokenizer=tiny_tokenizer, sampling_config=SamplingConfig())
    collator = ExtractorCollator(proc, is_training=True, architecture="boundary")
    batch = collator([(TEXT, ALL_FIVE)])

    queries = len(batch.query_layouts[0].queries)
    markers = int(batch.query_marker_mask.sum())
    assert queries == markers, (
        f"seed={seed} task_types={batch.task_types[0]}: layout has {queries} queries "
        f"but {markers} query markers were emitted -- some group's marker is missing "
        f"from _EXTRACTIVE_MARKERS, so its queries occupy slots with no targets"
    )
