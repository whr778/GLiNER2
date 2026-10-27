"""A surface that is not its own word drops that mention. It is not widened onto the enclosing word."""

from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.processing.boundary_preprocessing import suppressed_query_counts
from gliner2.training.trainer import ExtractorCollator

_SAMPLING = SamplingConfig(
    synthetic_entity_label_prob=0.0,
    remove_entities_prob=0.0,
    remove_entity_prob=0.0,
    remove_events_prob=0.0,
)


def _collator(tokenizer, **kwargs):
    proc = SchemaTransformer(tokenizer=tokenizer, sampling_config=_SAMPLING)
    return ExtractorCollator(
        proc, is_training=True, architecture="boundary", error_policy="raise",
        on_capacity_exceeded="raise", **kwargs,
    )


def _spans(batch):
    words = list(batch.text_tokens[0])
    roles = [q.role_name for q in batch.query_layouts[0].queries]
    pairs = batch.targets.mention_pairs[0]
    mask = batch.targets.mention_mask[0]
    found = []
    for qi, role in enumerate(roles):
        for g in range(pairs.shape[1]):
            if int(mask[qi, g]) == 0:
                continue
            s, e = int(pairs[qi, g, 0]), int(pairs[qi, g, 1])
            found.append((role, words[s:e]))
    return roles, found


def test_substring_entity_is_dropped_and_the_sibling_span_is_kept(tiny_tokenizer):
    coll = _collator(tiny_tokenizer, ignore_missing_entities=True, suppress_orphaned_queries=True)
    batch = coll([("The Iraqi army advanced.", {"entities": {"GPE": ["Iraq"], "Organization": ["army"]}})])
    roles, spans = _spans(batch)
    assert roles == ["Organization"]
    assert spans == [("Organization", ["army"])]
    assert int(batch.query_marker_mask[0].sum()) == len(roles)
    assert int(batch.targets.mention_mask.shape[1]) == len(roles)
    assert suppressed_query_counts()["GPE"] >= 1


def test_a_type_whose_only_surface_misses_leaves_the_menu(tiny_tokenizer):
    coll = _collator(tiny_tokenizer, ignore_missing_entities=True, suppress_orphaned_queries=True)
    batch = coll([("The Iraqi army advanced.", {"entities": {"GPE": ["Iraq"]}})])
    roles, spans = _spans(batch)
    assert roles == []
    assert spans == []
    assert int(batch.query_marker_mask.sum()) == 0


def test_raise_still_aborts_when_the_flag_is_off(tiny_tokenizer):
    coll = _collator(tiny_tokenizer)
    try:
        coll([("The Iraqi army advanced.", {"entities": {"GPE": ["Iraq"]}})])
    except ValueError as exc:
        assert "was not found" in str(exc)
    else:
        raise AssertionError("expected the unaligned surface to raise")


def test_offset_inside_a_word_is_not_widened(tiny_tokenizer):
    text = "TexasA is not Texas today."
    schema = {"events": [{
        "event_type": "Life.Die",
        "triggers": [{"text": "is", "start": text.index(" is ") + 1, "end": text.index(" is ") + 3}],
        "arguments": [{"role": "Place", "entity": {"text": "Texas", "start": 0, "end": 5}}],
    }]}
    coll = _collator(tiny_tokenizer, ignore_missing_entities=True, suppress_orphaned_queries=True, event_records=True)
    roles, spans = _spans(coll([(text, schema)]))
    assert ("Place", ["texasa"]) not in spans
    assert ("trigger", ["is"]) in spans
    assert all(span != ["texasa"] for _, span in spans)
