"""Argument coreference mentions become gold fillers of the same role, and only that cluster."""
import random

import torch

from gliner2.processing.argument_coref import attach_coref_copies, expand_argument_coreferents
from gliner2.training.chunking import _chunk_clusters, _filter_events
from gliner2.training.trainer import ExtractorCollator, ExtractorDataset
from tests.models.boundary.test_record_head_pipeline import _build_tiny_records_model

TEXT = "Paul said he left . Mary said he stayed ."


def _span(text, start):
    return {"text": text, "start": start, "end": start + len(text)}


def _record():
    paul = TEXT.index("Paul")
    he1 = TEXT.index("he")
    mary = TEXT.index("Mary")
    he2 = TEXT.index("he", he1 + 1)
    return {
        "input": TEXT,
        "output": {"events": [{
            "event_type": "Leave",
            "triggers": ["left"],
            "arguments": [{"role": "Person", "entity": "Paul"}],
        }]},
        "coref_clusters": [
            [_span("Paul", paul), _span("he", he1)],
            [_span("Mary", mary), _span("he", he2)],
        ],
    }


def test_expansion_adds_the_clusters_other_mentions_and_not_a_second_cluster():
    record = _record()
    expanded = expand_argument_coreferents(record["output"], record["coref_clusters"], TEXT)
    entities = [a["entity"] for a in expanded["events"][0]["arguments"]]
    assert entities[0] == "Paul"
    assert entities[1] == _span("he", TEXT.index("he"))
    assert expanded["_argument_coref_added"] == 1
    assert expanded["_argument_coref_skipped"] == 0
    assert record["output"]["events"][0]["arguments"] == [{"role": "Person", "entity": "Paul"}]


def test_a_string_in_two_clusters_is_not_expanded():
    record = _record()
    record["coref_clusters"][1].append(_span("Paul", TEXT.index("Paul")))
    expanded = expand_argument_coreferents(record["output"], record["coref_clusters"], TEXT)
    assert expanded["events"][0]["arguments"] == [{"role": "Person", "entity": "Paul"}]
    assert expanded["_argument_coref_skipped"] == 1
    assert "_argument_coref_added" not in record["output"]


def test_getitem_expands_a_copy_and_a_second_read_does_not_stack():
    dataset = ExtractorDataset([_record()], shuffle=False, validate=False)
    _, schema = dataset[0]
    _, again = dataset[0]
    assert schema["events"][0]["arguments"][1]["entity"]["text"] == "he"
    assert again["events"][0]["arguments"] == schema["events"][0]["arguments"]
    assert dataset.data[0]["output"]["events"][0]["arguments"] == [{"role": "Person", "entity": "Paul"}]


def test_coref_only_copies_attach_to_the_event_record_and_stay_in_the_mix():
    event = {"input": TEXT, "output": _record()["output"]}
    coref = {"input": TEXT, "output": {}, "coref_clusters": _record()["coref_clusters"]}
    stats = attach_coref_copies([event, coref])
    assert stats == {"attached": 1, "ambiguous": 0, "coref_only": 1}
    assert event["coref_clusters"] == coref["coref_clusters"]
    assert coref["output"] == {}


def test_chunk_offsets_keep_a_pronoun_inside_the_window_and_drop_one_outside():
    text = "Paul left yesterday . he was gone"
    paul, he = text.index("Paul"), text.index("he")
    events = [{"event_type": "Leave", "triggers": ["left"], "arguments": [{"role": "Person", "entity": "Paul"}]}]
    clusters = [[_span("Paul", paul), _span("he", he)]]
    inside_end = he + len("he")
    inside = expand_argument_coreferents(
        {"events": _filter_events(events, text[:inside_end], 0, inside_end)},
        _chunk_clusters(clusters, 0, inside_end),
        text[:inside_end],
    )
    assert inside["events"][0]["arguments"][1]["entity"] == _span("he", he)
    outside = expand_argument_coreferents(
        {"events": _filter_events(events, text[:he], 0, he)},
        _chunk_clusters(clusters, 0, he),
        text[:he],
    )
    assert outside["events"][0]["arguments"] == [{"role": "Person", "entity": "Paul"}]


def test_the_pronoun_span_is_a_positive_column_and_the_other_pronoun_is_not():
    from gliner2.models.boundary.records import _field_target_cols, compute_group_loss
    from gliner2.processor import SamplingConfig, SchemaTransformer
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    model = _build_tiny_records_model()
    model.train()
    proc = SchemaTransformer(
        tokenizer=build_tiny_tokenizer(),
        sampling_config=SamplingConfig(remove_events_prob=0.0, shuffle_event_roles=False),
    )
    coll = ExtractorCollator(
        proc, is_training=True, architecture="boundary", max_gold_per_query=64,
        on_capacity_exceeded="raise", error_policy="raise", event_records=True,
    )
    record = _record()
    schema = expand_argument_coreferents(record["output"], record["coref_clusters"], TEXT)
    random.seed(0)
    batch = coll([(TEXT, schema)])
    toks = batch.text_tokens[0]
    surfaces = []
    for rec in batch.targets.records[0]:
        for field in rec.fields:
            surfaces.append([" ".join(toks[s:e]) for value in field.values for (s, e) in value])
    flat = [span for field in surfaces for span in field]
    assert "he" in flat
    assert flat.count("he") == 1

    seen = []
    real = compute_group_loss

    def spy(group, records, *args, **kwargs):
        for rec in records:
            for f_idx, fspec in enumerate(group.field_specs):
                cols, _ = _field_target_cols(fspec, rec, {
                    (int(span[0]), int(span[1])): i
                    for i, span in enumerate(group.field_spans[f_idx].tolist())
                })
                texts = []
                spans = group.field_spans[f_idx].tolist()
                for col in cols:
                    if col == 0:
                        continue
                    start, end = spans[col - 1]
                    texts.append(" ".join(toks[start:end]))
                seen.append(texts)
        return real(group, records, *args, **kwargs)

    import gliner2.models.boundary.records as records_mod
    records_mod.compute_group_loss = spy
    try:
        out = model(batch)
    finally:
        records_mod.compute_group_loss = real
    assert torch.isfinite(out.loss)
    assert any("he" in texts for texts in seen), seen
    assert all(texts.count("he") <= 1 for texts in seen)
