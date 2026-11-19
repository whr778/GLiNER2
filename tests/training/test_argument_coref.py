"""Argument coreference mentions become gold fillers of the same role, and only that cluster."""
import random
from dataclasses import replace

import torch

from gliner2.processing.argument_coref import (
    added_argument_mentions, attach_coref_copies, expand_argument_coreferents,
)
from gliner2.training.eval_metrics import score_predictions
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
    returned = []
    real = compute_group_loss

    def spy(group, records, *args, **kwargs):
        out = real(group, records, *args, **kwargs)
        returned.append({
            k: (float(v.detach()) if torch.is_tensor(v) and v.numel() == 1 else v)
            for k, v in out.items()
        })
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
        return out

    import gliner2.models.boundary.records as records_mod
    model.boundary_settings = replace(model.boundary_settings, record_link_column_weight=0.3)
    records_mod.compute_group_loss = spy
    try:
        out = model(batch)
    finally:
        records_mod.compute_group_loss = real
    assert torch.isfinite(out.loss)
    assert float(out.losses["record_object_loss"].detach()) == 0.0
    field_loss = out.losses["record_field_loss"].detach()
    assert float(field_loss) > 0.0
    assert torch.isfinite(field_loss)
    assert any(item.get("column_count", 0) > 0 and item.get("column_loss", 0.0) > 0.0 for item in returned), returned
    assert any("he" in texts for texts in seen), seen
    assert all(texts.count("he") <= 1 for texts in seen)


def _collate(text, schema):
    from gliner2.processor import SamplingConfig, SchemaTransformer
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    proc = SchemaTransformer(
        tokenizer=build_tiny_tokenizer(),
        sampling_config=SamplingConfig(remove_events_prob=0.0, shuffle_event_roles=False),
    )
    coll = ExtractorCollator(
        proc, is_training=True, architecture="boundary", max_gold_per_query=64,
        on_capacity_exceeded="raise", error_policy="raise", event_records=True,
    )
    random.seed(0)
    return coll([(text, schema)])


def _event_surfaces(batch):
    words = batch.text_tokens[0]
    found = []
    for rec in batch.targets.records[0]:
        spec = batch.record_specs[0][rec.task_index]
        if spec.task_type != "events":
            continue
        for field in rec.fields:
            name = next(item.name for item in spec.fields if item.query_id == field.query_id)
            for value in field.values:
                for start, end in value:
                    found.append((name, " ".join(words[int(start):int(end)])))
    return found


def test_expanded_counts_survive_the_collator_and_the_pronoun_is_gold():
    record = _record()
    schema = expand_argument_coreferents(record["output"], record["coref_clusters"], TEXT)
    batch = _collate(TEXT, schema)
    assert batch.original_schemas[0]["_argument_coref_added"] == 1
    assert batch.original_schemas[0]["_argument_coref_skipped"] == 0
    surfaces = _event_surfaces(batch)
    assert ("Person", "he") in surfaces
    assert ("Person", "paul") in surfaces
    assert sum(1 for name, surface in surfaces if surface == "he") == 1
    # Coreferent mentions are grouped as occurrence alternatives of the single filler.
    rec = batch.targets.records[0][0]
    spec = batch.record_specs[0][rec.task_index]
    person_qid = next(f.query_id for f in spec.fields if f.name == "Person")
    ft = rec.field_for_query(person_qid)
    assert len(ft.values) == 1
    assert len(ft.values[0]) == 2


def test_string_event_argument_groups_multiple_occurrences_as_one_value():
    text = "hackers attacked the bank yesterday . later the mayor visited the bank ."
    schema = {"events": [{
        "event_type": "Attack",
        "triggers": ["attacked"],
        "arguments": [{"role": "Target", "entity": "bank"}],
    }]}
    batch = _collate(text, schema)
    rec = batch.targets.records[0][0]
    spec = batch.record_specs[0][rec.task_index]
    target_qid = next(f.query_id for f in spec.fields if f.name == "Target")
    ft = rec.field_for_query(target_qid)
    assert ft.values == (((3, 4), (11, 12)),)


def test_a_gold_span_inside_a_longer_word_is_not_widened():
    text = "bombs in TexasA string of deadly bombs"
    start = text.index("Texas")
    schema = {"events": [{
        "event_type": "Attack",
        "triggers": ["bombs"],
        "arguments": [{"role": "Place", "entity": {"text": "Texas", "start": start, "end": start + len("Texas")}}],
    }]}
    batch = _collate(text, schema)
    surfaces = _event_surfaces(batch)
    assert ("trigger", "bombs") in surfaces
    assert ("Place", "texasa") not in surfaces
    assert ("Place", "texas") not in surfaces


def _leave(arguments):
    return {"event_extraction": {"Leave": [{"triggers": ["left"], "arguments": arguments}]}}


def test_argument_coref_recall_counts_an_emitted_mention_and_is_not_a_head():
    """The published argument F1 scores the annotated filler. The companion counts the
    mention expansion would have added, and stays out of head_min."""
    import inspect

    from gliner2.training import eval_metrics

    keys = added_argument_mentions(_record()["output"], _record()["coref_clusters"], TEXT)
    assert keys == [("Leave", "Person", "he")]
    gold = dict(_record()["output"], entities={"person": ["Paul"]}, _argument_coref_opportunities=keys)
    missed = _leave([{"role": "Person", "entity": "Paul"}])
    hit = _leave([{"role": "Person", "entity": "Paul"}, {"role": "Person", "entity": "he"}])
    wrong = _leave([{"role": "Person", "entity": "Paul"}, {"role": "Place", "entity": "he"}])
    for pred in (missed, hit, wrong):
        pred["entities"] = {"person": ["Paul"]}
    m0 = score_predictions([gold], [missed], report=False)
    m1 = score_predictions([gold], [hit], report=False)
    m2 = score_predictions([gold], [wrong], report=False)
    assert m0["eval_argument_coref_opportunities"] == 1
    assert m0["eval_argument_coref_emitted"] == 0 and m0["eval_argument_coref_recall"] == 0.0
    assert m1["eval_argument_coref_emitted"] == 1 and m1["eval_argument_coref_recall"] == 1.0
    assert m2["eval_argument_coref_emitted"] == 0
    assert m0["eval_event_argument_strict_micro_f1"] == 1.0
    assert m1["eval_event_argument_strict_micro_f1"] < 1.0
    assert m1["eval_overall_strict_head_min_f1"] == min(
        m1["eval_entity_strict_micro_f1"], m1["eval_event_argument_strict_micro_f1"])
    block = inspect.getsource(eval_metrics.score_predictions).split("primitive_heads = (", 1)[1].split("\n    )", 1)[0]
    assert "event_argument" in block and "argument_coref" not in block


def test_sliding_window_and_wired_gate_honour_expand_argument_coref_false(tmp_path):
    """Regression for ACE 2005 collapse: sliding_window=True previously hardcoded
    expand_argument_coref=is_train, ignoring TrainingConfig(expand_argument_coref=False)."""
    import pytest

    from gliner2.processing import coref
    from gliner2.training.config import TrainingConfig
    from gliner2.training.trainer import GLiNER2Trainer
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    model = _build_tiny_records_model()
    for sw in (False, True):
        cfg = TrainingConfig(
            output_dir=str(tmp_path / f"sw_{sw}"),
            sliding_window=sw,
            max_len=128,
            window_stride=96,
            expand_argument_coref=False,
            validate_data=False,
        )
        trainer = GLiNER2Trainer(model, cfg)
        trainer.processor.tokenizer = build_tiny_tokenizer()
        ds_train = trainer._prepare_data([_record()], is_train=True)
        assert ds_train.expand_argument_coref is False
        _, schema = ds_train[0]
        assert schema["events"][0]["arguments"] == [{"role": "Person", "entity": "Paul"}]
        assert "_argument_coref_added" not in schema
        # MentionLink still receives gold coref_clusters via coref.attach/take.
        _, clusters = coref.take([ds_train[0]])
        assert clusters[0] == _record()["coref_clusters"]

        ds_train.expand_argument_coref = True
        with pytest.raises(SystemExit, match="expand_argument_coref"):
            trainer._wired(ds_train, is_train=True)


def test_compute_metrics_and_splice_never_expand_coref_into_eval_gold():
    """Standalone ExtractorDataset(records) defaults to expand_argument_coref=True;
    compute_metrics / score_model_schema / _splice_dataset_predictions must still
    grade against unexpanded local gold while preserving _argument_coref_opportunities."""
    from types import SimpleNamespace

    from gliner2.training.eval_metrics import (
        _splice_dataset_predictions,
        compute_metrics,
        score_model_schema,
    )

    class _ExactLocalStub:
        config = SimpleNamespace(
            default_schema={"events": {"Leave": ["Person"]}},
            label_map=None,
            label_style=None,
        )

        def batch_extract(self, texts, schemas, **kwargs):
            return [_leave([{"role": "Person", "entity": "Paul"}]) for _ in texts]

    bare_ds = ExtractorDataset([_record()], shuffle=False, validate=False)
    assert bare_ds.expand_argument_coref is True

    m = compute_metrics(_ExactLocalStub(), bare_ds, threshold=0.5, report=False)
    assert bare_ds.expand_argument_coref is True
    assert m["eval_event_argument_strict_support"] == 1
    assert m["eval_event_argument_strict_micro_f1"] == 1.0
    assert m["eval_argument_coref_opportunities"] == 1
    assert m["eval_argument_coref_emitted"] == 0

    ms = score_model_schema(_ExactLocalStub(), bare_ds, threshold=0.5, report=False)
    assert ms["eval_event_argument_strict_support"] == 1
    assert ms["eval_event_argument_strict_micro_f1"] == 1.0

    golds, _, kept = _splice_dataset_predictions(
        _ExactLocalStub(), bare_ds, {"event": 0.3}, 0.5
    )
    assert kept == [0]
    assert golds[0]["events"][0]["arguments"] == [{"role": "Person", "entity": "Paul"}]


def test_coref_clusters_ignore_disables_training_and_decode_coref(tmp_path):
    """training.coref_clusters='ignore' drops coref-only records/chunks, strips
    _coref_clusters, zeroes entity_coref_link_weight, and clears inherited
    entity_coref_threshold / copy / boost on both live settings and saved config."""
    import pytest

    from gliner2.processing import coref
    from gliner2.training.config import TrainingConfig
    from gliner2.training.trainer import GLiNER2Trainer
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    with pytest.raises(ValueError, match="coref_clusters"):
        TrainingConfig(coref_clusters="bogus", bf16=False, fp16=False)

    coref_only = {
        "input": TEXT,
        "output": {},
        "coref_clusters": _record()["coref_clusters"],
    }
    mix = [_record(), coref_only]

    for sw in (False, True):
        model = _build_tiny_records_model()
        model.config.boundary_head = {
            **dict(getattr(model.config, "boundary_head", None) or {}),
            "entity_coref_link": True,
            "entity_coref_link_weight": 0.3,
            "entity_coref_threshold": 0.5,
            "record_argument_coref_copy": "canonical",
            "record_argument_coref_boost": 0.25,
        }
        model.enable_mention_link()
        cfg = TrainingConfig(
            output_dir=str(tmp_path / f"ig_{sw}"),
            bf16=False,
            fp16=False,
            sliding_window=sw,
            max_len=128,
            window_stride=96,
            coref_clusters="ignore",
            expand_argument_coref=True,
            validate_data=False,
        )
        trainer = GLiNER2Trainer(model, cfg)
        trainer.processor.tokenizer = build_tiny_tokenizer()

        assert model.boundary_settings.entity_coref_link_weight == 0.0
        assert model.boundary_settings.entity_coref_threshold is None
        assert model.boundary_settings.record_argument_coref_copy == "none"
        assert model.boundary_settings.record_argument_coref_boost == 0.0
        assert model.config.boundary_head["entity_coref_threshold"] is None
        assert not any(p.requires_grad for p in model.mention_link.parameters())

        ds_train = trainer._prepare_data([dict(r) for r in mix], is_train=True)
        assert len(ds_train) == 1  # coref-only record dropped
        assert ds_train.use_coref_clusters is False
        assert ds_train.expand_argument_coref is False
        _, schema = ds_train[0]
        assert schema["events"][0]["arguments"] == [{"role": "Person", "entity": "Paul"}]
        _, clusters = coref.take([ds_train[0]])
        assert clusters == [None]

        ds_train.use_coref_clusters = True
        with pytest.raises(SystemExit, match="use_coref_clusters"):
            trainer._wired(ds_train, is_train=True)


