"""Smoke tests for the event-extraction pipeline.

Covers the Phase 1-4 surface area end-to-end without external corpora:
schema round-trip, training-data round-trip, model forward+backward
including events, inference via extract_events, and the new
event_trigger / event_argument keys returned by compute_metrics.

Marked ``slow`` because the in-process trainer step instantiates the
mmBERT encoder (~550 MB download on first run).
"""

from __future__ import annotations

import pytest

from gliner2.inference.schema import Schema
from gliner2.training.data import Event, EventArgument, InputExample
from gliner2.training.metrics import (
    _gold_event_argument_set,
    _gold_event_trigger_set,
    _pred_event_argument_set,
    _pred_event_trigger_set,
    _schema_from_gold,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_events_round_trip():
    s = Schema()
    s.events({
        "Attack": ["Attacker", "Victim", "Place"],
        "Meet": {
            "roles": ["Entity", "Place"],
            "description": "Two parties meet",
            "role_descriptions": {"Entity": "Participant"},
        },
    })
    built = s.build()
    assert list(built["events"]) == ["Attack", "Meet"]
    assert built["events"]["Attack"] == ["Attacker", "Victim", "Place"]

    out = s.to_dict()
    s2 = Schema.from_dict(out)
    assert s2._event_metadata["Meet"]["description"] == "Two parties meet"
    assert ("Meet", "Entity") in s2._event_role_descriptions


def test_schema_events_rejects_empty_inputs():
    with pytest.raises(ValueError):
        Schema().events({})
    with pytest.raises(ValueError):
        Schema().events({"Bad": []})


# ---------------------------------------------------------------------------
# InputExample
# ---------------------------------------------------------------------------

def test_event_round_trip_validate_and_to_dict():
    ex = InputExample(
        text="John fired Bob in Paris.",
        events=[Event(event_type="Attack", triggers=["fired"], arguments=[
            EventArgument(role="Attacker", entity="John"),
            EventArgument(role="Victim", entity="Bob"),
            EventArgument(role="Place", entity="Paris"),
        ])],
    )
    assert ex.validate() == []
    rt = InputExample.from_dict(ex.to_dict())
    assert rt.to_dict() == ex.to_dict()
    assert rt.events[0].arguments[0].role == "Attacker"


def test_sanitize_drops_unresolvable_events_and_arguments():
    ex = InputExample(
        text="John fired Bob.",
        events=[
            Event(event_type="Attack", triggers=["exploded"], arguments=[]),
            Event(event_type="Attack", triggers=["fired"], arguments=[
                EventArgument(role="Attacker", entity="John"),
                EventArgument(role="Place", entity="Mars"),
            ]),
        ],
    )
    warns, ok = ex.sanitize()
    assert ok
    # The first event (bad trigger) is dropped entirely.
    assert len(ex.events) == 1
    # The bad Place argument is dropped from the surviving event.
    assert [(a.role, a.entity) for a in ex.events[0].arguments] == [("Attacker", "John")]
    assert any("not in text" in w for w in warns)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_event_metric_helpers_match_expected_tuples():
    gold = {
        "events": [
            {"event_type": "Attack", "triggers": ["fired"],
             "arguments": [
                 {"role": "Attacker", "entity": "John"},
                 {"role": "Victim", "entity": "Bob"},
             ]},
            {"event_type": "Meet", "triggers": ["met"],
             "arguments": [{"role": "Entity", "entity": "Alice"}]},
        ]
    }
    pred = {
        "event_extraction": {
            "Attack": [
                {"triggers": ["fired"],
                 "arguments": [
                     {"role": "Attacker", "entity": "John"},
                     {"role": "Victim", "entity": "Bob"},
                     {"role": "Victim", "entity": "Carla"},
                 ]},
            ],
            "Meet": [],
        }
    }
    assert _gold_event_trigger_set(gold) == {("Attack", "fired"), ("Meet", "met")}
    assert _pred_event_trigger_set(pred) == {("Attack", "fired")}
    assert ("Attack", "Victim", "Carla", ("fired",)) in _pred_event_argument_set(pred)


def test_schema_from_gold_picks_up_events():
    gold = {
        "events": [
            {"event_type": "Attack", "triggers": ["fired"],
             "arguments": [{"role": "Attacker", "entity": "John"}]},
        ]
    }
    schema = _schema_from_gold(gold)
    assert schema == {"events": {"Attack": ["Attacker"]}}


# ---------------------------------------------------------------------------
# Multiple trigger spans per mention
# ---------------------------------------------------------------------------

def test_event_with_two_triggers_validates_and_round_trips():
    ex = InputExample(
        text="Rebels shot and killed the guard.",
        events=[Event(event_type="Attack", triggers=["shot", "killed"], arguments=[
            EventArgument(role="Attacker", entity="Rebels"),
            EventArgument(role="Victim", entity="the guard"),
        ])],
    )
    assert ex.validate() == []
    rt = InputExample.from_dict(ex.to_dict())
    assert rt.events[0].triggers == ["shot", "killed"]


def test_event_rejects_non_list_triggers():
    with pytest.raises(TypeError):
        Event(event_type="Attack", triggers="shot")  # bare str, not a list


def test_sanitize_drops_only_the_missing_trigger_not_the_whole_event():
    ex = InputExample(
        text="Rebels shot the guard.",
        events=[Event(event_type="Attack", triggers=["shot", "exploded"], arguments=[
            EventArgument(role="Attacker", entity="Rebels"),
        ])],
    )
    warns, ok = ex.sanitize()
    assert ok
    assert len(ex.events) == 1
    assert ex.events[0].triggers == ["shot"]
    assert any("exploded" in w and "dropping trigger" in w for w in warns)


def test_sanitize_drops_whole_event_when_no_triggers_survive():
    ex = InputExample(
        text="Rebels shot the guard.",
        events=[Event(event_type="Attack", triggers=["exploded", "vanished"], arguments=[
            EventArgument(role="Attacker", entity="Rebels"),
        ])],
    )
    warns, ok = ex.sanitize()
    assert ok is False  # no valid tasks remain
    assert ex.events == []


def test_gold_trigger_set_flattens_multiple_triggers():
    gold = {"events": [
        {"event_type": "Attack", "triggers": ["shot", "killed"],
         "arguments": [{"role": "Attacker", "entity": "Rebels"}]},
    ]}
    assert _gold_event_trigger_set(gold) == {("Attack", "shot"), ("Attack", "killed")}


def test_gold_argument_set_uses_canonical_sorted_trigger_key():
    gold = {"events": [
        {"event_type": "Attack", "triggers": ["killed", "shot"],  # unsorted input order
         "arguments": [{"role": "Attacker", "entity": "Rebels"}]},
    ]}
    assert _gold_event_argument_set(gold) == {
        ("Attack", "Attacker", "Rebels", ("killed", "shot")),
    }


def test_pred_argument_set_matches_gold_when_both_triggers_predicted():
    """A two-trigger mention only strict-matches on the argument metric when
    BOTH trigger spans are predicted -- the whole trigger set is the key."""
    gold = {"event_type": "Attack", "triggers": ["shot", "killed"],
            "arguments": [{"role": "Attacker", "entity": "Rebels"}]}
    pred_full = {"event_extraction": {"Attack": [
        {"triggers": ["shot", "killed"],
         "arguments": [{"role": "Attacker", "entity": "Rebels"}]}]}}
    pred_partial = {"event_extraction": {"Attack": [
        {"triggers": ["shot"],
         "arguments": [{"role": "Attacker", "entity": "Rebels"}]}]}}
    gold_set = _gold_event_argument_set({"events": [gold]})
    assert gold_set & _pred_event_argument_set(pred_full) == gold_set
    assert gold_set & _pred_event_argument_set(pred_partial) == set()


def test_chunking_keeps_event_only_when_all_triggers_survive():
    from gliner2.training.chunking import _filter_events

    events = [{"event_type": "Attack", "triggers": ["shot", "killed"],
               "arguments": [{"role": "Attacker", "entity": "Rebels"}]}]

    # Both trigger words present in the chunk -> mention kept.
    both = _filter_events(events, "Rebels shot and killed the guard.")
    assert len(both) == 1
    assert both[0]["triggers"] == ["shot", "killed"]

    # Only one trigger word present -> mention dropped entirely.
    partial = _filter_events(events, "Rebels shot the guard.")
    assert partial == []


# ---------------------------------------------------------------------------
# End-to-end (slow): trainer + inference + metrics
# ---------------------------------------------------------------------------

class _IndexableDataset:
    """Minimal indexable adapter accepted by compute_metrics."""

    def __init__(self, records):
        self._records = list(records)

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        rec = self._records[idx]
        return rec["input"], rec["output"]


@pytest.mark.slow
def test_event_pipeline_end_to_end():
    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    text = "John fired Bob in Paris yesterday."
    gold = {
        "input": text,
        "output": {
            "events": [
                {"event_type": "Attack", "triggers": ["fired"], "arguments": [
                    {"role": "Attacker", "entity": "John"},
                    {"role": "Victim", "entity": "Bob"},
                    {"role": "Place", "entity": "Paris"},
                ]}
            ]
        },
    }

    model = GLiNER2.from_encoder(
        "jhu-clsp/mmBERT-small", max_width=8, max_len=256, map_location="cpu",
    )
    cfg = TrainingConfig(
        output_dir="/tmp/event_pipeline_test_out",
        num_epochs=1, batch_size=1, fp16=False, bf16=False,
        eval_strategy="no", save_total_limit=1, num_workers=0,
        pin_memory=False, validate_data=False, report_to_wandb=False,
        logging_steps=1, max_steps=2,
    )
    ex = InputExample.from_dict(gold)
    trainer = GLiNER2Trainer(model, cfg)
    trainer.train(train_data=[ex])

    # Inference: returns the event_extraction block with the right shape.
    result = model.extract_events(
        text,
        {"Attack": ["Attacker", "Victim", "Place", "Time"]},
        threshold=0.01,
    )
    assert "event_extraction" in result
    assert "Attack" in result["event_extraction"]

    # Format-regression guard: the scorer must parse the engine's real event
    # output (mirrors the relation tuple-format bug). The model emitted an Attack
    # event above, so the trigger scorer must read at least one trigger from it.
    assert _pred_event_trigger_set(result), "scorer parsed no triggers from real model output"

    # compute_metrics: event_trigger + event_argument keys populated.
    metrics = compute_metrics(
        model,
        _IndexableDataset([gold]),
        batch_size=1,
        threshold=0.01,
    )
    assert "eval_event_trigger_strict_micro_f1" in metrics
    assert "eval_event_argument_strict_micro_f1" in metrics
    assert "eval_event_trigger_relaxed_micro_f1" in metrics
    assert "eval_event_argument_relaxed_micro_f1" in metrics
    assert metrics["eval_event_trigger_strict_support"] == 1
    assert metrics["eval_event_argument_strict_support"] == 3


@pytest.mark.slow
def test_event_pipeline_multiple_triggers_end_to_end():
    """A mention with two trigger spans trains and decodes without error,
    and the engine's decoded shape is a 'triggers' list, not a scalar."""
    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    text = "Rebels shot and killed the guard yesterday."
    gold = {
        "input": text,
        "output": {
            "events": [
                {"event_type": "Attack", "triggers": ["shot", "killed"], "arguments": [
                    {"role": "Attacker", "entity": "Rebels"},
                    {"role": "Victim", "entity": "the guard"},
                ]}
            ]
        },
    }

    model = GLiNER2.from_encoder(
        "jhu-clsp/mmBERT-small", max_width=8, max_len=256, map_location="cpu",
    )
    cfg = TrainingConfig(
        output_dir="/tmp/event_pipeline_multi_trigger_test_out",
        num_epochs=1, batch_size=1, fp16=False, bf16=False,
        eval_strategy="no", save_total_limit=1, num_workers=0,
        pin_memory=False, validate_data=False, report_to_wandb=False,
        logging_steps=1, max_steps=2,
    )
    ex = InputExample.from_dict(gold)
    assert ex.events[0].triggers == ["shot", "killed"]
    trainer = GLiNER2Trainer(model, cfg)
    trainer.train(train_data=[ex])

    result = model.extract_events(
        text,
        {"Attack": ["Attacker", "Victim"]},
        threshold=0.01,
    )
    assert "event_extraction" in result
    attacks = result["event_extraction"].get("Attack", [])
    if attacks:
        # Decode shape is always a list, regardless of how many spans
        # actually cleared threshold after 2 training steps.
        assert isinstance(attacks[0]["triggers"], list)


GOLD_WITH_DESCRIPTIONS = {
    "entities": {"e_0": ["product installation"], "e_1": []},
    "entity_descriptions": {"e_0": "the process of installing software", "e_1": "a registry key"},
}


def test_schema_from_gold_reaches_the_processor_with_descriptions(tiny_span_model):
    """Corpora like pile_ner_def / nuner_full name types e_0/e_1 and put the
    meaning in a parallel map. Dropping it asked the model to find "e_0" with an
    empty description -- base-v1 scores 0.5381 strict entity F1 on 100
    pile_ner_def records with the descriptions and 0.0174 without.

    Asserted through the processor, not on the dict shape: an earlier fix put
    the descriptions under schema["entities"] as their values, which type-checks
    and reads as correct but changes nothing the model sees -- those values are
    label targets, and the prompt is built from schema["entity_descriptions"]."""
    processor = tiny_span_model.processor
    processor.change_mode(is_training=False)

    prompt = " ".join(
        tok for part in processor._infer_from_json(_schema_from_gold(GOLD_WITH_DESCRIPTIONS))["schemas"]
        for tok in part
    )

    assert "the process of installing software" in prompt
    assert "a registry key" in prompt


def test_schema_from_gold_without_descriptions_omits_the_key():
    schema = _schema_from_gold({"entities": {"person": ["Ada"]}})
    assert schema["entities"] == {"person": ""}
    assert "entity_descriptions" not in schema
