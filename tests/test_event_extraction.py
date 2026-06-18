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
        events=[Event(event_type="Attack", trigger="fired", arguments=[
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
            Event(event_type="Attack", trigger="exploded", arguments=[]),
            Event(event_type="Attack", trigger="fired", arguments=[
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
            {"event_type": "Attack", "trigger": "fired",
             "arguments": [
                 {"role": "Attacker", "entity": "John"},
                 {"role": "Victim", "entity": "Bob"},
             ]},
            {"event_type": "Meet", "trigger": "met",
             "arguments": [{"role": "Entity", "entity": "Alice"}]},
        ]
    }
    pred = {
        "event_extraction": {
            "Attack": [
                {"trigger": "fired",
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
    assert ("Attack", "Victim", "Carla", "fired") in _pred_event_argument_set(pred)


def test_schema_from_gold_picks_up_events():
    gold = {
        "events": [
            {"event_type": "Attack", "trigger": "fired",
             "arguments": [{"role": "Attacker", "entity": "John"}]},
        ]
    }
    schema = _schema_from_gold(gold)
    assert schema == {"events": {"Attack": ["Attacker"]}}


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
                {"event_type": "Attack", "trigger": "fired", "arguments": [
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
