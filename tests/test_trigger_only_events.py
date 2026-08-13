"""Trigger-only event corpora (MAVEN: 168 event types, zero arguments) must be
both queryable and scoreable.

Two independent defects made them silently unmeasurable, and because each one
returned "nothing" rather than raising, a 15-epoch A/B ran to completion reporting
0.0000 at every threshold while checkpoint selection fell back to eval_loss:

* ``_process_events`` skipped any event type with an empty role list, so no
  ``[V]`` trigger query was emitted at all and the decode had nothing to assemble.
* ``_schema_from_gold`` dropped role-less event types, emptying the schema for
  every record so ``compute_metrics`` returned ``{}``.
"""

from gliner2.processor import SchemaTransformer
from gliner2.training.eval_metrics import _schema_from_gold

TRIGGER_ONLY_GOLD = {
    "events": [
        {"event_type": "Process_start", "triggers": ["began"], "arguments": []},
        {"event_type": "Killing", "triggers": ["killed"], "arguments": []},
    ]
}


def _events_queries(schema):
    """Run the inference-path event branch and return its emitted query types.

    Built with ``__new__`` so no tokenizer download is needed: the event branch
    touches only ``is_training`` and the class-level marker tokens.
    """
    st = SchemaTransformer.__new__(SchemaTransformer)
    st.is_training = False
    schemas, labels, types = [], [], []
    st._process_events(schema, schemas, labels, types, None)
    return schemas, types


def test_roleless_event_type_still_emits_a_trigger_query():
    _, types = _events_queries({"events": {"Process_start": [], "Killing": []}})
    assert types == ["events", "events"], "role-less event types emitted no query"


def test_roled_event_type_unaffected():
    schemas, types = _events_queries({"events": {"Attack": ["Attacker", "Target"]}})
    assert types == ["events"] and len(schemas) == 1


def test_schema_from_gold_keeps_roleless_types():
    schema = _schema_from_gold(TRIGGER_ONLY_GOLD)
    assert schema.get("events") == {"Process_start": [], "Killing": []}


def test_schema_from_gold_still_collects_roles_when_present():
    gold = {"events": [{
        "event_type": "Attack",
        "triggers": ["struck"],
        "arguments": [{"role": "Target", "entity": "the base"}],
    }]}
    assert _schema_from_gold(gold)["events"] == {"Attack": ["Target"]}


def test_trigger_only_gold_is_scoreable_end_to_end():
    """The gold -> schema -> query chain must survive with zero arguments anywhere."""
    schema = _schema_from_gold(TRIGGER_ONLY_GOLD)
    _, types = _events_queries(schema)
    assert types, "trigger-only gold produced a schema that emits no queries"
