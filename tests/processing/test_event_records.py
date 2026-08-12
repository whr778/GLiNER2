"""Event groups compiled as records (JOINT_IE_SCALING Tier 2), behind `event_records`.

Both decode paths emit ONE instance per event type, because events never reach the record
head: `compile_record_specs` skips any group whose task type is not in
`RECORD_TASK_TYPES == ("json_structures",)`. Measured cost in gold instances that are
unreachable under that cap -- CASIE 78.8%, WikiEvents 62.5%, MAVEN 38.3%, and RAMS 0.0%
(RAMS is 100% single-event documents, so it cannot see this at all).

The corpora need no change. `_process_events` already appends one label row per event
mention, so `structure[1]` is a list of instance rows and `structure[0]` is the count --
the instance dimension is in the supervision today. Only the spec was missing.

`event_records` is OFF by default, per the repo convention that new behaviour does not
move existing numbers until asked for: it changes what the record head is supervised on.
"""

from __future__ import annotations

from gliner2.models.base import QueryLayout, QuerySpec
from gliner2.processing.records import FieldCardinality, compile_record_specs


def _events_layout(task_type: str = "events") -> QueryLayout:
    return QueryLayout(queries=(
        QuerySpec(query_id=0, task_index=0, task_type=task_type, task_name="attack",
                  role_index=0, role_name="trigger"),
        QuerySpec(query_id=1, task_index=0, task_type=task_type, task_name="attack",
                  role_index=1, role_name="victim"),
        QuerySpec(query_id=2, task_index=0, task_type=task_type, task_name="attack",
                  role_index=2, role_name="place"),
    ))


def test_events_compile_no_record_spec_by_default():
    """The default must reproduce the older numbers exactly."""
    specs = compile_record_specs(query_layout=_events_layout(), record_metadata=None)

    assert specs == {}


def test_events_compile_a_record_spec_when_opted_in():
    specs = compile_record_specs(
        query_layout=_events_layout(), record_metadata=None, event_records=True
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.mode == "natural"
    assert spec.task_type == "events"


def test_the_trigger_is_the_anchor_without_any_stored_metadata():
    """`field_names = ["trigger"] + roles`, so role_index 0 is the anchor by construction.

    This is what makes the change a compile-time synthesis rather than a corpus rebuild
    across RAMS, CASIE, WikiEvents, MAVEN, DuEE and CMNEE.
    """
    spec = compile_record_specs(
        query_layout=_events_layout(), record_metadata=None, event_records=True
    )[0]

    assert spec.anchor_query_id == 0
    trigger = spec.field_for_query(0)
    assert trigger.is_anchor
    assert trigger.cardinality == FieldCardinality.REQUIRED_ONE


def test_roles_keep_multi_filler_cardinality():
    """A role legitimately has several fillers; only the anchor is required-one."""
    spec = compile_record_specs(
        query_layout=_events_layout(), record_metadata=None, event_records=True
    )[0]

    for query_id in (1, 2):
        field = spec.field_for_query(query_id)
        assert not field.is_anchor
        assert field.cardinality == FieldCardinality.ZERO_OR_MORE


def test_the_flag_does_not_leak_into_other_task_types():
    """Relations stay deferred: `event_records` opts in events, nothing else."""
    layout = QueryLayout(queries=(
        QuerySpec(query_id=0, task_index=0, task_type="relations", task_name="employment",
                  role_index=0, role_name="head"),
        QuerySpec(query_id=1, task_index=0, task_type="relations", task_name="employment",
                  role_index=1, role_name="tail"),
    ))

    specs = compile_record_specs(
        query_layout=layout,
        record_metadata={"employment": {"mode": "natural", "anchor": "head"}},
        event_records=True,
    )

    assert specs == {}


def test_explicit_metadata_still_wins_over_the_synthesized_config():
    """A corpus that DOES annotate its events keeps its own anchor."""
    spec = compile_record_specs(
        query_layout=_events_layout(),
        record_metadata={"attack": {"mode": "natural", "anchor": "victim"}},
        event_records=True,
    )[0]

    assert spec.anchor_query_id == 1
    assert spec.field_for_query(1).is_anchor
    assert not spec.field_for_query(0).is_anchor
