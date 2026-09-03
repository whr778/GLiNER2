"""Public schema/annotation contract for Instance Formation record metadata."""

from __future__ import annotations

import pytest

from gliner2.inference.schema import Schema
from gliner2.processing.records import (
    FieldCardinality,
    compile_record_specs,
    normalize_record_metadata,
)
from gliner2.models.base import QueryLayout, QuerySpec


def _purchase_schema():
    s = Schema()
    (
        s.structure("purchase", mode="natural", anchor="buyer")
        .field("buyer", dtype="str", cardinality="required_one")
        .field("item", dtype="list", cardinality="zero_or_more", exclusive=True)
    )
    return s


def test_build_emits_record_metadata():
    built = _purchase_schema().build()
    meta = built["record_metadata"]["purchase"]
    assert meta["mode"] == "natural"
    assert meta["anchor"] == "buyer"
    assert meta["fields"]["buyer"]["cardinality"] == "required_one"
    assert meta["fields"]["item"]["exclusive"] is True


def test_to_dict_from_dict_round_trip_preserves_records():
    d = _purchase_schema().to_dict()
    assert d["structures"]["purchase"]["mode"] == "natural"
    assert d["structures"]["purchase"]["anchor"] == "buyer"
    fields = {f["name"]: f for f in d["structures"]["purchase"]["fields"]}
    assert fields["buyer"]["cardinality"] == "required_one"
    assert fields["item"]["exclusive"] is True

    rebuilt = Schema.from_dict(d).build()
    assert rebuilt["record_metadata"]["purchase"]["mode"] == "natural"
    assert rebuilt["record_metadata"]["purchase"]["anchor"] == "buyer"


def test_plain_structure_still_gets_record_metadata():
    """The plain form must NOT be legacy -- that was the defect, asserted as a contract.

    This test previously required `record_metadata` to be ABSENT for a structure that
    declares no mode. That absence is precisely what makes the boundary record head
    decode nothing while raising no error, and it is how `structure` came to read exactly
    0.0000 on every model this project measured. A caller writing the obvious
    `Schema().structure(n).field(...)` got silence, not a warning.

    The default matches what `_store_record_metadata` already picks when a caller sets
    mode and omits anchor: natural, anchored on the first declared field. Opting out is
    still possible and now explicit -- pass `mode="latent"` or declare no fields.
    """
    s = Schema()
    s.structure("plain").field("a").field("b")
    built = s.build()
    assert built["record_metadata"]["plain"] == {"mode": "natural", "anchor": "a"}


def test_structure_with_no_fields_emits_no_metadata():
    """No fields means no anchor is possible, so there is nothing to declare."""
    s = Schema()
    s.structure("empty")._auto_finish()
    assert "record_metadata" not in s.build()


def test_natural_requires_valid_anchor():
    s = Schema()
    s.structure("bad", mode="natural", anchor="missing").field("a")
    with pytest.raises(ValueError, match="anchor"):
        s.build()


def test_latent_must_not_declare_anchor():
    s = Schema()
    with pytest.raises(ValueError, match="must not set an anchor"):
        s.structure("x", mode="latent", anchor="a").field("a").build()


def test_normalize_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        normalize_record_metadata({"r": {"mode": "weird"}})


def test_compile_specs_respects_layout_order_and_dtype():
    # Two fields declared in one order; layout presents them shuffled.
    layout = QueryLayout(queries=(
        QuerySpec(query_id=0, task_index=0, task_type="json_structures",
                  task_name="purchase", role_index=0, role_name="item"),
        QuerySpec(query_id=1, task_index=0, task_type="json_structures",
                  task_name="purchase", role_index=1, role_name="buyer"),
    ))
    meta = {"purchase": {"mode": "natural", "anchor": "buyer"}}
    dtypes = {"purchase": {"buyer": "str", "item": "list"}}
    specs = compile_record_specs(
        query_layout=layout, record_metadata=meta, field_dtypes=dtypes
    )
    spec = specs[0]
    assert spec.mode == "natural"
    assert spec.anchor_query_id == 1
    buyer = spec.field_for_query(1)
    item = spec.field_for_query(0)
    assert buyer.is_anchor and buyer.cardinality == FieldCardinality.REQUIRED_ONE
    assert item.cardinality == FieldCardinality.ZERO_OR_MORE


def test_compile_specs_uses_dtype_for_implicit_scalar_cardinality():
    schema = Schema()
    (
        schema.structure("transaction", mode="natural", anchor="merchant")
        .field("merchant", dtype="str")
        .field("date", dtype="str")
        .field("tags", dtype="list")
    )
    built = schema.build()
    layout = QueryLayout(queries=(
        QuerySpec(
            query_id=0,
            task_index=0,
            task_type="json_structures",
            task_name="transaction",
            role_index=0,
            role_name="merchant",
        ),
        QuerySpec(
            query_id=1,
            task_index=0,
            task_type="json_structures",
            task_name="transaction",
            role_index=1,
            role_name="date",
        ),
        QuerySpec(
            query_id=2,
            task_index=0,
            task_type="json_structures",
            task_name="transaction",
            role_index=2,
            role_name="tags",
        ),
    ))

    spec = compile_record_specs(
        query_layout=layout,
        record_metadata=built["record_metadata"],
        field_dtypes={
            "transaction": {
                "merchant": "str",
                "date": "str",
                "tags": "list",
            }
        },
    )[0]

    assert spec.field_for_query(0).cardinality == FieldCardinality.REQUIRED_ONE
    assert spec.field_for_query(1).cardinality == FieldCardinality.OPTIONAL_ONE
    assert spec.field_for_query(2).cardinality == FieldCardinality.ZERO_OR_MORE


def test_events_exclusive_roles_emit_record_metadata():
    """`.events(..., exclusive_roles=[...])` reaches decode the same way a structure's
    `.field(..., exclusive=True)` does -- record_metadata keyed by event NAME, anchored
    on the trigger, so `compile_record_specs` picks it up ahead of the bare-defaults
    `_event_record_cfg` fallback it would otherwise synthesize.
    """
    s = Schema()
    s.events({
        "Attack": {"roles": ["attacker", "victim", "instrument"],
                   "exclusive_roles": ["victim"]},
        "Movement": ["agent", "destination"],   # no exclusive_roles: untouched path
    })
    built = s.build()
    assert built["record_metadata"]["Attack"] == {
        "mode": "natural", "anchor": "trigger",
        "fields": {"victim": {"exclusive": True}},
    }
    # An event that never asked for exclusivity gets no explicit entry at all --
    # `_event_record_cfg` keeps synthesizing bare defaults for it, unchanged.
    assert "Movement" not in built.get("record_metadata", {})


def test_events_exclusive_roles_rejects_unknown_role():
    with pytest.raises(ValueError, match="exclusive_roles"):
        Schema().events({
            "Attack": {"roles": ["attacker"], "exclusive_roles": ["not_a_role"]},
        })


def test_compile_specs_marks_event_role_exclusive():
    """End-to-end: the layout an events task actually produces (`trigger` at role_index
    0, per `_process_events`), through `compile_record_specs`, lands `exclusive` on the
    right field and leaves the others at their normal defaults.
    """
    s = Schema()
    s.events({"Attack": {"roles": ["attacker", "victim"], "exclusive_roles": ["victim"]}})
    built = s.build()

    layout = QueryLayout(queries=(
        QuerySpec(query_id=0, task_index=0, task_type="events",
                  task_name="Attack", role_index=0, role_name="trigger"),
        QuerySpec(query_id=1, task_index=0, task_type="events",
                  task_name="Attack", role_index=1, role_name="attacker"),
        QuerySpec(query_id=2, task_index=0, task_type="events",
                  task_name="Attack", role_index=2, role_name="victim"),
    ))
    spec = compile_record_specs(
        query_layout=layout, record_metadata=built["record_metadata"], event_records=True,
    )[0]
    assert spec.field_for_query(0).is_anchor
    assert spec.field_for_query(1).exclusive is False
    assert spec.field_for_query(2).exclusive is True


def test_compile_specs_excludes_deferred_relation_groups():
    layout = QueryLayout(queries=(
        QuerySpec(query_id=0, task_index=0, task_type="relations",
                  task_name="employment", role_index=0, role_name="head"),
        QuerySpec(query_id=1, task_index=0, task_type="relations",
                  task_name="employment", role_index=1, role_name="tail"),
    ))
    specs = compile_record_specs(
        query_layout=layout,
        record_metadata={"employment": {"mode": "natural", "anchor": "head"}},
    )
    assert specs == {}
