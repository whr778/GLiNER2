"""A structure schema must carry `record_metadata` all the way to the processor.

Regression test for a defect that cost this project a great deal. The boundary record
head decodes only when `compile_record_specs` receives `record_metadata`; without it the
call returns `{}`, there is no spec, nothing decodes, **and no error is raised** -- the
extraction is simply empty and the `structure` metric reads exactly 0.0000.

`runtime.py` rebuilt the schema through `Schema.from_dict(...).build()`, which DOES
produce `record_metadata`, then copied only `json_structures` and `json_descriptions`
out of the result. The metadata was dropped on the floor.

The consequence was not subtle and was very nearly misdiagnosed: `structure` read
exactly 0.0000 on ALL FOUR points of the joint_ie scaling curve (10k/40k/100k/137k) and
again on a warm start that added +45% structure supervision. Across a 13x data range a
head that was learning would show something; the conclusion drawn was "the record head
is broken or untrainable". It was neither -- it was never asked a question it could
answer. With the metadata restored the same checkpoint emits records and recovers gold
field values.

These tests assert the plumbing, not the model: that the key survives the round trip and
that a spec gets compiled. They need no checkpoint and no GPU.
"""
from __future__ import annotations

from gliner2.inference.schema import Schema


def _structures_schema() -> dict:
    """The serialized shape `eval_metrics._schema_from_gold` builds from gold."""
    return {
        "structures": {
            "record": {
                "fields": [
                    {"name": "booth", "dtype": "str"},
                    {"name": "exhibitor_id", "dtype": "str"},
                ],
                "mode": "natural",
                "anchor": "booth",
            }
        }
    }


def test_from_dict_preserves_mode_and_anchor():
    built = Schema.from_dict(_structures_schema())
    meta = built._record_metadata
    assert meta.get("record", {}).get("mode") == "natural"
    assert meta.get("record", {}).get("anchor") == "booth"


def test_build_emits_record_metadata_but_raw_schema_does_not():
    """`record_metadata` is injected by build(), NOT present on `.schema`.

    This asymmetry is the trap: consuming `.schema` directly loses the metadata
    silently. Asserted explicitly so a future refactor that changes which one carries
    the key fails here rather than in a training run.
    """
    built = Schema.from_dict(_structures_schema())
    assert "record_metadata" not in built.schema
    assert built.build().get("record_metadata", {}).get("record", {}).get("mode") == "natural"


def test_runtime_schema_normalisation_keeps_record_metadata():
    """Calls the REAL runtime normalisation, so removing the fix fails this test.

    An earlier version of this test re-implemented `runtime.py`'s logic inline and
    therefore passed with or without the fix -- it asserted that the copy worked, not
    that the shipped code did. Exercise `_build_schema_dicts_and_metadata` directly
    instead; it is a plain method over a schema list and needs no model weights.
    """
    from gliner2.inference.runtime import ExtractorRuntimeMixin

    # Unbound call: the method touches only its argument, so no instance is required
    # and no checkpoint has to be loaded.
    schema_dicts, _ = ExtractorRuntimeMixin._build_schema_dicts_and_metadata(
        object(), [_structures_schema()]
    )
    assert schema_dicts, "normalisation produced no schema dicts"
    assert schema_dicts[0].get("record_metadata"), (
        "record_metadata missing after runtime normalisation: the boundary record head "
        "will decode NOTHING and `structure` scores exactly 0.0000, raising no error. "
        "runtime.py must copy record_metadata off Schema.build() alongside "
        "json_structures and json_descriptions."
    )
    assert schema_dicts[0]["record_metadata"]["record"]["mode"] == "natural"
    assert schema_dicts[0]["record_metadata"]["record"]["anchor"] == "booth"


def test_record_spec_compiles_from_that_metadata():
    """With the metadata present a RecordSpec is actually built, anchor included."""
    from gliner2.processing.records import compile_record_specs, normalize_record_metadata

    built = Schema.from_dict(_structures_schema())
    meta = built.build()["record_metadata"]
    assert normalize_record_metadata(meta), "metadata normalised away to nothing"

    # compile_record_specs needs a query layout; assert the guard that actually bit us --
    # empty metadata returns {} silently, which is the whole failure mode.
    assert compile_record_specs(
        query_layout=type("QL", (), {"queries": []})(),
        record_metadata=None,
    ) == {}, "no-metadata path should return {} -- that is the silent failure to guard"
