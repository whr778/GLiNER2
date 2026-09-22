"""A chunk must keep the record DECLARATION, not just the record gold.

`_filter_record_output` rebuilds the per-chunk ``output`` from scratch, and it used to
copy `json_structures` while silently leaving `record_metadata` behind. The result raised
nothing and trained nothing: the structure survived into the chunk as gold the boundary
record head could not compile a spec for, so every CHUNKED long document contributed zero
record supervision. Measured on text2json at window 4096 / stride 3072 before the fix:
98 records chunk, and 235 of the 7,891 chunks carrying a structure had no declaration.

These tests fail on the pre-fix function -- verified by deleting the two carry lines.
"""

from __future__ import annotations

from gliner2.training.chunking import _filter_record_output


def _record():
    """A structure whose anchor surface sits in the FIRST half of the text only."""
    return {
        "json_structures": [{"record": {"name": "Aaron Dessner", "roles": "guitarist"}}],
        "record_metadata": {"record": {"mode": "natural", "anchor": "name"}},
    }


def test_chunk_keeps_record_metadata_beside_its_structures():
    out = _filter_record_output(_record(), "Aaron Dessner plays guitarist here.")
    assert "json_structures" in out
    assert out["record_metadata"] == {"record": {"mode": "natural", "anchor": "name"}}


def test_declaration_is_not_filtered_by_surface():
    """The declaration describes the record TYPE, so it rides even when the anchor
    surface falls outside this window. The anchor's field QUERY comes from the gold
    keys, which pass through unchunked, so it still exists and simply carries no
    mention -- the same shape a short record already has when its anchor value is not
    verbatim in its text."""
    chunk_text = "A later window that never names the anchor value."
    out = _filter_record_output(_record(), chunk_text)
    assert "Aaron Dessner" not in chunk_text
    assert out["record_metadata"]["record"]["anchor"] == "name"
    assert "name" in out["json_structures"][0]["record"]


def test_field_dtypes_ride_with_the_structures():
    """`structure_field_dtypes` reads this carrier to decide cardinality and
    `Schema.build()` writes it, so dropping it would re-create the same silent defect
    one layer down: every field falling back to ZERO_OR_MORE."""
    rec = _record()
    rec["field_dtypes"] = {"record": {"name": "str"}}
    out = _filter_record_output(rec, "Aaron Dessner plays guitarist here.")
    assert out["field_dtypes"] == {"record": {"name": "str"}}


def test_no_structures_means_no_orphan_declaration():
    """Metadata without the structures it describes is not content; it must not
    resurrect a chunk that `_record_has_content` would otherwise drop."""
    out = _filter_record_output(
        {"record_metadata": {"record": {"mode": "natural", "anchor": "name"}}},
        "text with no structure gold at all",
    )
    assert "record_metadata" not in out
