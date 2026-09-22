"""Sanitization edits GOLD; `record_metadata` is DECLARATION. They must stay consistent.

`InputExample.sanitize()` removes structure fields whose values cannot be found in the
text -- "remove invalid parts". It did not look at `record_metadata`, which names the
anchor, so a record could emerge whose declaration pointed at a field that no longer
existed. `compile_record_specs` then aborted the run:

    record 'record' declares anchor 'type' but no matching field query was found in the
    layout; the extractive fields present are ['url']

That killed eb17-best and eb17-smoke, both at ~11% of a 5-epoch run, with no checkpoint
written. It is invisible unless `validate_data: true`, and eb17 is the first base config
to set it -- which is why six traces over ~600,000 samples missed it, all of them run with
validation off.

The quieter failure is the more dangerous one and is also covered here: a natural-mode
structure with no explicit anchor DEFAULTS to the first declared field, so dropping the
first field silently re-points the anchor at a different one. No crash, no warning, wrong
supervision.
"""

from __future__ import annotations

from gliner2.training.data import InputExample, Structure

TEXT = "Ada Lovelace wrote the first algorithm, published in 1843."


def _example(_field_values, **kw):
    # Field VALUES go through `_field_values`; a bare `fields=` kwarg would be captured by
    # `**fields` and create one field literally named "fields".
    return InputExample(
        text=TEXT,
        structures=[Structure(struct_name="record", _field_values=_field_values, **kw)])


def test_losing_an_explicit_anchor_drops_the_structure():
    """'Wikispecies' is not in the text, so the anchor field cannot survive."""
    ex = _example({"type": "Wikispecies", "author": "Ada Lovelace"},
                  mode="natural", anchor="type")
    warnings, valid = ex.sanitize()
    assert not ex.structures, "a structure whose anchor vanished must not survive"
    assert any("lost its anchor field" in w for w in warnings)


def test_a_structure_keeping_its_anchor_survives_and_loses_only_bad_fields():
    ex = _example({"author": "Ada Lovelace", "type": "Wikispecies"},
                  mode="natural", anchor="author")
    ex.sanitize()
    assert len(ex.structures) == 1
    assert "author" in ex.structures[0]._fields
    assert "type" not in ex.structures[0]._fields


def test_a_defaulted_anchor_is_not_silently_repointed():
    """No explicit anchor: `get_record_metadata` defaults to the FIRST field. If that
    field is dropped, the anchor would silently become 'author' -- a different claim
    than the record declares. The structure must be dropped instead."""
    ex = _example({"type": "Wikispecies", "author": "Ada Lovelace"}, mode="natural")
    assert ex.structures[0].get_record_metadata()["record"]["anchor"] == "type"
    warnings, _ = ex.sanitize()
    assert not ex.structures
    assert any("lost its anchor field" in w for w in warnings)


def test_metadata_and_gold_never_disagree_after_sanitize():
    """The invariant the whole fix exists to hold."""
    ex = _example({"type": "Wikispecies", "author": "Ada Lovelace"},
                  mode="natural", anchor="type")
    ex.sanitize()
    for struct in ex.structures:
        meta = struct.get_record_metadata()
        if not meta:
            continue
        anchor = meta[struct.struct_name].get("anchor")
        assert anchor is None or anchor in struct._fields


def test_a_structure_with_no_mode_is_left_alone():
    """Legacy span-era structures declare no mode and compile no record spec, so the
    anchor rule must not delete them."""
    ex = _example({"type": "Wikispecies", "author": "Ada Lovelace"})
    ex.structures[0].mode = None
    ex.sanitize()
    assert len(ex.structures) == 1
    assert "author" in ex.structures[0]._fields
