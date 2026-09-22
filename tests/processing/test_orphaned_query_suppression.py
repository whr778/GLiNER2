"""A label whose gold cannot be aligned must not be offered as an absent query.

The mention is skipped when a surface fails to align, but the label's QUERY survives --
so the model is shown the label, the entity is in the text, and the supervision says
"not here". Measured at 1.41% of label queries (649 of 46,122), worst in chfinann at
1.98%, and every one of them is a FALSE negative: nothing is abstractive, every unaligned
surface is physically present (tools/data/measure_surface_alignment.py).

Suppressing the query teaches nothing rather than teaching a falsehood.

Two guards matter as much as the feature:
  * an INTENTIONALLY injected negative is an EMPTY gold list, and must survive;
  * only ENTITIES are suppressed -- dropping a structure field or event role would remove
    an anchor's field query, which is the "declares anchor X but no matching field query"
    abort that killed eb17-best.
"""

from __future__ import annotations

from gliner2.processing.boundary_preprocessing import _orphaned_entity_fields

FIELDS = ["Person", "StockCode", "Date"]

ALIGNED = [(3, 5)]
UNALIGNED = [(-1, -1)]


def _labels(*per_field):
    """`[count, [instance]]` with one instance holding each field's positions."""
    return [1, [list(per_field)]]


def test_label_whose_every_surface_fails_is_orphaned():
    orphaned = _orphaned_entity_fields(FIELDS, _labels(ALIGNED, UNALIGNED, ALIGNED))
    assert orphaned == {1}


def test_injected_negative_is_never_orphaned():
    """`NegativeLabels.inject` writes `{label: []}` -- an empty gold list. That label is
    absent ON PURPOSE and is the positive class of `abstention_loss`; suppressing it
    would delete the very supervision the label-negatives work exists to add."""
    orphaned = _orphaned_entity_fields(FIELDS, _labels(ALIGNED, [], ALIGNED))
    assert orphaned == set()


def test_partially_aligned_label_is_kept():
    """Confirmed scope: only ORPHANED labels are suppressed. A label that keeps at least
    one aligned mention stays, even though another of its mentions was dropped."""
    orphaned = _orphaned_entity_fields(FIELDS, _labels(ALIGNED + UNALIGNED, ALIGNED, ALIGNED))
    assert orphaned == set()


def test_alignment_in_any_instance_saves_the_label():
    labels = [2, [[UNALIGNED, UNALIGNED, ALIGNED], [ALIGNED, UNALIGNED, ALIGNED]]]
    orphaned = _orphaned_entity_fields(FIELDS, labels)
    assert orphaned == {1}


def test_no_gold_at_all_orphans_nothing():
    assert _orphaned_entity_fields(FIELDS, [0, []]) == set()
    assert _orphaned_entity_fields(FIELDS, None) == set()


def test_positions_beyond_the_field_list_are_ignored():
    """A malformed instance longer than the schema must not index past `fields`."""
    labels = [1, [[ALIGNED, UNALIGNED, ALIGNED, UNALIGNED, UNALIGNED]]]
    assert _orphaned_entity_fields(FIELDS, labels) == {1}
