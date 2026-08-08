"""Every query marker the processor emits must get a layout query.

If a marker type is missing from `_EXTRACTIVE_MARKERS`, the layout under-counts
queries while `query_states`/`query_mask` still carry one entry per marker. Nothing
complains until gold injection compares a `[B, Q_gold, G]` mask against a
`[B, Q_marker, 1]` one and dies at `proposal.py`'s `gold_mask & query_mask`.

That is exactly what happened with `[V]` (event roles: trigger + arguments): every
event group produced ZERO layout queries, so boundary training blew up on the first
step of any events corpus the moment records or relations were enabled.
"""

from __future__ import annotations

from gliner2.processing.boundary_preprocessing import (
    _EXTRACTIVE_MARKERS,
    _extractive_fields,
)
from gliner2.processor import SchemaTransformer


def test_every_processor_query_marker_is_extractive():
    """The two marker sets must agree, or layouts silently under-count queries."""
    emitted = {
        SchemaTransformer.E_TOKEN,   # entities
        SchemaTransformer.C_TOKEN,   # classification choices
        SchemaTransformer.R_TOKEN,   # json-structure fields
        SchemaTransformer.V_TOKEN,   # event roles (trigger + arguments)
    }
    missing = emitted - set(_EXTRACTIVE_MARKERS)
    assert not missing, (
        f"markers emitted by the processor but absent from _EXTRACTIVE_MARKERS: "
        f"{sorted(missing)} -- these groups will get zero layout queries while their "
        f"markers still occupy query slots"
    )


def test_event_role_markers_produce_fields():
    """An events group must yield one field per role; it used to yield none."""
    tokens = ["(", "[P]", "Cyber.Ransom", "(",
              "[V]", "trigger", "[V]", "Victim", "[V]", "Price", ")", ")"]
    assert _extractive_fields(tokens) == ["trigger", "Victim", "Price"]


def test_entity_and_structure_markers_still_work():
    """Guard against a fix that widens one marker set and breaks another."""
    assert _extractive_fields(
        ["(", "[P]", "entities", "(", "[E]", "Money", "[E]", "Person", ")", ")"]
    ) == ["Money", "Person"]
    assert _extractive_fields(
        ["(", "[P]", "order", "(", "[R]", "buyer", "[R]", "item", ")", ")"]
    ) == ["buyer", "item"]
