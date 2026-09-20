"""TypedRole: an event role edge may only land on a compatibly typed span.

The gate the plan demands (step 2): a test that FAILS without the constraint -- a wrongly
typed edge is admitted before and refused after.
"""
from types import SimpleNamespace

import pytest

from gliner2.joint_ie.constraints import TypedRole, _unqualify
from gliner2.models.base import display_query_type, qualified_query_type


def node(entity_type, start, end):
    return SimpleNamespace(entity_type=entity_type, start=start, end=end)


def edge(relation_type, tail_role, start, end):
    """A role edge as the optimizer resolves it: tail is the ROLE-keyed mention node."""
    return SimpleNamespace(relation_type=relation_type,
                           head=node("7::Trigger", 0, 1),
                           tail=node(tail_role, start, end))


LOC = TypedRole(event_type="Attack", role="Location", allowed_types=("GPE", "Location"))


def test_wrongly_typed_filler_is_refused():
    """Paris-the-PERSON cannot fill Attack/Location when GPE is required."""
    e = edge("Attack::9::Location", "9::Location", 10, 15)
    nodes = [node("Person", 10, 15), e.tail]
    assert LOC.allows(e, (), nodes) is False


def test_correctly_typed_filler_is_admitted():
    e = edge("Attack::9::Location", "9::Location", 10, 15)
    nodes = [node("GPE", 10, 15), e.tail]
    assert LOC.allows(e, (), nodes) is True


def test_without_the_constraint_the_same_edge_is_admitted():
    """The control arm of the gate: no allowed_types means no opinion."""
    e = edge("Attack::9::Location", "9::Location", 10, 15)
    nodes = [node("Person", 10, 15), e.tail]
    assert TypedRole(event_type="Attack", role="Location").allows(e, (), nodes) is True


def test_untyped_span_is_allowed_by_default_and_refused_when_required():
    """No entity node at that span = no evidence. Default allows; require_typed refuses."""
    e = edge("Attack::9::Location", "9::Location", 10, 15)
    nodes = [e.tail]                                    # only the role-keyed node
    assert LOC.allows(e, (), nodes) is True
    strict = TypedRole(event_type="Attack", role="Location",
                       allowed_types=("GPE",), require_typed=True)
    assert strict.allows(e, (), nodes) is False


def test_the_role_keyed_node_cannot_type_itself():
    """Without discarding the tail's own role name the constraint is vacuous."""
    e = edge("Attack::9::Location", "9::Location", 10, 15)
    vacuous = TypedRole(event_type="Attack", role="Location", allowed_types=("Location",))
    assert vacuous.allows(e, (), [e.tail]) is True          # no evidence, not a match
    assert vacuous.allows(e, (), [node("Person", 10, 15), e.tail]) is False


def test_other_events_and_roles_are_untouched():
    other_role = edge("Attack::4::Target", "4::Target", 10, 15)
    other_event = edge("Meeting::9::Location", "9::Location", 10, 15)
    bad = [node("Person", 10, 15)]
    assert LOC.allows(other_role, (), bad + [other_role.tail]) is True
    assert LOC.allows(other_event, (), bad + [other_event.tail]) is True


def test_a_span_typed_several_ways_passes_on_any_allowed_type():
    e = edge("Attack::9::Location", "9::Location", 10, 15)
    nodes = [node("Person", 10, 15), node("GPE", 10, 15), e.tail]
    assert LOC.allows(e, (), nodes) is True


@pytest.mark.parametrize("raw", ["7::Location", "Location", "0::head", "not::adigit",
                                 "12::Event<Subject>", "", "::x"])
def test_unqualify_mirrors_display_query_type(raw):
    """The local mirror must not drift from the real one -- today's recurring bug class."""
    assert _unqualify(raw) == display_query_type(raw)


def test_unqualify_mirrors_on_generated_qualified_types():
    for qid in (0, 7, 913):
        for name in ("Location", "Date", "Event<Subject>", "head"):
            q = qualified_query_type(qid, name)
            assert _unqualify(q) == display_query_type(q) == name
