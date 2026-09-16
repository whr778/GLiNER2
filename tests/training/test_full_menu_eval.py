"""The full-menu eval mode: precision as a measurement rather than an identity.

`_schema_from_gold` builds the menu from each document's own gold, so the model is asked
"which of these are here?" where every option is there by construction. `event_type` has no
span to get wrong, so its precision is pinned at 1.0000 and F1 = 2R/(1+R) EXACTLY -- verified
against 12 of 12 readings on file. Widening the menu to the real taxonomy is what makes the
number mean something; keeping it OUT of the default keys is what keeps every historical
number comparable.
"""

from gliner2.training.eval_metrics import _widen_with_absent

MENU = {
    "entities": ["Chemical", "Disease", "Gene", "Species"],
    "events": {"Attack": ["Target"], "Experiment": ["Subject"], "Deploy": ["Unit"]},
    "relations": ["inhibits", "treats", "binds"],
}


def test_absent_entity_labels_are_added_and_gold_is_untouched():
    gold = {"entities": {"Chemical": "", "Disease": ""}}
    wide = _widen_with_absent(gold, MENU)

    assert set(wide["entities"]) == set(MENU["entities"])
    assert wide["entities"]["Chemical"] == "", "a gold label keeps its own value"


def test_absent_event_types_are_added_with_their_roles():
    gold = {"events": {"Experiment": ["Subject"]}}
    wide = _widen_with_absent(gold, MENU)

    assert set(wide["events"]) == {"Attack", "Experiment", "Deploy"}
    assert wide["events"]["Attack"] == ["Target"]
    assert wide["events"]["Experiment"] == ["Subject"], "the gold entry wins over the menu"


def test_absent_relations_are_added_without_duplicating_gold():
    gold = {"relations": [{"inhibits": {"head": "a", "tail": "b"}}]}
    wide = _widen_with_absent(gold, MENU)

    names = [n for r in wide["relations"] for n in r]
    assert sorted(names) == ["binds", "inhibits", "treats"]
    assert names.count("inhibits") == 1


def test_a_dimension_the_record_does_not_have_is_NOT_invented():
    """Widening must not add a task the record never carried -- that would change which
    documents are scored for that head, not just the menu they are scored against."""
    gold = {"entities": {"Chemical": ""}}
    wide = _widen_with_absent(gold, MENU)

    assert "events" not in wide
    assert "relations" not in wide


def test_widening_is_a_no_op_without_a_menu():
    gold = {"entities": {"Chemical": ""}}
    assert _widen_with_absent(gold, {}) == gold
