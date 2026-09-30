"""Tests for tools/data/merge_rams_passages.py."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "merge_rams_passages",
    Path(__file__).resolve().parent.parent / "tools" / "data" / "merge_rams_passages.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def _ev(etype, trig, *args):
    return {"event_type": etype, "triggers": [trig],
            "arguments": [{"role": r, "entity": e} for r, e in args]}


def test_na_catch_all_collapses_into_the_specific_subtype_with_arguments_unioned():
    out = m.merge_events([_ev("mv.art.hide", "smuggled", ("agent", "A")),
                          _ev("mv.art.n/a", "smuggled", ("artifact", "B"))])
    assert [e["event_type"] for e in out] == ["mv.art.hide"]
    assert {(a["role"], a["entity"]) for a in out[0]["arguments"]} == {("agent", "A"), ("artifact", "B")}


def test_different_parent_types_for_one_trigger_are_both_kept():
    out = m.merge_events([_ev("justice.arrest.jail", "imprisoned"),
                          _ev("movement.person.preventexit", "imprisoned")])
    assert sorted(e["event_type"] for e in out) == ["justice.arrest.jail", "movement.person.preventexit"]


def test_identical_events_and_distinct_triggers():
    out = m.merge_events([_ev("conflict.attack", "shot", ("victim", "V")),
                          _ev("conflict.attack", "shot", ("victim", "V")),
                          _ev("life.die", "killed", ("victim", "V"))])
    assert len(out) == 2
    assert next(e for e in out if e["triggers"] == ["shot"])["arguments"] == [{"role": "victim", "entity": "V"}]
