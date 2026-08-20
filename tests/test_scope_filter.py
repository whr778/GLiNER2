"""The scope-membership stage: reject figures keyed outside the event's declared scope.

Pinned against the real Helene keys and the corrected audit labels, because the value of
this stage is a measured number (4/6 cross-event at 7.3% FP) and a refactor that quietly
changes the dispositions would change that number.
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_pipeline", REPO / "tools" / "ekf_showcase" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


rp = _load()
ROLLUP = json.loads((REPO / "datasets" / "helene2024" / "rollup.json").read_text(encoding="utf-8"))


def _obs(*keys):
    return [{"event_key": k, "value": 1, "role": "dead", "t_hours": float(i)}
            for i, k in enumerate(keys)]


def test_declared_parts_and_aggregate_are_kept():
    kept, rejected, _ = rp.scope_filter(
        _obs("north carolina", "florida", "__aggregate__"), ROLLUP)
    assert len(kept) == 3 and not rejected


def test_out_of_scope_places_are_rejected():
    """The four cross-event cases the stage is measured to catch, by their real keys."""
    kept, rejected, _ = rp.scope_filter(
        _obs("mexico", "puerto rico", "bosnia", "reading pennsylvania"), ROLLUP)
    assert not kept
    assert {o["_out_of_scope"] for o in rejected} == {
        "mexico", "puerto rico", "bosnia", "reading pennsylvania"}


def test_unresolved_keys_are_kept_not_rejected():
    """A key that resolved to NO place is 'location unknown', not 'other event'.

    Rejecting these is what put the false-positive rate at 7.3% instead of ~2%: four of
    the six false positives were bare event-type keys with no place attached.
    """
    roll = dict(ROLLUP, unresolved=["unknown", "Storm", "Floods"])
    kept, rejected, unresolved = rp.scope_filter(_obs("Storm", "Floods", "unknown"), roll)
    assert len(kept) == 3, "unresolved keys must survive"
    assert not rejected
    assert len(unresolved) == 3, "and be reported so they can be aliased"


def test_no_declared_hierarchy_means_no_opinion():
    """An event with no declared scope must not have one invented for it."""
    kept, rejected, unresolved = rp.scope_filter(_obs("anywhere", "elsewhere"), {})
    assert len(kept) == 2 and not rejected and not unresolved


def test_it_cannot_see_in_scope_cross_event():
    """The residue the router exists for, pinned so nobody claims this stage solves it.

    A Taiwan typhoon keyed to `tennessee` and the 1916 hurricanes keyed to
    `north carolina` are cross-event and IN scope. Same place, different incident.
    """
    kept, rejected, _ = rp.scope_filter(_obs("tennessee", "north carolina"), ROLLUP)
    assert len(kept) == 2 and not rejected


def test_runs_after_rollup_not_before():
    """`Storm|asheville` is in scope only once the rollup folds it to north carolina.

    Note the key FORMAT: apply_rollup partitions on "|" and folds the place half, so a
    bare `asheville` with no type prefix is left alone. The stage therefore has to run
    after the rollup AND on keys the rollup can actually read.
    """
    obs = _obs("Storm|asheville")
    _, rejected_before, _ = rp.scope_filter(obs, ROLLUP)
    assert rejected_before, "unfolded city key looks out of scope"

    rp.apply_rollup(obs, ROLLUP)
    assert obs[0]["event_key"] == "north carolina"
    after, rejected_after, _ = rp.scope_filter(obs, ROLLUP)
    assert len(after) == 1 and not rejected_after
