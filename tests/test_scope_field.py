"""The scope field: extract extent beside the number instead of inferring it downstream.

`location` says WHICH place is named. It cannot say whether the figure is that place's
toll or merely adjacent to it -- TODO item 10's example, "The number of deaths stood at
225 on Friday; two more were recorded in South Carolina", binds 225 -> south carolina with
a perfectly correct location. The ratio gate reconstructs extent from magnitude after the
fact; this field states it.

CATEGORICAL with an abstention class, not a confidence float, and that is measured rather
than stylistic: every one of Helene's 106 `dead` observations carries confidence exactly
1.000 -- contaminants included -- and Turkiye's 89 sit in 0.997-1.000. A scalar from the
same head would very likely be constant too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "datasets/disaster_streams"))
sys.path.insert(0, str(REPO / "tools/ekf_showcase"))

from run_pipeline import SCOPE_CLASSES, normalize_scope  # noqa: E402
from scope_gate import apply_extracted_scope  # noqa: E402

STATES = {"north carolina": "North Carolina", "florida": "Florida"}


def _o(key, value, scope, t=1.0):
    return {"event_key": key, "value": value, "scope": scope, "t_hours": t,
            "role": "dead", "qualifier": "point", "source": "official"}


@pytest.mark.parametrize("raw,expected", [
    ("place", "place"), ("national", "national"), ("sub-place", "sub-place"),
    ("statewide", "place"), ("state", "place"),
    ("nationwide", "national"), ("countrywide", "national"), ("overall total", "national"),
    ("county", "sub-place"), ("a single town", "sub-place"),
    ("", "unclear"), (None, "unclear"), ("something else entirely", "unclear"),
])
def test_scope_normalises_to_declared_classes(raw, expected):
    assert normalize_scope(raw) == expected
    assert normalize_scope(raw) in SCOPE_CLASSES


def test_unknown_scope_is_unclear_not_a_guess():
    """`unclear` defers to the ratio gate. Mapping an unknown onto `place` would assert
    the very thing the field exists to measure."""
    assert normalize_scope("qwertyuiop") == "unclear"


def test_national_figure_is_rerouted_off_the_place_stream():
    """The 225-under-South-Carolina failure, stated as a routing test."""
    obs = [_o("north carolina", 225, "national"), _o("north carolina", 91, "place")]
    kept, moved, dropped, deferred = apply_extracted_scope(obs, STATES)
    assert [o["value"] for o in kept["north carolina"]] == [91]
    assert [o["value"] for o in moved] == [225]
    assert [o["value"] for o in kept["__aggregate__"]] == [225]
    assert not dropped and not deferred


def test_sub_place_figure_is_dropped_from_the_place_stream():
    """`"one"` against a North Carolina truth of 123 -- a single-incident count filed as
    the state total. The ratio gate is blind to it; it only rejects figures too LARGE."""
    obs = [_o("north carolina", 1, "sub-place"), _o("north carolina", 98, "place")]
    kept, moved, dropped, _ = apply_extracted_scope(obs, STATES)
    assert [o["value"] for o in kept["north carolina"]] == [98]
    assert [o["value"] for o in dropped] == [1]
    assert not moved


def test_unclear_is_deferred_and_still_kept():
    obs = [_o("florida", 40, "unclear")]
    kept, moved, dropped, deferred = apply_extracted_scope(obs, STATES)
    assert [o["value"] for o in deferred] == [40]
    assert [o["value"] for o in kept["florida"]] == [40], "deferred must not be discarded"
    assert not moved and not dropped


def test_streams_outside_the_hierarchy_are_untouched():
    """Only declared places are routed; the aggregate stream is not second-guessed."""
    obs = [_o("__aggregate__", 233, "national")]
    kept, moved, dropped, deferred = apply_extracted_scope(obs, STATES)
    assert [o["value"] for o in kept["__aggregate__"]] == [233]
    assert not moved and not dropped
