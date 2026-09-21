"""The spatial anchor: flag an observation whose PLACE is outside the event's footprint.

Three text signals infer ownership from the words near the number, and a Helene feed names
Milton and Katrina constantly. Measured against the corrected audit labels: A 3/6 at 23.2%
FP, B 3/6 at 19.5%, C 0/6. The spatial anchor reads the place the pipeline ALREADY bound and
scores 4/6 at 1.2% FP -- better on both axes, with no model call.

ABSTAINING IS THE LOAD-BEARING PART. `event_key` is sometimes a TYPE (`Storm`, `Floods`)
rather than a place. Flagging those would manufacture exactly the false positives that make
A and B unshippable, so they are passed, never flagged.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ekf_showcase"))
from spatial_anchor import spatial_flag, ctx_key, load_footprint  # noqa: E402

ALIASES = {"asheville": "north carolina", "east tennessee": "tennessee",
           "carolinas": "__aggregate__", "north carolina": "north carolina"}
FOOTPRINT = {"florida", "georgia", "south carolina", "north carolina",
             "tennessee", "virginia", "__aggregate__"}


def flag(key):
    return spatial_flag(key, ALIASES, FOOTPRINT)


@pytest.mark.parametrize("key", ["bosnia", "mexico", "puerto rico", "scotland",
                                 "reading pennsylvania", "taiwan"])
def test_a_place_outside_the_footprint_is_flagged(key):
    assert flag(key) is True


@pytest.mark.parametrize("key", ["north carolina", "florida", "virginia"])
def test_a_place_inside_the_footprint_passes(key):
    assert flag(key) is False


def test_an_alias_resolves_before_the_test():
    """`asheville` is inside; testing the raw string would flag it."""
    assert flag("asheville") is False
    assert flag("east tennessee") is False


def test_the_aggregate_key_is_inside():
    """`carolinas`/`southeastern us` roll up to __aggregate__, which IS the event."""
    assert flag("carolinas") is False
    assert flag("__aggregate__") is False


@pytest.mark.parametrize("key", ["Storm", "Floods", "Election", "storm"])
def test_a_type_key_abstains_rather_than_flagging(key):
    """THE POINT: no spatial evidence must never become a positive."""
    assert flag(key) is None


def test_a_missing_key_abstains():
    assert flag(None) is None
    assert flag("") is None


def test_case_and_whitespace_do_not_matter():
    assert flag("  BOSNIA ") is True
    assert flag(" North Carolina ") is False


def test_ctx_key_matches_the_binding_probe():
    """Labels must join across the two tools, so the keying has to be identical."""
    from event_binding_probe import _ctx_key
    ctx, val = "  Helene's death   toll HITS 230 today ", 230
    assert ctx_key(ctx, val) == _ctx_key(ctx, val)


def test_load_footprint_reads_the_events_own_rollup(tmp_path):
    import json
    p = tmp_path / "rollup.json"
    p.write_text(json.dumps({
        "aliases": {"Asheville": "north carolina"},
        "hierarchy": {"aggregate": "__aggregate__", "parts": ["north carolina", "florida"]},
    }), encoding="utf-8")
    aliases, footprint = load_footprint(p)
    assert aliases["asheville"] == "north carolina", "aliases must be lowercased"
    assert footprint == {"north carolina", "florida", "__aggregate__"}


# --------------------------------------------------------------------------------------
# The temporal anchor. It exists to add ONE case the spatial anchor structurally cannot
# see -- the 1916 Appalachian hurricanes, correctly keyed `north carolina` and 108 years
# early -- and its evidence is that single positive. Treat it as a conservative complement.
# --------------------------------------------------------------------------------------

from spatial_anchor import temporal_flag  # noqa: E402


def test_an_ancient_year_is_flagged():
    assert temporal_flag("the 1916 Appalachian hurricanes killed 80") is True


@pytest.mark.parametrize("ctx", [
    "Helene is already the deadliest hurricane to hit the mainland U.S. since Katrina in 2005",
    "Helene passed the 35 people killed in the state after Hurricane Hugo in 1989",
    "the wettest storm since August 2023.All five who died were in one Florida county",
])
def test_a_comparative_clause_abstains(ctx):
    """THE FP MODE THAT DECIDES THE DESIGN. 10 of 81 genuine Helene observations carry a
    non-2024 year, essentially all in a comparison -- the year is attached to the COMPARISON,
    not to the figure. A permissive rule inherits exactly the proximity-is-not-attachment
    failure that caps signals A and B: a year<2010 rule scores 11.1% FP."""
    assert temporal_flag(ctx) is None


def test_no_year_abstains():
    assert temporal_flag("at least 16 killed in Bosnia floods") is None
    assert temporal_flag("") is None
    assert temporal_flag(None) is None


def test_it_never_returns_false():
    """Absence of an ancient year is not evidence a figure is current, so this signal only
    ever ADDS a flag on top of the spatial one -- it must never overrule it."""
    for ctx in ("nothing here", "in 2024 Helene struck", "since Katrina in 2005", ""):
        assert temporal_flag(ctx) is not False


def test_the_cutoff_is_configurable_and_permissive_ones_are_worse():
    ctx = "the deadliest since Katrina in 2005"
    assert temporal_flag(ctx, cutoff=1950) is None
    assert temporal_flag(ctx, cutoff=2010) is True, "a permissive cutoff is what creates FPs"
