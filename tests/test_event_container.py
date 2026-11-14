"""Tests for EventContainer, role-independent scope gating, and span-boundary hygiene."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    showcase = str(ROOT / "tools" / "ekf_showcase")
    streams = str(ROOT / "datasets" / "disaster_streams")
    for p in (showcase, streams):
        if p not in sys.path:
            sys.path.insert(0, p)
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ec = _load_module("ekf_event_container", "tools/ekf_showcase/event_container.py")
surface = _load_module("ekf_surface_extract", "datasets/disaster_streams/extract.py")


def _obs(t: float, val: float, key: str, role: str = "dead", art: int = None, conf: float = 0.9):
    o = {"t_hours": t, "value": val, "event_key": key, "role": role, "confidence": conf}
    if art is not None:
        o["_article_id"] = art
    return o


def test_span_boundary_defect_catches_truncated_numbers_and_dates_only():
    txt = "Oct. 1, 2024: At least 35,418 deaths and 1.2 million displaced around 11:10 p.m."
    i_date = txt.index("1, 2024")
    assert ec.span_boundary_defect(txt, "1", i_date, i_date + 1) == "calendar_date"

    i_full = txt.index("35,418")
    assert ec.span_boundary_defect(txt, "35,418", i_full, i_full + 6) is None
    assert ec.span_boundary_defect(txt, "35", i_full, i_full + 2) == "partial_right_sep"
    assert ec.span_boundary_defect(txt, "418", i_full + 3, i_full + 6) == "partial_left_sep"

    i_dec = txt.index("1.2 million")
    assert ec.span_boundary_defect(txt, "1", i_dec, i_dec + 1) == "partial_right_sep"

    i_clk = txt.index("11:10")
    assert ec.span_boundary_defect(txt, "11", i_clk, i_clk + 2) == "partial_right_sep"

    # Genuine casualty count after a month name without ', YYYY' must NOT be rejected
    valid = "In October 12 people died in flooding."
    i_ok = valid.index("12")
    assert ec.span_boundary_defect(valid, "12", i_ok, i_ok + 2) is None


def test_english_scale_words_and_bare_decimals():
    val, _ = surface.value_qualifier("1.41 million")
    assert val == 1_410_000

    val_m, _ = surface.value_qualifier("1.2-meter drifts")
    assert val_m is None

    val_int, _ = surface.value_qualifier("35,418")
    assert val_int == 35_418


def test_role_independence_injured_never_alters_dead_decisions():
    states = {"turkey": "turkey", "syria": "syria"}
    dead_rows = [
        _obs(12.5, 1014, "turkey", "dead", art=1),
        _obs(23.4, 2316, "turkey", "dead", art=2),
        _obs(23.4, 1293, "syria", "dead", art=2),
        _obs(72.3, 9057, "turkey", "dead", art=3),
        _obs(72.3, 2530, "turkey", "dead", art=3),
    ]
    injured_rows = [
        _obs(72.3, 380_000, "turkey", "injured", art=3),
        _obs(72.3, 500, "syria", "injured", art=3),
    ]
    kept_dead_only, _, _ = ec.container_gate(
        [dict(x) for x in dead_rows], states
    )
    kept_mixed, _, _ = ec.container_gate(
        [dict(x) for x in dead_rows + injured_rows], states
    )
    for k in ("turkey", "syria", "__aggregate__"):
        d1 = [(x["t_hours"], x["value"]) for x in kept_dead_only.get(k, []) if x["role"] == "dead"]
        d2 = [(x["t_hours"], x["value"]) for x in kept_mixed.get(k, []) if x["role"] == "dead"]
        assert d1 == d2


def test_multi_region_container_stages_unknown_promotes_regional_totals_and_saves_nc():
    states = {s: s for s in ("florida", "georgia", "north carolina", "tennessee", "virginia")}
    rows = [
        _obs(8.3, 1, "virginia", art=1),
        _obs(34.2, 44, "unknown", art=2),
        _obs(65.9, 20, "georgia", art=3),
        _obs(65.9, 64, "tennessee", art=3),
        _obs(94.5, 4, "virginia", art=4),
        _obs(94.5, 120, "florida", art=4),
        _obs(133.7, 166, "unknown", art=5),
        _obs(133.7, 57, "north carolina", art=5),
        _obs(145.5, 57, "north carolina", art=6),
        _obs(153.2, 180, "north carolina", art=7),
        _obs(153.2, 61, "north carolina", art=7),
        _obs(153.2, 4, "unknown", art=7),
        _obs(153.2, 11, "unknown", art=7),
        _obs(223.4, 215, "tennessee", art=8),
        _obs(398.6, 230, "georgia", art=9),
        _obs(887.1, 250, "tennessee", art=10),
    ]
    tagged = [dict(o, _idx=i) for i, o in enumerate(rows)]
    kept, moved, dropped = ec.container_gate(tagged, states)

    # Strict 1-to-1 partition check on _idx
    all_kept_ids = [o["_idx"] for lst in kept.values() for o in lst]
    moved_ids = {o["_idx"] for o in moved}
    dropped_ids = {o["_idx"] for o in dropped}
    assert len(all_kept_ids) + len(dropped_ids) == len(tagged)
    assert set(all_kept_ids).isdisjoint(dropped_ids)
    assert moved_ids.issubset(set(all_kept_ids))

    agg_vals = [o["value"] for o in kept["__aggregate__"]]
    assert agg_vals == [44, 64, 120, 166, 180, 215, 230, 250]

    # North Carolina keeps 57, 57, 61 while 180 promotes to __aggregate__
    nc_vals = [o["value"] for o in kept["north carolina"]]
    assert nc_vals == [57, 57, 61]

    # Late unlocated fragments (4, 11 at t=153.2) are rejected, not added to __aggregate__
    assert {o["value"] for o in dropped} == {4, 11}


def test_single_active_child_reconciles_unknown_and_rejects_repeated_article_spike():
    states = {"Izmir": "Izmir", "Samos": "Samos"}
    rows = [
        _obs(15.7, 30, "unknown", art=1),
        _obs(47.3, 55, "Izmir", art=2),
        _obs(66.8, 79, "Izmir", art=3),
        _obs(77.6, 92, "Izmir", art=4),
        _obs(100.7, 109, "unknown", art=5),
        _obs(118.8, 114, "Izmir", art=6),
        # Four repeated chunk extractions of 10,887 in a single COVID-19 article (art=7)
        _obs(220.4, 10887, "Izmir", art=7),
        _obs(220.4, 10887, "Izmir", art=7),
        _obs(220.4, 10887, "Izmir", art=7),
        _obs(220.4, 10887, "Izmir", art=7),
        _obs(236.8, 115, "unknown", art=8),
        _obs(382.2, 116, "unknown", art=9),
    ]
    kept, moved, dropped = ec.container_gate(rows, states)
    assert "__aggregate__" not in kept
    iz_vals = [o["value"] for o in kept["Izmir"]]
    assert iz_vals == [30, 55, 79, 92, 109, 114, 115, 116]
    assert all(o["value"] == 10887 for o in dropped)
    assert len(dropped) == 4


def test_two_children_without_container_seed_preserve_dominant_and_reject_sibling_leaks():
    states = {"turkey": "turkey", "syria": "syria"}
    rows = [
        _obs(12.5, 1014, "turkey", art=1),
        _obs(23.4, 2316, "turkey", art=2, conf=0.95),
        _obs(23.4, 1293, "turkey", art=2, conf=0.80),
        _obs(23.4, 2316, "syria", art=2, conf=0.70),
        _obs(23.4, 1293, "syria", art=2, conf=0.90),
        _obs(48.0, 5434, "turkey", art=3),
        _obs(48.0, 1832, "turkey", art=3),
        _obs(121.0, 20213, "turkey", art=4),
        _obs(121.0, 3553, "turkey", art=4),
        _obs(121.0, 3553, "syria", art=4),
    ]
    kept, moved, _ = ec.container_gate(rows, states)
    assert moved == []
    assert "__aggregate__" not in kept
    assert [o["value"] for o in kept["turkey"]] == [1014, 2316, 5434, 20213]
    assert [o["value"] for o in kept["syria"]] == [1293, 3553]


def test_decay_role_missing_allows_legitimate_downward_trajectory():
    states = {"Izmir": "Izmir", "Samos": "Samos"}
    rows = [
        _obs(12.0, 100, "Izmir", role="missing", art=1),
        _obs(24.0, 75, "Izmir", role="missing", art=2),
        _obs(48.0, 40, "Izmir", role="missing", art=3),
        _obs(72.0, 12, "Izmir", role="missing", art=4),
    ]
    kept, _, dropped = ec.container_gate(rows, states)
    assert dropped == []
    assert [o["value"] for o in kept["Izmir"]] == [100, 75, 40, 12]
