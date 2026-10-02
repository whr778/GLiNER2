"""tools/derive/emit_config.py: measured picks, refusal rules, no structural keys."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "derive"))
import emit_config as E  # noqa: E402


def _calib(pool="per_query", companion=None, blocked=False):
    reach = {"as_built_" + pool: {"coverage": 0.28},
             **{f"per_query_k{k}": {"coverage": c} for k, c in ((16, 0.40), (32, 0.46), (64, 0.62), (128, 0.65))}}
    gpu = {"eval_settings": {"window": 4096, "global_decode": True, "threshold": 0.5, "source": "x"},
           "reachability": reach,
           "span_sweep": {"picked": 0.3, "at_grid_edge": False, "rule": "r", "curve": {"0.3": {"entity": 0.2}}},
           "record_sweep": {"picked": 0.1, "at_grid_edge": False, "rule": "r"}}
    if companion is not None:
        gpu["per_query_companion"] = {"strict_f1": companion}
    return {"model": {"architecture": "boundary", "max_len": 8192, "inference_defaults": None},
            "checkpoint_boundary_head": {"candidate_pool": pool, "candidate_budget": 192},
            "corpus": {"gold_capacity": {"recommended_cap": 128, "largest_group": 115, "pct_groups_over_32": 2.0},
                       "lengths": {"train": {"p99": 1100}}},
            "data_health": {"blocked": blocked, "findings": []}, "gpu": gpu}


def test_reach_pick_is_smallest_k_at_the_plateau():
    cfg, _, _ = E.build("n", "m", "c", _calib())
    assert cfg["model"]["boundary_head"]["start_top_k"] == 64      # 0.62 >= 0.95 * 0.65


def test_shared_pool_switches_only_when_companion_holds_up():
    keep, _, _ = E.build("n", "m", "c", _calib("shared", {"entity": 0.1}))
    switch, _, _ = E.build("n", "m", "c", _calib("shared", {"entity": 0.25}))
    assert "candidate_pool" not in keep["model"]["boundary_head"]
    assert switch["model"]["boundary_head"]["candidate_pool"] == "per_query"


def test_record_gate_and_threshold_come_from_the_sweeps():
    cfg, cm, _ = E.build("n", "m", "c", _calib())
    bh = cfg["model"]["boundary_head"]
    assert bh["record_anchor_threshold"] == 0.1 and bh["record_anchor_threshold_wins"] is True
    assert cfg["eval"]["threshold"] == 0.3
    assert "labels:" not in str(cfg.keys())                         # no inline labels block


def test_no_structural_key_is_ever_emitted():
    cfg, _, _ = E.build("n", "m", "c", _calib("shared", {"entity": 0.25}))
    assert not set(cfg["model"]["boundary_head"]) & E.T._STRUCTURAL_BOUNDARY_KEYS
