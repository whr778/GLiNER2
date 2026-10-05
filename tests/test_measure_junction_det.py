"""tools/train/measure_junction_det.py: the DET maths and the train/val pairing (no model load)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
import measure_junction_det as J  # noqa: E402

# (score, is_gold_argument, owner_is_gold_trigger)
POINTS = [(0.9, True, True), (0.6, True, False), (0.2, True, True),
          (0.8, False, False), (0.1, False, False), (0.05, False, False), (0.01, False, False)]


def test_rates_at_the_gate():
    d = J.det(POINTS, 0.5)
    assert d["miss_at_op"] == pytest.approx(1 / 3)      # 0.2 falls below the gate
    assert d["fa_at_op"] == pytest.approx(1 / 4)        # 0.8 clears it
    assert d["correct_at_op"] == pytest.approx(1 / 3)   # 0.6 clears the gate but its owner is wrong


def test_curve_hits_every_positive_score_and_the_gate():
    thresholds = {t for t, _, _ in J.det(POINTS, 0.5)["curve"]}
    assert {0.9, 0.6, 0.2, 0.5} <= thresholds


def test_miss_rate_at_a_false_alarm_budget():
    # FA <= 0.01 means no negative may pass, so the threshold must exceed 0.8: only 0.9 survives
    assert J.det(POINTS, 0.5)["miss_at_fa"]["0.01"] == pytest.approx(2 / 3)


def test_one_sided_input_reports_nothing():
    assert J.det([(0.5, True, True)], 0.5) == {}


def test_family_pairs_train_and_val_files():
    assert J.family("cmnee_ner") == J.family("scaling_joint/cmnee") == "cmnee"
    assert J.family("casie") == J.family("scaling_joint/casie") == "casie"
