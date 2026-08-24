"""Guards for two defects that cost this project real time.

1. The EKF maintained P at every step and returned only the mean, so the one thing an
   EKF offers over a last-value baseline -- a statement of how sure it is -- never
   reached any consumer.
2. `vector_state_test` scored the aggregate-constraint experiment with a range-normalised
   macro-average ONLY. That weighs a state ranging 1->2 like one ranging 6->123, and it
   inverted the verdict: the aggregate LOSES on that metric and WINS on absolute error.
   See vector_state_results/METRIC_INVERTED_THE_VERDICT.md.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "datasets/disaster_streams"))
sys.path.insert(0, str(REPO / "tools/ekf_showcase"))

import evaluate as ekf  # noqa: E402

GRID = [0.0, 6.0, 12.0, 48.0, 120.0]
OBS = [{"t_hours": 0.0, "value": 10, "source": "official", "qualifier": "point"},
       {"t_hours": 6.0, "value": 40, "source": "major_outlet", "qualifier": "at_least"},
       {"t_hours": 12.0, "value": 44, "source": "official", "qualifier": "point"}]


@pytest.mark.parametrize("role", ["dead", "injured", "missing"])
def test_ci_means_match_the_point_estimator(role):
    """Exposing sigma must not perturb the estimate six callers already depend on."""
    assert [c["mean"] for c in ekf.est_ekf_ci(OBS, GRID, role)] == ekf.est_ekf(OBS, GRID, role)


def test_sigma_widens_without_news_and_narrows_on_agreement():
    """The behaviour that makes sigma worth reporting at all.

    Flat mean, growing uncertainty: the point estimate cannot distinguish "settled" from
    "nobody has reported in four days" and this is what does.
    """
    ci = ekf.est_ekf_ci(OBS, GRID, "dead")
    assert ci[1]["sigma"] < ci[0]["sigma"], "agreeing reports must reduce uncertainty"
    assert ci[-1]["sigma"] > ci[2]["sigma"] * 2, "silence must widen the band"
    assert ci[-1]["mean"] == pytest.approx(ci[2]["mean"]), "mean flat while sigma grows"


def test_ci_is_absent_not_zero_before_any_report():
    """A band of zero would read as certainty about a number nobody has reported."""
    ci = ekf.est_ekf_ci([{"t_hours": 99.0, "value": 5, "source": "official",
                          "qualifier": "point"}], [0.0, 1.0], "dead")
    assert all(c["sigma"] is None and c["lo"] is None for c in ci)


def test_bounds_bracket_the_mean_and_stay_non_negative():
    for c in ekf.est_ekf_ci(OBS, GRID, "dead"):
        if c["sigma"] is not None:
            assert c["lo"] <= c["mean"] <= c["hi"]
            assert c["lo"] >= 0.0, "a negative death toll is not a bound"
            assert c["hi"] - c["mean"] == pytest.approx(1.96 * c["sigma"], rel=1e-6)


def test_vector_state_reports_absolute_metrics_not_only_nrmse():
    """The regression that inverted a published verdict.

    `run_trial` must return range-normalised AND absolute error. If this collapses back to
    a single normalised number, a 1-death state can outvote a 120-death one again.
    """
    import vector_state_test as vst

    truth = vst.truth_series(REPO / "datasets/helene2024/ground_truth.json")
    a, b, _ = vst.run_trial(truth, random.Random(0), 0.35, 0.10, 9.0)
    assert len(a) == 3 and len(b) == 3, "expected (nrmse, mean abs RMSE, total RMSE)"
    for v in (*a, *b):
        assert isinstance(v, float) and math.isfinite(v)
    # The absolute columns are in DEATHS, so they must not be on the normalised scale.
    assert a[1] > a[0], "absolute per-state RMSE should not be a normalised fraction"
    assert a[2] > a[1], "national-total RMSE should exceed mean per-state RMSE"
