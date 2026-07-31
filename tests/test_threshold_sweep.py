"""Tests for gliner2.training.metrics.sweep_thresholds / _selection_score."""

from gliner2.training.metrics import (
    _selection_score, make_sweeping_compute_metrics, sweep_thresholds,
)


class _FakeModel:
    """batch_extract's output depends on threshold, mimicking a real model
    whose decode surfaces fewer candidates as the threshold rises."""

    def __init__(self, preds_by_threshold):
        self.preds_by_threshold = preds_by_threshold

    def batch_extract(self, texts, schemas, batch_size=8, threshold=0.5):
        return self.preds_by_threshold[threshold]


class _FakeDS:
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


GOLD = {"entities": {"PER": ["Marie Curie"]}}


def _ds():
    return _FakeDS([("Marie Curie discovered radium.", GOLD)])


class TestSelectionScore:
    def test_weights_by_support_not_unweighted_mean(self):
        # entity: perfect score, huge support. event_argument: zero score, tiny
        # support. An unweighted mean would average them down to ~0.5; support
        # weighting should keep the score close to 1.0.
        metrics = {
            "eval_entity_strict_support": 1000,
            "eval_entity_strict_micro_f1": 1.0,
            "eval_event_argument_strict_support": 2,
            "eval_event_argument_strict_micro_f1": 0.0,
        }
        assert _selection_score(metrics) > 0.99

    def test_missing_categories_ignored(self):
        metrics = {"eval_entity_strict_support": 10, "eval_entity_strict_micro_f1": 0.5}
        assert _selection_score(metrics) == 0.5

    def test_no_categories_present_gives_zero(self):
        assert _selection_score({}) == 0.0


class TestSweepThresholds:
    def test_picks_threshold_with_best_score(self):
        model = _FakeModel({
            0.1: [{"entities": {"PER": ["Marie Curie"]}}],  # exact match -> F1=1.0
            0.5: [{"entities": {"PER": []}}],                # nothing -> F1=0.0
            0.9: [{"entities": {"PER": []}}],                # nothing -> F1=0.0
        })

        best_threshold, best_metrics, all_results = sweep_thresholds(
            model, _ds(), thresholds=(0.1, 0.5, 0.9)
        )

        assert best_threshold == 0.1
        assert best_metrics["eval_entity_strict_micro_f1"] == 1.0
        assert set(all_results.keys()) == {0.1, 0.5, 0.9}
        assert all_results[0.5]["eval_entity_strict_micro_f1"] == 0.0

    def test_default_grid_is_five_coarse_points(self):
        from gliner2.training.metrics import DEFAULT_THRESHOLD_GRID
        assert DEFAULT_THRESHOLD_GRID == (0.1, 0.3, 0.5, 0.7, 0.9)


class TestMakeSweepingComputeMetrics:
    """The per-epoch hook selects the threshold that optimizes metric_for_best,
    not a fixed 0.5 -- the fix for bce_posweight shifting the score distribution."""

    def _model(self):
        return _FakeModel({
            0.1: [{"entities": {"PER": ["Marie Curie", "radium"]}}],  # 1 TP + 1 FP -> F1<1
            0.5: [{"entities": {"PER": ["Marie Curie"]}}],            # exact match -> F1=1.0
            0.9: [{"entities": {"PER": []}}],                         # nothing -> F1=0.0
        })

    def test_selects_threshold_maximizing_metric(self):
        hook = make_sweeping_compute_metrics(
            "eval_entity_strict_micro_f1", thresholds=(0.1, 0.5, 0.9),
        )
        out = hook(self._model(), _ds())
        assert out["eval_chosen_threshold"] == 0.5
        assert out["eval_entity_strict_micro_f1"] == 1.0

    def test_greater_is_better_false_selects_min(self):
        hook = make_sweeping_compute_metrics(
            "eval_entity_strict_micro_f1", thresholds=(0.1, 0.5, 0.9),
            greater_is_better=False,
        )
        out = hook(self._model(), _ds())
        assert out["eval_chosen_threshold"] == 0.9
        assert out["eval_entity_strict_micro_f1"] == 0.0
