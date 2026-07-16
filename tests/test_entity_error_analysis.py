"""Tests for the fine-grained entity error analysis (Ortmann 2022, fair
evaluation) in gliner2.training.metrics.

Entity sets are ``{(label, surface)}``. Numeric parity with the FairEval repo
is not checkable here -- it needs token offsets our surface-based gold does not
carry -- so we test the typed-error classification directly plus the key
property: fair scoring avoids the strict double penalty for near-misses.
"""

import pytest

from gliner2.training.metrics import (
    _boundary_subtype,
    _classify_entity_errors,
    _counters,
    _finalize_entity_errors,
    _pr_f1,
    _tally,
)


def _errs(gold, pred):
    return _classify_entity_errors(set(gold), set(pred))


class TestClassify:
    def test_exact_match_is_correct(self):
        c = _errs([("PER", "Marie Curie")], [("PER", "Marie Curie")])
        assert c["COR"] == 1
        assert sum(v for k, v in c.items() if k != "COR") == 0

    def test_same_surface_wrong_label_is_labeling_error(self):
        c = _errs([("LOC", "Washington")], [("PER", "Washington")])
        assert c["LE"] == 1
        assert c["FP"] == 0 and c["FN"] == 0

    def test_system_smaller_is_bes(self):
        c = _errs([("LOC", "New York")], [("LOC", "York")])
        assert c["BES"] == 1

    def test_system_larger_is_bel(self):
        c = _errs([("LOC", "York")], [("LOC", "New York")])
        assert c["BEL"] == 1

    def test_overlap_without_containment_is_beo(self):
        # share "bank"/"america" but neither surface contains the other
        c = _errs([("ORG", "Bank of America")], [("ORG", "America Bank")])
        assert c["BEO"] == 1

    def test_wrong_label_and_boundary_is_lbe(self):
        c = _errs([("LOC", "New York")], [("ORG", "York City")])
        assert c["LBE"] == 1

    def test_no_gold_match_is_false_positive(self):
        c = _errs([], [("PER", "Ghost")])
        assert c["FP"] == 1

    def test_no_pred_match_is_false_negative(self):
        c = _errs([("PER", "Alice")], [])
        assert c["FN"] == 1

    def test_one_to_one_second_pred_cannot_reuse_matched_gold(self):
        # both preds could match the single gold; only one may.
        c = _errs([("LOC", "New York")], [("LOC", "New York"), ("LOC", "New York City")])
        assert c["COR"] == 1
        assert c["FP"] == 1  # the leftover pred, not a second boundary credit


class TestBoundarySubtype:
    def test_subtypes(self):
        assert _boundary_subtype("York", "New York") == "BES"
        assert _boundary_subtype("New York", "York") == "BEL"
        assert _boundary_subtype("America Bank", "Bank of America") == "BEO"


class TestFairScoring:
    def test_fair_reduces_to_strict_without_near_misses(self):
        # one hit, one pure hallucination, one pure miss (no overlap) -> fair == strict
        c = _errs([("PER", "Alice"), ("LOC", "Rome")], [("PER", "Alice"), ("ORG", "Xyzzy")])
        out = _finalize_entity_errors(c)
        assert out["eval_entity_fair_f1"] == pytest.approx(0.5)

    def test_fair_beats_strict_on_label_confusion(self):
        # 1 correct + 1 label confusion. Strict double-penalizes the confusion
        # (FP for ORG + FN for PER); fair charges it as half an error.
        gold = {("LOC", "Paris"), ("PER", "Marie")}
        pred = {("LOC", "Paris"), ("ORG", "Marie")}

        tp, fp, fn = _counters()
        _tally(gold, pred, tp, fp, fn, key=lambda x: x[0])
        _, _, strict_f1 = _pr_f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))

        out = _finalize_entity_errors(_classify_entity_errors(gold, pred))
        assert out["eval_entity_error_COR"] == 1
        assert out["eval_entity_error_LE"] == 1
        assert strict_f1 == pytest.approx(0.5)
        assert out["eval_entity_fair_f1"] == pytest.approx(2 / 3)
        assert out["eval_entity_fair_f1"] > strict_f1

    def test_report_and_keys_present(self):
        out = _finalize_entity_errors(_errs([("PER", "A")], [("PER", "A")]))
        for t in ("COR", "LE", "BES", "BEL", "BEO", "LBE", "FP", "FN"):
            assert f"eval_entity_error_{t}" in out
        assert "fair P / R / F1" in out["eval_entity_error_report"]
