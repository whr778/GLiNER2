"""Class-weighting schemes: micro and macro are the ends, three schemes sit between.

Harbecke, Chen, Hennig and Alt (2022), "Why only Micro-F1?", Table 2. Values below are
hand-computed from the published formulas, not captured from the implementation.
"""
from collections import Counter

import pytest

from gliner2.training.eval_metrics import _finalize

STRICT = "eval_relation_strict_{}_f1"


def scores(tp, fp, fn):
    m = _finalize("relation", "strict", Counter(tp), Counter(fp), Counter(fn))
    return {k: m.get(STRICT.format(k)) for k in
            ("micro", "weighted", "dodrans", "entropy", "macro")}


def test_imbalanced_matches_hand_computed_weights():
    """90/10 support, per-class F1 1.0 and 0.0.

        weighted  w = [90, 10]                      -> 0.900
        dodrans   w = [90^.75, 10^.75]              -> 29.22 / 34.84  = 0.8386
        entropy   w = [-90log2(.9), -10log2(.1)]    -> 13.68 / 46.90  = 0.2917
        macro     w = [1, 1]                        -> 0.500
    """
    s = scores({"freq": 90}, {}, {"rare": 10})
    assert s["weighted"] == pytest.approx(0.900, abs=1e-3)
    assert s["dodrans"] == pytest.approx(0.8386, abs=1e-3)
    assert s["entropy"] == pytest.approx(0.2917, abs=1e-3)
    assert s["macro"] == pytest.approx(0.500, abs=1e-3)


def test_entropy_is_not_bracketed_by_weighted_and_macro():
    """The property most likely to be assumed and be wrong.

    -n*log2(n/N) peaks at n/N = 1/e, so the 90% class gets LESS weight than the 10%
    one and entropy lands below macro rather than between macro and weighted.
    """
    s = scores({"freq": 90}, {}, {"rare": 10})
    assert s["entropy"] < s["macro"] < s["dodrans"] < s["weighted"]


def test_balanced_classes_collapse_to_macro():
    s = scores({"a": 5, "b": 5}, {}, {"a": 5, "b": 5})
    assert s["weighted"] == pytest.approx(s["macro"])
    assert s["dodrans"] == pytest.approx(s["macro"])
    assert s["entropy"] == pytest.approx(s["macro"])


def test_entropy_omitted_when_one_class_holds_everything():
    """log2(1) = 0 makes every weight zero. Omit it rather than report a fake 0.0."""
    m = _finalize("relation", "strict", Counter({"only": 7}), Counter(), Counter())
    assert STRICT.format("entropy") not in m
    assert STRICT.format("weighted") in m and STRICT.format("dodrans") in m


def test_false_positive_only_label_carries_no_weight():
    """n_i = 0 earns no weight in the three schemes, but still hurts micro and macro."""
    s = scores({"real": 10}, {"ghost": 4}, {})
    assert s["weighted"] == pytest.approx(1.0)      # only `real` has support
    assert s["macro"] < 1.0                          # macro counts `ghost` as a zero
    assert s["micro"] < 1.0                          # the 4 FPs are charged in full


def test_reported_for_events_and_relations_alike():
    for prefix in ("relation", "event_argument", "event_trigger", "event_type", "event"):
        m = _finalize(prefix, "relaxed", Counter({"a": 9}), Counter(), Counter({"b": 3}))
        for scheme in ("weighted", "dodrans", "entropy"):
            assert f"eval_{prefix}_relaxed_{scheme}_f1" in m
