"""Tests for the fine-grained labeled-span error analysis (Ortmann 2022, fair
evaluation) in gliner2.training.metrics.

Span sets are ``{(label, surface)}``. Numeric parity with the FairEval repo is
not checkable here -- it needs token offsets our surface-based gold does not
carry -- so we test the typed-error classification directly plus the key
property: fair scoring avoids the strict double penalty for near-misses.
"""

import pytest

from gliner2.training.metrics import (
    _boundary_subtype,
    _classify_span_errors,
    _counters,
    _finalize_span_errors,
    _pr_f1,
    _tally,
    compute_metrics,
)


def _classify(gold, pred):
    return _classify_span_errors(set(gold), set(pred))


def _errs(gold, pred):
    return _classify(gold, pred)[0]


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
        c = _errs([("LOC", "New York")], [("LOC", "New York"), ("LOC", "New York City")])
        assert c["COR"] == 1
        assert c["FP"] == 1


class TestConfusions:
    def test_labeling_error_records_gold_to_pred(self):
        _c, conf = _classify([("LOC", "Washington")], [("PER", "Washington")])
        assert conf[("LOC", "PER")] == 1

    def test_labeling_boundary_error_records_confusion(self):
        _c, conf = _classify([("LOC", "New York")], [("ORG", "York City")])
        assert conf[("LOC", "ORG")] == 1

    def test_boundary_and_correct_produce_no_confusion(self):
        _c, conf = _classify([("LOC", "New York")], [("LOC", "York")])
        assert conf == {}

    def test_confusion_table_rendered_in_report(self):
        counts, conf = _classify([("LOC", "Washington")], [("PER", "Washington")])
        out = _finalize_span_errors("entity", counts, conf)
        assert "label confusions (gold -> pred):" in out["eval_entity_error_report"]
        assert "LOC -> PER: 1" in out["eval_entity_error_report"]


class TestBoundarySubtype:
    def test_subtypes(self):
        assert _boundary_subtype("York", "New York") == "BES"
        assert _boundary_subtype("New York", "York") == "BEL"
        assert _boundary_subtype("America Bank", "Bank of America") == "BEO"


class TestFairScoring:
    def test_fair_reduces_to_strict_without_near_misses(self):
        counts, conf = _classify(
            [("PER", "Alice"), ("LOC", "Rome")], [("PER", "Alice"), ("ORG", "Xyzzy")]
        )
        out = _finalize_span_errors("entity", counts, conf)
        assert out["eval_entity_fair_micro_f1"] == pytest.approx(0.5)

    def test_fair_beats_strict_on_label_confusion(self):
        # 1 correct + 1 label confusion. Strict double-penalizes the confusion
        # (FP for ORG + FN for PER); fair charges it as half an error.
        gold = {("LOC", "Paris"), ("PER", "Marie")}
        pred = {("LOC", "Paris"), ("ORG", "Marie")}

        tp, fp, fn = _counters()
        _tally(gold, pred, tp, fp, fn, key=lambda x: x[0])
        _, _, strict_f1 = _pr_f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))

        counts, conf = _classify(gold, pred)
        out = _finalize_span_errors("entity", counts, conf)
        assert out["eval_entity_error_COR"] == 1
        assert out["eval_entity_error_LE"] == 1
        assert strict_f1 == pytest.approx(0.5)
        assert out["eval_entity_fair_micro_f1"] == pytest.approx(2 / 3)
        assert out["eval_entity_fair_micro_f1"] > strict_f1

    def test_support_is_total_gold(self):
        counts, conf = _classify(
            [("PER", "A"), ("PER", "B"), ("LOC", "C")], [("PER", "A"), ("ORG", "B")]
        )
        # gold = 3 (COR A, LE B, FN C) -> support 3
        out = _finalize_span_errors("entity", counts, conf)
        assert out["eval_entity_fair_support"] == 3


class TestPrefix:
    def test_prefix_flows_into_keys(self):
        counts, conf = _classify([("Attack", "bombed")], [("Movement", "bombed")])
        out = _finalize_span_errors("event_trigger", counts, conf)
        assert out["eval_event_trigger_error_LE"] == 1
        assert "eval_event_trigger_fair_micro_f1" in out
        assert out["eval_event_trigger_error_report"].startswith("event_trigger error analysis")


class TestEndToEndEvents:
    """Drive compute_metrics so the trigger/argument diagnostics are wired."""

    class _FakeModel:
        def batch_extract(self, texts, schemas, batch_size=8, threshold=0.5):
            return [{"event_extraction": {"Attack": [{
                "triggers": ["bombed the base"],          # gold "bombed" is a substring -> BEL
                "arguments": [{"role": "Attacker", "entity": "base"}],  # gold role Target -> LE
            }]}}]

    class _FakeDS:
        def __init__(self, pairs): self.pairs = pairs
        def __len__(self): return len(self.pairs)
        def __getitem__(self, i): return self.pairs[i]

    def test_trigger_and_argument_diagnostics_emitted(self):
        gold = {"events": [{
            "event_type": "Attack",
            "triggers": ["bombed"],
            "arguments": [{"role": "Target", "entity": "base"}],
        }]}
        ds = self._FakeDS([("rebels bombed the base", gold)])
        m = compute_metrics(self._FakeModel(), ds)

        assert m["eval_event_trigger_error_BEL"] == 1
        assert m["eval_event_argument_error_LE"] == 1
        assert "Target -> Attacker: 1" in m["eval_event_argument_error_report"]
        assert "eval_event_trigger_fair_micro_f1" in m
        assert "eval_event_argument_fair_micro_f1" in m
