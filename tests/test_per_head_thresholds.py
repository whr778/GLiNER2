"""Per-head decision thresholds: picked per decode head from the sweep already run, applied by splicing one
decode per distinct value, and shipped beside the global pick (gliner2.training.eval_metrics; tools/train/train.py)."""
import json
import sys
from pathlib import Path

import pytest

from gliner2.training.eval_metrics import (
    compute_metrics, compute_metrics_per_head, per_head_report, pick_per_head_thresholds, splice_heads,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))


class _FakeModel:
    """batch_extract returns a fixed prediction per threshold; relations come back under relation_extraction."""

    def __init__(self, preds_by_threshold):
        self.preds_by_threshold = preds_by_threshold
        self.calls = []

    def batch_extract(self, texts, schemas, batch_size=8, threshold=0.5):
        self.calls.append(threshold)
        return [dict(p) for p in self.preds_by_threshold[threshold]]


class _DS:
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


GOLD = {"entities": {"PER": ["Marie Curie"]}, "relations": [{"discovered": {"head": "Marie Curie", "tail": "radium"}}],
        "classifications": [{"task": "topic", "labels": ["science", "sport"], "true_label": "science"}]}
TEXT = "Marie Curie discovered radium."
HIT_ENT, MISS_ENT = {"PER": ["Marie Curie"]}, {"PER": ["radium"]}
HIT_REL, MISS_REL = {"discovered": [{"head": "Marie Curie", "tail": "radium"}]}, {"discovered": []}


def _model():
    """Entities are right only at 0.1, relations only at 0.5, classification right everywhere."""
    return _FakeModel({
        0.1: [{"entities": HIT_ENT, "relation_extraction": MISS_REL, "topic": "science"}],
        0.3: [{"entities": MISS_ENT, "relation_extraction": MISS_REL, "topic": "science"}],
        0.5: [{"entities": MISS_ENT, "relation_extraction": HIT_REL, "topic": "science"}],
    })


def _sweep(model):
    return {t: compute_metrics(model, _DS([(TEXT, GOLD)]), threshold=t, report=False) for t in (0.1, 0.3, 0.5)}


class TestPick:
    def test_each_head_takes_its_own_best(self):
        picks = pick_per_head_thresholds(_sweep(_model()), global_threshold=0.3)
        assert picks["entity"] == 0.1
        assert picks["relation"] == 0.5

    def test_flat_head_keeps_the_global_pick_not_the_first_grid_point(self):
        # classification is argmax over the menu: identical at every threshold, so max() must not hand it 0.1
        picks = pick_per_head_thresholds(_sweep(_model()), global_threshold=0.3)
        assert picks["classification"] == 0.3

    def test_partial_tie_with_the_global_keeps_the_global(self):
        # entities right at 0.1 AND at the global 0.3, wrong at 0.5: no measured gain, so no move off 0.3
        model = _FakeModel({
            0.1: [{"entities": HIT_ENT, "relation_extraction": MISS_REL, "topic": "science"}],
            0.3: [{"entities": HIT_ENT, "relation_extraction": MISS_REL, "topic": "science"}],
            0.5: [{"entities": MISS_ENT, "relation_extraction": HIT_REL, "topic": "science"}],
        })
        picks = pick_per_head_thresholds(_sweep(model), global_threshold=0.3)
        assert picks["entity"] == 0.3
        assert picks["relation"] == 0.5

    def test_head_without_support_is_absent(self):
        picks = pick_per_head_thresholds(_sweep(_model()), global_threshold=0.3)
        assert "structure" not in picks and "event" not in picks

    def test_report_marks_flat_and_scores_both_points(self):
        by_t = _sweep(_model())
        report = per_head_report(by_t, pick_per_head_thresholds(by_t, 0.3), 0.3)
        assert report["classification"]["flat"] is True
        assert report["entity"] == {"threshold": 0.1, "f1_at_own": 1.0, "f1_at_global": 0.0, "flat": False}


class TestSplice:
    def test_per_head_scores_each_head_at_its_own_threshold(self):
        model = _model()
        m = compute_metrics_per_head(model, _DS([(TEXT, GOLD)]), {"entity": 0.1, "relation": 0.5}, default=0.3, report=False)
        assert m["eval_entity_strict_micro_f1"] == 1.0
        assert m["eval_relation_strict_micro_f1"] == 1.0
        assert m["eval_classification_strict_micro_f1"] == 1.0
        assert m["per_head_thresholds"] == {"entity": 0.1, "relation": 0.5}
        # one decode per distinct value, the default included
        assert sorted(model.calls) == [0.1, 0.3, 0.5]

    def test_no_single_threshold_matches_the_spliced_result(self):
        for t in (0.1, 0.3, 0.5):
            m = compute_metrics(_model(), _DS([(TEXT, GOLD)]), threshold=t, report=False)
            assert m["eval_entity_strict_micro_f1"] + m["eval_relation_strict_micro_f1"] < 2.0

    def test_splice_routes_every_key_to_its_head(self):
        runs = {
            0.1: {"entities": "E@0.1", "relation_extraction": "R@0.1", "event_extraction": "V@0.1", "topic": "C@0.1", "casualty": "S@0.1"},
            0.5: {"entities": "E@0.5", "relation_extraction": "R@0.5", "event_extraction": "V@0.5", "topic": "C@0.5", "casualty": "S@0.5"},
        }
        out = splice_heads(GOLD, runs, {"entity": 0.1, "relation": 0.5, "event": 0.5, "classification": 0.1, "structure": 0.5}, default=0.1)
        assert out == {"entities": "E@0.1", "relation_extraction": "R@0.5", "event_extraction": "V@0.5", "topic": "C@0.1", "casualty": "S@0.5"}

    def test_unnamed_heads_fall_back_to_the_default(self):
        runs = {0.1: {"entities": "E@0.1", "casualty": "S@0.1"}, 0.5: {"entities": "E@0.5", "casualty": "S@0.5"}}
        assert splice_heads({}, runs, {"entity": 0.1}, default=0.5) == {"entities": "E@0.1", "casualty": "S@0.5"}


class TestShipping:
    def test_ship_thresholds_records_global_and_per_head_in_inference_defaults(self, tmp_path):
        import train as T
        (tmp_path / "config.json").write_text(json.dumps({"inference_defaults": {"threshold": 0.3, "chunk_size": 4096}}))
        T._ship_thresholds(tmp_path, 0.5, {"entity": 0.1, "relation": 0.5})
        cfg = json.loads((tmp_path / "config.json").read_text())
        assert cfg["inference_defaults"] == {"threshold": 0.5, "chunk_size": 4096, "per_head_thresholds": {"entity": 0.1, "relation": 0.5}}

    def test_per_head_pass_skips_the_decode_when_every_pick_is_the_global(self, tmp_path):
        import train as T
        ev = {"threshold": 0.3, "batch_size": 2, "stopwords": frozenset()}
        out = T._per_head_pass(tmp_path, ["no-such-file.jsonl"], ev, {"entity": 0.3, "event": 0.3}, {})
        assert out == {"thresholds": {"entity": 0.3, "event": 0.3}, "identical_to_global": True}

    def test_read_per_head_picks_from_sweep_file_or_empty(self, tmp_path):
        import train as T
        assert T._read_per_head_picks(tmp_path) == {}
        (tmp_path / "threshold_sweep.json").write_text(json.dumps({"chosen_threshold": 0.3, "per_head_thresholds": {"entity": 0.1}}))
        assert T._read_per_head_picks(tmp_path) == {"entity": 0.1}


class TestTrainingSplits:
    """training_splits returns the val/test training scores: train_only out by BARE name, event_files in, and the
    cross-split hygiene applied (split_hygiene.py: train yields to val and test, val yields to test). The tool used
    raw files and missed all three."""

    def _write(self, path, texts):
        path.write_text("".join(json.dumps({"input": t, "output": {"entities": {"PER": [t.split()[0]]}}}) + "\n" for t in texts),
                        encoding="utf-8")

    def _cfg(self, tmp_path):
        d = tmp_path / "data"; d.mkdir()
        self._write(d / "a.train.jsonl", ["Ann trains here."])
        self._write(d / "a.val.jsonl", ["Ann validates.", "Ann is also in test."])
        self._write(d / "a.test.jsonl", ["Ann tests.", "Ann is also in test."])
        self._write(d / "b.train.jsonl", ["Bob trains only."])
        self._write(d / "b.val.jsonl", ["Bob must not be scored."])
        self._write(d / "b.test.jsonl", ["Bob must not be tested."])
        self._write(d / "c.val.jsonl", ["Cy is an event-file val."])
        return {"data": {"corpora": [str(d / "a"), str(d / "b")], "train_only": ["b"],
                         "event_files": {"c": {"val": str(d / "c.val.jsonl")}}}, "training": {}}

    def test_val_and_test_match_training(self, tmp_path):
        import train as T
        val, test = T.training_splits(self._cfg(tmp_path), str(tmp_path / "c.yaml"))
        assert sorted(r["input"] for r in val) == ["Ann validates.", "Cy is an event-file val."]   # val yields to test
        assert sorted(r["input"] for r in test) == ["Ann is also in test.", "Ann tests."]
        assert {r["_corpus"] for r in val} == {"a", "c"}

    def test_sample_per_corpus_takes_n_from_each(self):
        import per_head_thresholds as P
        recs = [{"_corpus": c, "i": i} for c in ("x", "y") for i in range(5)]
        out = P.sample_per_corpus(recs, 2)
        assert [r["_corpus"] for r in out] == ["x", "x", "y", "y"]
        assert P.sample_per_corpus(recs, None) is recs
