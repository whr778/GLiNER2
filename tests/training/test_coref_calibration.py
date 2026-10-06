"""Coreference merge-threshold calibration: merge counting, the gated choice, and the config write."""
import json
import sys
from pathlib import Path

import pytest

from gliner2.training import coref_calibration as CC

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "train"))
import train as T  # noqa: E402

GOLD = [("Attack", {"attacked", "assault"}, set()), ("Attack", {"shelling"}, set()),
        ("Meet", {"meeting"}, set())]


def _row(merged=30, right=27, fused=0, f1=0.5, arg=0.3, control=None):
    r = {"merged": merged, "right": right, "fused": fused, "event_cluster_f1": f1, "event_cluster_argument_f1": arg,
         "merge_precision": right / merged if merged else None, "fusion": fused / merged if merged else 0.0}
    if control is not None:
        r["control"] = {"event_cluster_f1": control[0], "event_cluster_argument_f1": control[1],
                        "fusion": control[2] if len(control) > 2 else 0.0}
    return r


class TestMergeQuality:
    def test_right_fused_absorbed(self):
        pred = [("Attack", {"attacked", "assault"}, set()),      # one gold event: right
                ("Attack", {"assault", "shelling"}, set()),      # two gold events: fused
                ("Meet", {"meeting", "talks"}, set())]           # non-gold 'talks': absorbed
        c = CC.merge_quality(GOLD, pred, [2, 2, 2])
        assert (c["merged"], c["right"], c["fused"], c["absorbed"]) == (3, 1, 1, 1)

    def test_a_repeated_word_is_still_a_merge(self):
        c = CC.merge_quality(GOLD, [("Meet", {"meeting"}, set())], [2])
        assert (c["merged"], c["right"]) == (1, 1)

    def test_single_mention_is_not_a_merge(self):
        assert CC.merge_quality(GOLD, [("Meet", {"meeting"}, set())], [1])["merged"] == 0

    def test_hard_negative_pairs(self):
        apart = [("Attack", {"assault"}, set()), ("Attack", {"shelling"}, set())]
        together = [("Attack", {"assault", "shelling"}, set())]
        assert CC.merge_quality(GOLD, apart, [1, 1])["hardneg_merged"] == 0
        c = CC.merge_quality(GOLD, together, [2])
        assert (c["hardneg_pairs"], c["hardneg_merged"]) == (1, 1)


class TestChoose:
    def test_picks_the_best_eligible_by_cluster_f1(self):
        rows = {"off": _row(0, 0, f1=0.40), "0.7": _row(f1=0.45), "0.8": _row(f1=0.48)}
        assert CC.choose(rows)[0] == 0.8

    def test_too_few_merges_is_not_eligible_even_at_perfect_precision(self):
        rows = {"off": _row(0, 0, f1=0.40), "0.9": _row(merged=5, right=5, f1=0.45)}
        assert CC.choose(rows)[0] is None

    def test_low_precision_is_not_eligible(self):
        rows = {"off": _row(0, 0, f1=0.40), "0.5": _row(merged=30, right=20, f1=0.60)}
        assert CC.choose(rows)[0] is None

    def test_fusion_is_not_eligible(self):
        rows = {"off": _row(0, 0, f1=0.40), "0.6": _row(merged=40, right=34, fused=2, f1=0.60)}
        assert CC.choose(rows)[0] is None

    def test_worse_than_off_is_not_eligible(self):
        rows = {"off": _row(0, 0, f1=0.50), "0.8": _row(f1=0.49)}
        assert CC.choose(rows)[0] is None


    def test_argument_f1_below_off_is_not_eligible(self):
        rows = {"off": _row(0, 0, f1=0.40, arg=0.30), "0.8": _row(f1=0.45, arg=0.29)}
        assert CC.choose(rows)[0] is None

    def test_control_regression_is_not_eligible(self):
        rows = {"off": _row(0, 0, f1=0.40, control=(0.60, 0.20)), "0.8": _row(f1=0.45, control=(0.59, 0.20))}
        assert CC.choose(rows)[0] is None

    def test_control_argument_regression_is_not_eligible(self):
        rows = {"off": _row(0, 0, f1=0.40, control=(0.60, 0.20)), "0.8": _row(f1=0.45, control=(0.60, 0.19))}
        assert CC.choose(rows)[0] is None

    def test_control_unchanged_passes(self):
        rows = {"off": _row(0, 0, f1=0.40, control=(0.60, 0.20)), "0.8": _row(f1=0.45, control=(0.60, 0.20))}
        assert CC.choose(rows)[0] == 0.8


    def test_control_fusion_is_not_eligible_even_when_f1_rises(self):
        rows = {"off": _row(0, 0, f1=0.40, control=(0.54, 0.21)), "0.5": _row(f1=0.45, control=(0.55, 0.24, 0.48))}
        assert CC.choose(rows)[0] is None


class TestMultiTrigger:
    def test_counts_events_with_several_triggers(self):
        recs = [{"output": {"events": [{"triggers": ["a"]}, {"triggers": ["b", "c"]}]}}, {"output": {}}]
        assert CC.multi_trigger_events(recs) == 1


class TestWrite:
    def _ckpt(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"boundary_head": {"record_coref_link": True}}))
        return tmp_path

    def test_link_and_threshold_written(self, tmp_path):
        CC.write(self._ckpt(tmp_path), {"off": {}}, 0.85, "why", source=["v.jsonl"])
        bh = json.loads((tmp_path / "config.json").read_text())["boundary_head"]
        assert (bh["record_merge_coreferent"], bh["record_coref_link_threshold"]) == ("link", 0.85)
        assert json.loads((tmp_path / "coref_threshold_sweep.json").read_text())["chosen_threshold"] == 0.85

    def test_no_choice_writes_off(self, tmp_path):
        CC.write(self._ckpt(tmp_path), {"off": {}}, None, "why", source=[])
        assert json.loads((tmp_path / "config.json").read_text())["boundary_head"]["record_merge_coreferent"] == "off"


class TestStartupCheck:
    def test_calibration_without_the_link_is_refused(self):
        with pytest.raises(SystemExit, match="record_coref_link"):
            T.check_coref_calibration({"eval": {"coref_calibration": ["data/x"]}, "model": {"boundary_head": {}}})

    def test_no_calibration_is_a_no_op(self):
        T.check_coref_calibration({"model": {}})
