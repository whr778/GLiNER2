"""The cut-off sweep's scorer: pair P/R/F1 against gold, and the string-match baseline."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))

import coref_threshold as T

LOGITS = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-1.0, -1.0, 0.0]])
DOC = (LOGITS, [0, 0, 1], ["ann", "she", "bea"])


def test_perfect_clustering_scores_one():
    r = T.pair_prf([DOC], 1.0)
    assert r["all"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert r["pronoun"]["f1"] == 1.0


def test_too_low_a_threshold_merges_and_loses_precision():
    assert T.pair_prf([DOC], -5.0)["all"]["precision"] == round(1 / 3, 4)


def test_string_match_baseline_cannot_link_a_pronoun():
    assert T.pair_prf([DOC], None)["all"]["recall"] == 0.0
