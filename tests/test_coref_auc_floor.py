"""The coref AUC floor scores the same subsets coref_readout does."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))

import coref_auc_floor as F

# (label, string_match, pronoun, cosine, link)
DOC = np.array([[1, 0, 1, 0.9, 0.1],
                [0, 0, 1, 0.1, 0.9],
                [1, 1, 0, 0.1, 0.9],
                [0, 0, 0, 0.9, 0.1]])


def test_subsets_filter_like_the_readout():
    assert F.subset_auc([DOC], F.COLS["cosine"], "pronoun") == 1.0
    assert F.subset_auc([DOC], F.COLS["link"], "pronoun") == 0.0
    assert F.subset_auc([DOC], F.COLS["cosine"], "different_surface") == 0.75
    assert F.subset_auc([DOC], F.COLS["link"], "all") == 0.5


def test_documents_pool_before_scoring():
    assert F.subset_auc([DOC[:2], DOC[2:]], F.COLS["link"], "all") == F.subset_auc([DOC], F.COLS["link"], "all")
