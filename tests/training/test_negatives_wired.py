"""The label-negative injector must reach the dataset on BOTH dataset paths.

2026-09-16 to 2026-09-19: `negatives=` was added to the non-sliding branch of
`_prepare_data` and not to the sliding-window branch. Every config combining
`negative_pools` with `sliding_window: true` therefore trained with no injected negatives
-- silently, because "label negatives ON" is logged by the LOADER and says nothing about
whether the dataset received the injector. It cost an A/B whose both arms were inert and
put an inert treatment into a published verdict.
"""
import types

import pytest
from transformers import AutoTokenizer

from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

POOLS = "tools/train/config/labels/negative_pools.json"
DATA = "data/docee.train.jsonl"


def _trainer(sliding: bool, pools=POOLS):
    cfg = TrainingConfig(
        negative_pools=pools,
        negative_labels_per_dim={"entities": 1, "events": 1} if pools else None,
        negative_label_seed=42,
        sliding_window=sliding,
        window_stride=3072,
        max_len=4096,
        validate_data=False,
        max_train_samples=60,
    )
    t = ExtractorTrainer.__new__(ExtractorTrainer)
    t.config = cfg
    t._guide_scores = lambda: None
    t.is_distributed = False
    t.processor = types.SimpleNamespace(
        tokenizer=AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    )
    return t


@pytest.mark.parametrize("sliding", [False, True])
def test_injector_reaches_the_dataset_on_both_paths(sliding):
    ds = ExtractorTrainer._prepare_data(_trainer(sliding), DATA, is_train=True)
    assert ds.negatives is not None, (
        f"sliding_window={sliding} built a dataset with no injector"
    )


def test_gate_refuses_a_configured_injector_that_did_not_land(monkeypatch):
    """The gate must FAIL when the wiring is missing -- this is the 2026-09-16 regression."""
    t = _trainer(sliding=True)
    ds = ExtractorTrainer._prepare_data(t, DATA, is_train=True)
    ds.negatives = None                      # simulate the branch that forgot to pass it
    with pytest.raises(SystemExit, match="carries NO injector"):
        t._wired(ds, is_train=True)


def test_gate_is_silent_when_negatives_are_not_configured():
    t = _trainer(sliding=True, pools=None)
    ds = ExtractorTrainer._prepare_data(t, DATA, is_train=True)
    assert t._wired(ds, is_train=True) is ds
