"""The sliding-window `max_train_samples` cap is a seeded RANDOM draw, not a file-order prefix.

A prefix cap on eb18's mix was 100% sentence_rex (no events): a fast event A/B would have
trained every arm on zero event data. The draw's seed is `max_samples_seed`, independent of
`seed`, so A/B arms that differ in seed still train on the same subset.
"""
import tempfile

from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model

RECORDS = ([{"input": f"first corpus doc {i} .", "output": {"entities": {"thing": ["doc"]}}} for i in range(50)]
           + [{"input": f"second corpus text {i} .", "output": {"entities": {"thing": ["text"]}}} for i in range(50)])


def _texts(cap, seed, max_samples_seed=42):
    tc = TrainingConfig(output_dir=tempfile.mkdtemp(), sliding_window=True, max_len=512, window_stride=512,
                        max_train_samples=cap, bf16=False, fp16=False, seed=seed,
                        max_samples_seed=max_samples_seed, validate_data=False)
    ds = ExtractorTrainer(build_tiny_boundary_model(), tc)._prepare_data(list(RECORDS), True)
    return {r["input"] if isinstance(r, dict) else r[0] for r in ds.data}


def test_cap_draws_from_the_whole_mix_not_a_prefix():
    texts = _texts(20, seed=42)
    assert len(texts) == 20
    assert any(t.startswith("second") for t in texts), "the cap kept a file-order prefix"


def test_cap_subset_does_not_depend_on_the_training_seed():
    assert _texts(20, seed=42) == _texts(20, seed=43)
    assert _texts(20, seed=42) != _texts(20, seed=42, max_samples_seed=7)


def test_uncapped_keeps_every_record():
    assert len(_texts(-1, seed=42)) == len(RECORDS)
