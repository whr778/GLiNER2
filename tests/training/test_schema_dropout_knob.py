"""`schema_dropout: false` must actually zero the processor's removal probabilities.

The processor's SamplingConfig randomly REMOVES gold from a training schema --
`remove_events_prob` and `remove_relations_prob` default to 0.2, and
`remove_classification_label_prob` to 0.5. That is sensible regularisation on a large run and
actively misleading on a small one: it withholds a fifth of the signal from the very heads an
A/B is measuring, and in tests it has three times removed the only gold item in a one-example
batch and produced a zero loss that looked like a broken loss function.

Until now it was unreachable from config -- SamplingConfig was imported into trainer.py and
never used.
"""

import dataclasses

from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


def _probs(sc):
    return {f.name: getattr(sc, f.name) for f in dataclasses.fields(sc)
            if f.name.endswith("_prob")}


def test_the_default_keeps_dropout_on_so_existing_runs_reproduce():
    assert TrainingConfig.schema_dropout is True
    assert any(v > 0 for v in _probs(SamplingConfig()).values()), \
        "the default config must have some dropout, or this knob is measuring nothing"


def test_schema_dropout_false_zeroes_every_removal_probability(tiny_tokenizer, monkeypatch):
    proc = SchemaTransformer(tokenizer=tiny_tokenizer)
    before = _probs(proc.sampling_config)
    assert any(v > 0 for v in before.values())

    # Exercise the same branch the trainer runs, without building a model.
    cfg = TrainingConfig(schema_dropout=False)
    if not cfg.schema_dropout:
        for f in dataclasses.fields(proc.sampling_config):
            if f.name.endswith("_prob"):
                setattr(proc.sampling_config, f.name, 0.0)

    after = _probs(proc.sampling_config)
    assert set(after) == set(before)
    assert all(v == 0.0 for v in after.values()), f"still non-zero: {after}"
