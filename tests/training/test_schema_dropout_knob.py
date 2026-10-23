"""`training.schema_dropout` must really change the processor's removal probabilities.

The processor's SamplingConfig randomly REMOVES gold from a training schema: `remove_relations_prob` 0.2 per gold
relation INSTANCE (a dropped instance flips to a negative when proposed), `remove_events_prob` 0.2 per event TYPE,
`remove_classification_label_prob` 0.5. `false` zeroes every `*_prob`; a dict sets only the named ones, so the
relation drop can be switched off while the event and entity augmentation stays. The tests call the trainer's own
function (an earlier version re-implemented the branch and could not fail on the trainer's code).
"""

import dataclasses

import pytest

from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.training.trainer import TrainingConfig, apply_schema_dropout


def _probs(sc):
    return {f.name: getattr(sc, f.name) for f in dataclasses.fields(sc) if f.name.endswith("_prob")}


def test_the_default_keeps_dropout_on_so_existing_runs_reproduce():
    assert TrainingConfig.schema_dropout is True
    assert any(v > 0 for v in _probs(SamplingConfig()).values()), \
        "the default config must have some dropout, or this knob is measuring nothing"
    sc = SamplingConfig()
    assert apply_schema_dropout(sc, True) == {} and _probs(sc) == _probs(SamplingConfig())


def test_schema_dropout_false_zeroes_every_removal_probability(tiny_tokenizer):
    proc = SchemaTransformer(tokenizer=tiny_tokenizer)
    before = _probs(proc.sampling_config)
    assert any(v > 0 for v in before.values())
    changed = apply_schema_dropout(proc.sampling_config, TrainingConfig(schema_dropout=False).schema_dropout)
    after = _probs(proc.sampling_config)
    assert set(after) == set(before)
    assert all(v == 0.0 for v in after.values()), f"still non-zero: {after}"
    assert set(changed) == {k for k, v in before.items() if v}


def test_schema_dropout_dict_sets_only_the_named_probabilities():
    sc = SamplingConfig()
    before = _probs(sc)
    changed = apply_schema_dropout(sc, {"remove_relations_prob": 0.0, "swap_head_tail_prob": 0.0})
    assert changed == {"remove_relations_prob": 0.0, "swap_head_tail_prob": 0.0}
    after = _probs(sc)
    assert after["remove_relations_prob"] == 0.0 and after["swap_head_tail_prob"] == 0.0
    untouched = {k: v for k, v in before.items() if k not in changed}
    assert {k: after[k] for k in untouched} == untouched           # events, structures, entities unchanged
    assert after["remove_events_prob"] == 0.2


def test_schema_dropout_dict_refuses_an_unknown_name():
    with pytest.raises(ValueError, match="remove_relation_prob"):
        apply_schema_dropout(SamplingConfig(), {"remove_relation_prob": 0.0})      # typo: singular
