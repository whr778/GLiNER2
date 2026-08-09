"""Evaluation must not abort on an entity surface it cannot align.

``collate_fn_inference`` never forwarded ``on_missing_surface``, so the parameter
sat at its ``"raise"`` default with no caller able to reach it. Training was
tolerant (``error_policy: skip`` maps to ``on_missing_surface="skip"``), but the
epoch-end evaluation was not -- so a single unalignable mention anywhere in the
val set killed the run at the first epoch boundary, hours in, after the training
half had already handled the identical case.

That is exactly how the joint-boundary 10K base died:

    ValueError: entity 'DiseaseOrPhenotypicFeature' was not found in sample 7

Note ``error_policy`` (malformed records, in ``_collate_batch``) and
``on_missing_surface`` (surface alignment) are different knobs; the bug was
assuming the first covered the second.
"""

from __future__ import annotations

import pytest

from gliner2.processor import SchemaTransformer

TEXT = "John works at Apple."
# "Nonexistent" never appears in TEXT, so its span cannot be located.
UNALIGNABLE = (TEXT, {"entities": {"person": ["Nonexistent"], "company": ["Apple"]}})


def _processor(tiny_tokenizer):
    return SchemaTransformer(tokenizer=tiny_tokenizer)


def test_eval_collator_tolerates_unalignable_surface(tiny_tokenizer):
    """The default eval path must survive it -- this is the regression."""
    proc = _processor(tiny_tokenizer)
    batch = proc.collate_fn_inference(
        [UNALIGNABLE], architecture="boundary", build_targets=True
    )
    assert len(batch) == 1


def test_eval_collator_still_strict_when_asked(tiny_tokenizer):
    """Tolerance must be a default, not a removal of the strict path."""
    proc = _processor(tiny_tokenizer)
    with pytest.raises(ValueError, match="was not found"):
        proc.collate_fn_inference(
            [UNALIGNABLE], architecture="boundary", build_targets=True,
            error_policy="raise",
        )


def test_training_collator_keeps_both_behaviours(tiny_tokenizer):
    """Guard the training side against a fix that only moved the problem."""
    proc = _processor(tiny_tokenizer)
    assert len(proc.collate_fn_train(
        [UNALIGNABLE], architecture="boundary", error_policy="skip")) == 1
    with pytest.raises(ValueError, match="was not found"):
        proc.collate_fn_train(
            [UNALIGNABLE], architecture="boundary", error_policy="raise")


def test_alignable_gold_survives_the_skip(tiny_tokenizer):
    """Skipping the bad mention must not discard the record's good gold too."""
    proc = _processor(tiny_tokenizer)
    batch = proc.collate_fn_inference(
        [UNALIGNABLE], architecture="boundary", build_targets=True
    )
    assert batch.targets is not None, "record kept but all supervision dropped"
