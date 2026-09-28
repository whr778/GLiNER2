"""`true_label` is OPTIONAL on the inference path, and may be a bare string.

Two independent properties, and before the 2026-09-28 upstream merge each side of
the merge held exactly one of them:

ABSENT-SAFE  Upstream's 022cfc6 stopped the compiler emitting the mandatory
             `true_label: ["N/A"]` sentinel. That change lands in TWO places --
             `compiler.py` stops writing the key, and `processor.py` must stop
             reading it unconditionally. `compiler.py` merged cleanly and
             `processor.py` did not, so the read stayed, and every
             inference-path collate of a compiled schema raised
             `KeyError: 'true_label'`. It did not surface as a crash either:
             `error_policy="fallback"` caught it and substituted a dummy record,
             so the prompt silently became the string "dummy".

STRING-SAFE  A single-label task carries a bare string, a multi-label task a
             list. Upstream's `.get()` version iterates the value directly, so a
             bare "positive" maps its CHARACTERS and the label is destroyed.

A test for either one alone passes on the code that fails the other, which is
why they are asserted together here.
"""

from __future__ import annotations

import pytest

from gliner2.classification.compiler import compile_schema
from gliner2.classification.schema import ClassificationSchema


@pytest.fixture(scope="module")
def processor():
    try:
        from gliner2.processor import SchemaTransformer

        return SchemaTransformer(model_name="fastino/gliner2-base-v1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"real tokenizer unavailable: {exc!r}")


@pytest.fixture(scope="module")
def entry():
    compiled = compile_schema(
        ClassificationSchema().single("sentiment", ["positive", "negative"])
    )
    return compiled.model_schema["classifications"][0]


def test_the_compiler_no_longer_emits_the_sentinel(entry):
    """The premise. If this fails the other two prove nothing."""
    assert "true_label" not in entry


def test_a_compiled_schema_collates_without_true_label(processor, entry):
    schema = {"classifications": [dict(entry)]}
    record = processor._transform_record(
        {"text": "The service was excellent.", "schema": schema},
        max_len=512,
        build_targets=False,
    )
    assert len(record.schema_tokens_list) == 1


def test_a_bare_string_is_normalised_not_iterated(processor, entry):
    schema = {"classifications": [dict(entry, true_label="positive")]}
    processor._transform_record(
        {"text": "Great.", "schema": schema}, max_len=512, build_targets=False
    )
    assert schema["classifications"][0]["true_label"] == ["positive"]


def test_a_list_is_left_intact(processor, entry):
    schema = {"classifications": [dict(entry, true_label=["positive", "negative"])]}
    processor._transform_record(
        {"text": "Mixed.", "schema": schema}, max_len=512, build_targets=False
    )
    assert schema["classifications"][0]["true_label"] == ["positive", "negative"]
