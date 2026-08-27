"""The stage-0 language gate: what it admits, and that it short-circuits the model.

The gate's relevance model reads English and Chinese. On Turkish it returns confident
verdicts with AUC 0.4733 -- below chance -- so the language check exists to reject that
input BEFORE inference rather than to score it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ekf_showcase"))

from language_gate import is_supported, language_of                      # noqa: E402

ENGLISH = ("At least 27 people died and 159 were injured when a residential building "
           "collapsed in the early hours of Tuesday, officials said.")
TURKISH = ("Kars-Ardahan kara yolunda kar nedeniyle mahsur kalan 110 kisi kurtarildi. "
           "Kars Valiligi, ekiplerin bolgede calisma yaptigini bildirdi.")
# Real row from data/gate2.test.jsonl: 24 characters, which lumi_language_id calls
# `und` at prob 0.42. Script, not the classifier, is what identifies it.
SHORT_CHINESE = "无锡一小吃店爆炸6死9伤 附近店面员工:头被震晕"


@pytest.mark.parametrize("text, expected", [(ENGLISH, "en"), (SHORT_CHINESE, "zh")])
def test_supported_languages_are_admitted(text, expected):
    supported, language, _ = is_supported(text)
    assert supported and language == expected


def test_turkish_is_rejected():
    supported, language, _ = is_supported(TURKISH)
    assert not supported and language == "tr"


def test_han_override_rescues_a_short_headline_the_classifier_misses():
    """The override fires on script, so it must not resurrect a non-CJK language."""
    assert language_of(SHORT_CHINESE)[0] == "zh"
    assert language_of(TURKISH)[0] == "tr"


def test_gate_does_not_run_the_model_on_unsupported_text():
    """Rejection happens before inference -- a wrong answer is not cheaper than none."""
    from run_pipeline import gate

    class ExplodingModel:
        def create_schema(self):
            raise AssertionError("schema built for unsupported text")

        def extract(self, *args, **kwargs):
            raise AssertionError("model ran on unsupported text")

    class SchemaOnlyModel(ExplodingModel):
        def create_schema(self):
            class Schema:
                def classification(self, *args, **kwargs):
                    return self
            return Schema()

    results = gate(SchemaOnlyModel(), [TURKISH], threshold=0.5, with_type=False)
    assert results[0]["relevant"] is False
    assert results[0]["relevance"] == "unsupported_language"
    assert results[0]["language"] == "tr"


def test_gate_still_scores_supported_text():
    """The companion to the test above: rejecting everything would also pass it."""
    from run_pipeline import gate

    class StubModel:
        def create_schema(self):
            class Schema:
                def classification(self, *args, **kwargs):
                    return self
            return Schema()

        def extract(self, text, schema, include_confidence=False):
            return {"relevance": {"label": "mass_casualty", "confidence": 0.99}}

    results = gate(StubModel(), [ENGLISH], threshold=0.5, with_type=False)
    assert results[0]["relevant"] is True
    assert results[0]["relevance"] == "mass_casualty"
    assert results[0]["language"] == "en"
