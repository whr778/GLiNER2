"""The stage-0 language gate: what it admits, and that it short-circuits the model.

Turkish moved from rejected to supported on 2026-09-04, paired with switching
`run_pipeline.py`'s default `--gate-model` to `gate2-mmbert-tr` (Turkish AUC 0.4980,
chance, -> 0.8105). The two changes are coupled: this allowlist governs which languages
reach the gate model at all, not what any given checkpoint does with them once there.
Passing an older checkpoint (`gate2-mmbert-v2`, AUC 0.4733 on Turkish -- below chance)
here would let Turkish back through to a model still blind to it.
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
# Not en/zh/tr -- exercises the still-live reject path. Content mirrors ENGLISH so a
# failure here is about language support, not about this text being irrelevant.
FRENCH = ("Au moins 27 personnes sont mortes et 159 ont ete blessees lorsqu'un immeuble "
         "residentiel s'est effondre tot mardi matin, ont annonce les autorites.")


@pytest.mark.parametrize("text, expected",
                         [(ENGLISH, "en"), (SHORT_CHINESE, "zh"), (TURKISH, "tr")])
def test_supported_languages_are_admitted(text, expected):
    supported, language, _ = is_supported(text)
    assert supported and language == expected


def test_unsupported_language_is_rejected():
    supported, language, _ = is_supported(FRENCH)
    assert not supported and language not in ("en", "zh", "tr")


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

    results = gate(SchemaOnlyModel(), [FRENCH], threshold=0.5, with_type=False)
    assert results[0]["relevant"] is False
    assert results[0]["relevance"] == "unsupported_language"
    assert results[0]["language"] not in ("en", "zh", "tr")


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
