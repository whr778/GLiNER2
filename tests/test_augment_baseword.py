"""Unit tests for the base-word (lemmatized) duplication pass.

The whole risk in this augmentation is silent: boundary collation locates each gold surface
in the text and SKIPS what it cannot find, so a misaligned label does not raise -- it just
removes supervision and reads as "augmentation didn't help". These tests pin the alignment
contract directly, without a lemmatizer dependency.
"""

import importlib.util
import sys
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "tools" / "data"
_AUG = _DATA / "augment_baseword.py"


def _load():
    sys.path.insert(0, str(_DATA))          # the script imports `_split` as a sibling
    spec = importlib.util.spec_from_file_location("augment_baseword", _AUG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _labels(out):
    for ev in out.get("events") or []:
        for t in ev.get("triggers") or []:
            yield t
        for a in ev.get("arguments") or []:
            yield a["entity"]
    for block in ([out["entities"]] if isinstance(out.get("entities"), dict) else []):
        for values in block.values():
            for v in values:
                yield v


def test_every_label_stays_verbatim_in_the_rewritten_text():
    record = {
        "input": "Rebels attacked the convoys near Aleppo , killing three soldiers .",
        "output": {"events": [{"event_type": "life.die", "triggers": ["killing"],
                               "arguments": [{"role": "victim", "entity": "three soldiers"},
                                             {"role": "place", "entity": "Aleppo"}]}]},
    }
    out = mod.augment(record, mod.mock_lemma, "en")

    assert out["input"] != record["input"], "nothing was rewritten -- the test is vacuous"
    for label in _labels(out["output"]):
        assert label in out["input"], f"{label!r} is no longer verbatim"


def test_labels_are_rewritten_too_not_just_the_text():
    """The point of the pass: base-word form on BOTH sides, not text-only."""
    record = {
        "input": "Officials confirmed three soldiers died .",
        "output": {"entities": {"victim": ["three soldiers"]}},
    }
    out = mod.augment(record, mod.mock_lemma, "en")

    assert out["output"]["entities"]["victim"] == ["three soldier"]


def test_a_token_with_no_lemma_keeps_its_surface_form():
    """Real lemmatizers return nothing for plenty of tokens; substituting a guess there is
    how alignment breaks, so the fallback is the surface word."""
    empty = lambda word, lang: ""
    record = {"input": "Aleppo fell .", "output": {"entities": {"place": ["Aleppo"]}}}

    out = mod.augment(record, empty, "en")

    assert out["input"] == "Aleppo fell ."
    assert out["output"]["entities"]["place"] == ["Aleppo"]


def test_a_label_covering_part_of_a_token_is_refused_not_mangled():
    """`Armenian` inside `Armenians` cannot survive lemmatization of its host token, so the
    record is dropped whole rather than emitted with a broken span."""
    record = {
        "input": "Armenians reported shelling .",
        "output": {"entities": {"group": ["Armenian"]}},
    }

    assert mod.augment(record, mod.mock_lemma, "en") is None


def test_a_label_that_lemmatization_merges_with_other_text_is_refused():
    """The mirror of the missing-surface failure, and invisible to the same checks.

    Lemmatization COLLAPSES surface forms, so a label can start matching positions that
    were never annotated -- gold `guns` occurs once, but as `gun` it also matches the
    unrelated `gun` later in the text, and collation then builds two gold mentions where
    one was annotated. Measured on RAMS with simplemma before this guard existed: gold
    inflated by 1,085 mentions on 31,773 (3.4%), across 718 of 6,680 augmented records.
    Nothing raises and `missing_surface_counts()` stays at zero, because these are
    invented positives rather than lost ones.
    """
    record = {
        "input": "Soldiers seized guns . The gun was later destroyed .",
        "output": {"entities": {"weapon": ["guns"]}},
    }

    assert mod.augment(record, mod.mock_lemma, "en") is None


def test_occurrence_counting_uses_the_collators_tokenization():
    """`str.split()` is the wrong ruler and silently under-counts.

    `WhitespaceTokenSplitter` splits trailing punctuation into its own token and
    lower-cases, so `they,` contains the token `they` while whitespace splitting sees only
    `they,`. Counting the wrong way passed records that then gained phantom gold.
    """
    text = "they ran and they, too, fled."

    assert text.split().count("they") == 1          # the wrong ruler: sees only `they,`
    assert mod.token_count(text, "they") == 2       # what collation actually matches


def test_occurrence_counting_handles_cjk():
    """The splitter emits one token per CJK character, so counting must follow it.

    The mix is multilingual (CMNEE/DuEE/ChFinAnn Chinese, KLUE Korean), and a whitespace
    ruler would see `三人死亡,五人死亡` as a single token and count every label zero times.
    """
    assert mod.token_count("三人死亡,五人死亡", "死亡") == 2
    assert mod.token_count("三人在北京死亡", "北京") == 1


def test_the_original_record_is_never_mutated():
    record = {"input": "Rebels attacked the convoys .",
              "output": {"entities": {"target": ["the convoys"]}}}

    mod.augment(record, mod.mock_lemma, "en")

    assert record["output"]["entities"]["target"] == ["the convoys"]
