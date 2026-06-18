"""Unit tests for the token-NER converters (kaznerd/bc4chemd/bc5cdr, stockmark, finer-ord).

Loads the converter scripts by path and exercises their pure span/BIO logic --
no network / dataset download.
"""

import importlib.util
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "tools" / "data"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DATA / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


token_ner = _load("convert_hf_token_ner")
stockmark = _load("convert_stockmark_ner")
finer = _load("convert_finer_ord")


# --- generic BIO (B-/I- prefix) -------------------------------------------

def test_bio_runs_to_entities():
    text, ents = token_ner.bio_to_entities(
        ["John", "Smith", "works", "at", "Acme", "Inc"],
        ["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG"],
    )
    assert text == "John Smith works at Acme Inc"
    assert ents == {"PER": ["John Smith"], "ORG": ["Acme Inc"]}


def test_bio_adjacent_same_type_split_on_B():
    _, ents = token_ner.bio_to_entities(["A", "B"], ["B-X", "B-X"])
    assert ents == {"X": ["A", "B"]}  # two separate single-token X entities, deduped order-preserving


def test_bio_dedups_repeated_surface():
    _, ents = token_ner.bio_to_entities(["Acme", "and", "Acme"], ["B-ORG", "O", "B-ORG"])
    assert ents == {"ORG": ["Acme"]}


# --- finer-ord suffix scheme (PER_B / PER_I) ------------------------------

def test_finer_sentence_to_record():
    rec = finer.sentence_to_record(
        ["Barack", "Obama", "visited", "Kenya"],
        [1, 2, 0, 3],  # PER_B, PER_I, O, LOC_B
    )
    assert rec["input"] == "Barack Obama visited Kenya"
    assert rec["output"]["entities"] == {"PER": ["Barack Obama"], "LOC": ["Kenya"]}


def test_finer_none_when_all_O():
    assert finer.sentence_to_record(["just", "words"], [0, 0]) is None


# --- stockmark span entities ----------------------------------------------

def test_stockmark_groups_spans_by_type_verbatim():
    row = {"text": "Acme Corp partnered with Globex.",
           "entities": [{"name": "Acme Corp", "span": [0, 9], "type": "法人名"},
                        {"name": "Globex", "span": [25, 31], "type": "法人名"},
                        {"name": "NotInText", "span": [0, 0], "type": "人名"}]}
    rec = stockmark.convert_row(row)
    assert rec["output"]["entities"] == {"法人名": ["Acme Corp", "Globex"]}  # 人名 dropped (not verbatim)


def test_stockmark_none_without_entities():
    assert stockmark.convert_row({"text": "nothing here", "entities": []}) is None
