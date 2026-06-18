"""Unit tests for the KLUE converter's parsing logic (no network)."""

import importlib.util
from pathlib import Path

_CONV = Path(__file__).resolve().parents[1] / "tools" / "data" / "convert_klue.py"


def _load():
    spec = importlib.util.spec_from_file_location("convert_klue", _CONV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


klue = _load()

NER_TSV = "\n".join([
    "## column header",
    "## klue-ner-v1_dev_0\t<경찰:OG>은",
    "경\tB-OG",
    "찰\tI-OG",
    "은\tO",
    "",
    "## klue-ner-v1_dev_1\tonly O",
    "x\tO",
    "y\tO",
])

RE_JSON = (
    '[{"sentence": "A works at B", '
    '"subject_entity": {"word": "A", "type": "PER"}, '
    '"object_entity": {"word": "B", "type": "ORG"}, "label": "org:member"}, '
    '{"sentence": "X met Y", '
    '"subject_entity": {"word": "X", "type": "PER"}, '
    '"object_entity": {"word": "Y", "type": "PER"}, "label": "no_relation"}]'
)


def test_ner_char_bio_concatenates_without_spaces():
    out = list(klue.iter_ner(NER_TSV))
    # second sentence is all-O -> empty entities
    (text0, ents0), (text1, ents1) = out
    assert text0 == "경찰은"               # chars joined with no separator
    assert ents0 == {"OG": ["경찰"]}       # B-OG/I-OG run
    assert ents1 == {}


def test_re_emits_entities_and_skips_no_relation():
    recs = list(klue.iter_re(RE_JSON))
    (s0, o0), (s1, o1) = recs
    assert o0["entities"] == {"PER": ["A"], "ORG": ["B"]}
    assert o0["relations"] == [{"org:member": {"head": "A", "tail": "B"}}]
    # no_relation -> entities only, no relation key
    assert o1["entities"] == {"PER": ["X", "Y"]}
    assert "relations" not in o1
