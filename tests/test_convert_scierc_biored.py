"""Unit tests for the SciERC and BioRED converters' doc -> GLiNER2 logic (no network)."""

import importlib.util
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "tools" / "data"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DATA / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scierc = _load("convert_scierc")
biored = _load("convert_biored")


def test_scierc_flattens_tokens_endinclusive_ner_and_relations():
    doc = {
        "sentences": [["English", "is", "a", "language"], ["It", "uses", "nouns"]],
        "ner": [[[0, 0, "Material"]], [[6, 6, "OtherScientificTerm"]]],  # global idx: nouns=6
        "relations": [[], [[6, 6, 0, 0, "USED-FOR"]]],
    }
    rec = scierc.convert_doc(doc)
    assert rec["input"] == "English is a language It uses nouns"
    assert rec["output"]["entities"] == {"Material": ["English"], "OtherScientificTerm": ["nouns"]}
    assert rec["output"]["relations"] == [{"USED-FOR": {"head": "nouns", "tail": "English"}}]


def test_scierc_none_when_empty():
    assert scierc.convert_doc({"sentences": [], "ner": [], "relations": []}) is None


def test_biored_entities_and_identifier_linked_relations():
    doc = {
        "passages": [{"text": "Gene X causes disease Y.", "annotations": [
            {"infons": {"type": "GeneOrGeneProduct", "identifier": "G1"}, "text": "Gene X"},
            {"infons": {"type": "DiseaseOrPhenotypicFeature", "identifier": "D1"}, "text": "disease Y"},
        ]}],
        "relations": [
            {"infons": {"type": "Positive_Correlation", "entity1": "G1", "entity2": "D1"}},
            {"infons": {"type": "Association", "entity1": "G1", "entity2": "UNKNOWN"}},  # dangling -> skipped
        ],
    }
    rec = biored.convert_doc(doc)
    assert rec["output"]["entities"] == {
        "GeneOrGeneProduct": ["Gene X"], "DiseaseOrPhenotypicFeature": ["disease Y"]}
    assert rec["output"]["relations"] == [
        {"Positive_Correlation": {"head": "Gene X", "tail": "disease Y"}}]


def test_biored_none_without_entities():
    assert biored.convert_doc({"passages": [{"text": "x", "annotations": []}], "relations": []}) is None
