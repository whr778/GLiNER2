"""Title_Snake label style (LABEL_STYLE_SPEC.md): one function for the generator, training and inference."""
import sys
from pathlib import Path

import pytest

from gliner2.inference.label_map import apply_label_map
from gliner2.inference.label_style import styler, title_snake

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "train"))
import train as T  # noqa: E402

STYLE = {"name": "title_snake", "acronyms": ["NORP"], "words": ["LAW"]}


@pytest.mark.parametrize("label,expected", [
    ("street address", "Street_Address"), ("streetAddress", "Street_Address"), ("STREET-ADDRESS", "Street_Address"),
    ("PlaceOfEmployment", "Place_Of_Employment"), ("PERSON", "Person"), ("GPE", "GPE"), ("PART-OF", "Part_Of"),
    ("Contact.ThreatenCoerce.Broadcast", "Contact.Threaten_Coerce.Broadcast"), ("HTTPServer", "Http_Server"), ("PER", "PER"), ("CAUSE OF DEATH", "Cause_Of_Death"),
    ("REGULATION OR LAW", "Regulation_Or_Law"), ("USED-FOR", "Used_For"), ("Part_OF", "Part_Of"),
    ("人名", "人名"), ("Military_Rank", "Military_Rank"),
])
def test_title_snake(label, expected):
    assert title_snake(label) == expected


def test_long_acronym_kept_only_when_listed():
    """Inference has no dictionary: an unlisted long acronym is title-cased. The generator lists every
    long acronym in the training labels, so only acronyms never trained on read this way."""
    assert title_snake("HTTPServer", acronyms={"HTTP"}) == "HTTP_Server"


def test_preserved_words_are_kept_verbatim():
    P = {"ATPases", "CoA"}
    assert title_snake("ATPases Associated with Diverse Cellular Activities", preserve=P) == \
        "ATPases_Associated_With_Diverse_Cellular_Activities"
    assert title_snake("Acetyl-CoA Carboxylase", preserve=P) == "Acetyl_CoA_Carboxylase"
    assert styler({"name": "title_snake", "preserve": ["CoA"]})("acetyl CoA") == "Acetyl_CoA"
    assert title_snake("Acetyl-CoA Carboxylase") == "Acetyl_Co_A_Carboxylase"
    assert title_snake("Cholesterol level in mg/dL", preserve={"mg/dL"}) == "Cholesterol_Level_In_mg/dL"
    assert title_snake("personnel.endposition.n/a", preserve={"mg/dL"}) == "Personnel.Endposition.N_A"


def test_listed_acronyms_and_short_words():
    assert title_snake("NORP") == "Norp" and title_snake("NORP", acronyms={"NORP"}) == "NORP"
    assert title_snake("LAW") == "LAW" and title_snake("LAW", short_words={"LAW"}) == "Law"


def test_idempotent_on_its_own_output():
    for label in ("street address", "HTTPServer", "Contact.ThreatenCoerce.Broadcast", "covid19 cases", "iPhone"):
        once = title_snake(label)
        assert title_snake(once) == once


def test_unknown_style_refused():
    with pytest.raises(ValueError):
        styler({"name": "kebab"})


class TestApplyLabelMap:
    MAP = {"entities": {"map": {"LOC": "Location"}}, "events": {"map": {"placeofemployment": "Place_Of_Employment"}},
           "structures": {"map": {"hotel": "Hotel"}}}
    SCHEMA = {"entities": ["LOC", "military rank"], "events": {"conflict.attack": ["placeofemployment", "victim"]},
              "structures": {"hotel": {"fields": [{"name": "hotel_name", "dtype": "str"}], "anchor": "hotel_name"}}}

    def test_style_reaches_unmapped_labels_and_structures(self):
        out, applied = apply_label_map(self.SCHEMA, self.MAP, STYLE)
        assert out["entities"] == ["Location", "Military_Rank"]
        assert out["events"] == {"Conflict.Attack": ["Place_Of_Employment", "Victim"]}
        assert out["structures"] == {"Hotel": {"fields": [{"name": "Hotel_Name", "dtype": "str"}], "anchor": "Hotel_Name"}}
        assert applied["entities"]["military rank"] == "Military_Rank"

    def test_without_style_nothing_changes_but_the_map(self):
        out, _ = apply_label_map(self.SCHEMA, self.MAP, None)
        assert out["entities"] == ["Location", "military rank"]
        assert out["events"] == {"conflict.attack": ["Place_Of_Employment", "victim"]}
        assert out["structures"] == self.SCHEMA["structures"]

    def test_built_schema_json_structures_record_metadata_and_descriptions_styled(self):
        from gliner2 import Schema

        built = (
            Schema()
            .structure("casualty_report", mode="natural", anchor="dead")
            .field("location", dtype="str", description="place")
            .field("dead", dtype="str", exclusive=True, description="count")
            .events({"hurricane": {"roles": ["dead"], "exclusive_roles": ["dead"]}})
            .classification("relevance", {"mass_casualty": "toll", "other": "not toll"})
            .entities({"Casualties and Losses": "losses"})
            .build()
        )
        out, applied = apply_label_map(built, None, STYLE)
        assert "Casualty_Report" in out["json_structures"][0]
        assert list(out["json_structures"][0]["Casualty_Report"]) == ["Location", "Dead"]
        assert out["json_descriptions"] == {"Casualty_Report": {"Location": "place", "Dead": "count"}}
        assert out["field_dtypes"] == {"Casualty_Report": {"Location": "str", "Dead": "str"}}
        assert out["record_metadata"]["Casualty_Report"] == {
            "mode": "natural",
            "anchor": "Dead",
            "fields": {"Dead": {"exclusive": True}},
        }
        assert out["record_metadata"]["Hurricane"] == {
            "mode": "natural",
            "anchor": "trigger",
            "fields": {"Dead": {"exclusive": True}},
        }
        assert out["classifications"][0]["labels"] == ["Mass_Casualty", "Other"]
        assert list(out["classifications"][0]["label_descriptions"]) == ["Mass_Casualty", "Other"]
        assert list(out["entities"]) == ["Casualties_And_Losses"]
        assert list(out["entity_descriptions"]) == ["Casualties_And_Losses"]
        assert applied["structures"]["casualty_report"] == "Casualty_Report"


class TestTraining:
    def test_labels_file_style_travels_and_styles_every_category(self, tmp_path):
        f = tmp_path / "v.yaml"
        f.write_text("style: {name: title_snake, acronyms: [], words: []}\nlabels:\n  entities: {map: {LOC: Location}}\n")
        cfg = T.load_labels_cfg({"labels_file": str(f)}, str(tmp_path / "c.yaml"))
        assert cfg["style"]["name"] == "title_snake"
        fns = T._category_fns(cfg)
        assert set(fns) == set(T.LABEL_CATEGORIES)
        assert fns["entities"]("LOC") == "Location" and fns["relations"]("part of") == "Part_Of"

        rec = {
            "input": "10 died in Florida.",
            "output": {
                "json_structures": [{"casualty_report": {"dead": ["10"], "location": ["Florida"]}}],
                "json_descriptions": {"casualty_report": {"dead": "count"}},
                "field_dtypes": {"casualty_report": {"dead": "str"}},
                "record_metadata": {
                    "casualty_report": {
                        "mode": "natural",
                        "anchor": "dead",
                        "fields": {"dead": {"exclusive": True}},
                    }
                },
            },
        }
        tr = T.transform_record(rec, fns)["output"]
        assert tr["record_metadata"]["Casualty_Report"]["anchor"] == "Dead"
        assert tr["record_metadata"]["Casualty_Report"]["fields"] == {"Dead": {"exclusive": True}}
        assert tr["json_descriptions"] == {"Casualty_Report": {"Dead": "count"}}
        assert tr["field_dtypes"] == {"Casualty_Report": {"Dead": "str"}}

    def test_no_style_keeps_map_only(self, tmp_path):
        f = tmp_path / "v.yaml"
        f.write_text("labels:\n  entities: {map: {LOC: Location}}\n")
        fns = T._category_fns(T.load_labels_cfg({"labels_file": str(f)}, str(tmp_path / "c.yaml")))
        assert set(fns) == {"entities"} and fns["entities"]("street address") == "street address"
