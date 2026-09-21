"""Schema names must not silently overwrite each other in the extraction output.

Extraction output is a FLAT dict. Three keys are fixed containers -- `entities`,
`relation_extraction`, `event_extraction` -- while classification TASK names and structure
NAMES are hoisted to the top level beside them. Two of those are user-chosen, so they compete
with the containers and with each other.

MEASURED ON A REAL MODEL BEFORE THIS GUARD EXISTED, and every case lost data silently:
  - a classification task named `entities` replaced the whole entity block with a label
    string, so `out["entities"]` was `'no'`;
  - a structure sharing a name with a classification task was dropped entirely;
  - nothing raised, at schema-build time or at extraction.

Eval inherits the same namespace (it reads a classification from `pred[task]`), so a
collision corrupts two heads at once. 130 corpora were scanned before adding this: none
collide, so the guard breaks nothing that exists.
"""

import pytest

from gliner2.inference.schema import Schema


@pytest.mark.parametrize("reserved", ["entities", "relation_extraction", "event_extraction"])
def test_classification_cannot_take_a_reserved_container_name(reserved):
    with pytest.raises(ValueError, match="reserved extraction output key"):
        Schema().classification(reserved, ["a", "b"])


@pytest.mark.parametrize("reserved", ["entities", "relation_extraction", "event_extraction"])
def test_structure_cannot_take_a_reserved_container_name(reserved):
    with pytest.raises(ValueError, match="reserved extraction output key"):
        Schema().structure(reserved).field("x")


def test_structure_cannot_shadow_an_existing_task():
    with pytest.raises(ValueError, match="already used by a classification task"):
        Schema().classification("report", ["a", "b"]).structure("report").field("x")


def test_task_cannot_shadow_an_existing_structure():
    """The reverse order must be caught too -- the builder is auto-finished first."""
    with pytest.raises(ValueError, match="already used by a structure"):
        Schema().structure("report").field("x").classification("report", ["a", "b"])


def test_from_dict_does_not_bypass_the_guard():
    """`from_dict` rebuilds through the same builders, so a stored schema cannot smuggle
    a collision past it."""
    with pytest.raises(ValueError, match="reserved extraction output key"):
        Schema.from_dict({
            "entities": {"person": ""},
            "classifications": [{"task": "entities", "labels": ["a", "b"]}],
        })


def test_an_ordinary_mixed_schema_is_unaffected():
    s = (Schema().entities(["person", "organization"])
         .classification("sentiment", ["positive", "negative"])
         .relations(["works_for"])
         .events({"Attack": ["Attacker"]})
         .structure("contact").field("email"))
    assert s is not None


def test_names_that_merely_resemble_a_container_are_allowed():
    """`events` and `relations` are SCHEMA keys, not OUTPUT keys -- only the three output
    container names are reserved, and over-reserving would reject legitimate task names."""
    Schema().classification("events", ["a", "b"])
    Schema().classification("relations", ["a", "b"])
    Schema().structure("events").field("x")


def test_distinct_names_still_coexist():
    Schema().structure("a").field("x").structure("b").field("y")
    Schema().classification("t1", ["a", "b"]).classification("t2", ["c", "d"])
