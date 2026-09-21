"""The merge must attach types where the pipeline can SEE them, and never invent a type.

Two things this guards, both found by tracing rather than by reasoning:
  * the dataset's `__getitem__` returns only `(text, schema)`, so a top-level key is dropped
    at the first hop -- the types must live INSIDE the schema;
  * naive containment matching invents types: '印度' (Location) sits inside '印度海军'
    (Indian Navy, an Organization), and a wrong type is worse than no type because it puts
    the training margin on the wrong candidate.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "_merge_test", ROOT / "tools" / "data" / "merge_entity_types.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def m():
    return _mod()


def test_exact_match_wins(m):
    surf = {"北京": {"Location"}, "北京大学": {"Organization"}}
    assert m.lookup("北京", surf, 0.85) == {"Location"}


def test_sub_span_of_an_annotated_entity_is_typed(m):
    """'23' inside '23人' -- the argument is part of the annotated entity."""
    assert m.lookup("23", {"23人": {"Quantity"}}, 0.85) == {"Quantity"}


def test_entity_covering_most_of_the_argument_is_typed(m):
    """'B-2轰炸机' covers most of 'B-2轰炸机队'."""
    assert m.lookup("B-2轰炸机队", {"B-2轰炸机": {"Product"}}, 0.85) == {"Product"}


def test_a_modifier_does_NOT_type_the_whole_phrase(m):
    """'印度'(Location) is 50% of '印度海军'(Organization). Accepting it invents a type."""
    assert m.lookup("印度海军", {"印度": {"Location"}}, 0.85) is None


def test_single_character_surfaces_are_refused(m):
    """'人' would otherwise match every '20人'/'21人' in the document."""
    assert m.lookup("人", {"20人": {"Quantity"}, "21人": {"Quantity"}}, 0.85) is None


def test_no_match_returns_none_rather_than_a_guess(m):
    assert m.lookup("完全没有关系", {"北京": {"Location"}}, 0.85) is None


@pytest.mark.parametrize("corpus", ["cmnee_typed", "duee_typed", "casie_typed"])
def test_built_corpus_carries_types_inside_the_schema(corpus):
    import json
    p = ROOT / "data" / f"{corpus}.train.jsonl"
    if not p.is_file():
        pytest.skip(f"{corpus} not built")
    with open(p, encoding="utf-8") as fh:
        rec = json.loads(fh.readline())
    schema = rec.get("output") or {}
    assert "entity_types" in schema, "types must be INSIDE the schema to survive __getitem__"
    assert "entity_types" not in rec, "no stale top-level copy"
    assert schema["entity_types"], "the first record should carry types"


def test_merge_does_not_add_entity_supervision():
    """The purchased gold is not exhaustive; adding an `entities` block would teach the
    entity head that every unannotated entity is absent."""
    import json
    p = ROOT / "data" / "cmnee_typed.train.jsonl"
    if not p.is_file():
        pytest.skip("not built")
    with open(p, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 200:
                break
            schema = json.loads(line).get("output") or {}
            assert "entities" not in schema, "cmnee must not gain entity supervision"
