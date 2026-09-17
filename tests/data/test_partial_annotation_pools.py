"""A PARTIAL corpus contributes positives to a dimension but never negatives.

`build_negative_pools.py` infers annotation from the presence of gold:

    "annotates": {d: bool(seen[d]) for d in DIMENSIONS}

A corpus DERIVED from another dimension -- event argument spans reframed as typed entities --
carries entity gold for the spans it derived and nothing for the entities it never looked at.
Inferred naively it qualifies as an entity annotator and draws entity negatives against
unlabelled gold, teaching the model that real entities are not entities, at loss weight.

A negatives pool has no natural correctness signal, so that damage is silent. These tests are
the signal.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mod():
    """Load the script by path; it is a tool, not an importable package member."""
    spec = importlib.util.spec_from_file_location(
        "_bnp", ROOT / "tools" / "data" / "build_negative_pools.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_declaration_is_parsed_and_normalised(mod):
    cfg = {"data": {"partial_annotation": {"derived": "entities",
                                           "other": ["entities", "events"]}}}
    got = mod._partial_annotation(cfg)
    assert got == {"derived": {"entities"}, "other": {"entities", "events"}}


def test_absent_declaration_is_empty(mod):
    assert mod._partial_annotation({}) == {}
    assert mod._partial_annotation({"data": {}}) == {}


def test_unknown_dimension_is_refused(mod):
    """A typo'd dimension must not silently protect nothing."""
    with pytest.raises(SystemExit, match="unknown dimension"):
        mod._partial_annotation({"data": {"partial_annotation": {"x": ["entites"]}}})


def test_partial_only_turns_annotation_off_never_on(mod):
    """The flag is safe in one direction only.

    Turning annotation ON for a dimension a corpus does not annotate would invent a pool from
    nothing; the flag must be incapable of it.
    """
    info = {"annotates": {"entities": False, "events": True,
                          "relations": False, "structures": False}}
    for dim in ("entities",):
        if info["annotates"][dim]:
            info["annotates"][dim] = False
            info.setdefault("partial", []).append(dim)
    assert info["annotates"]["entities"] is False
    assert "partial" not in info, "a dimension already off must not be marked partial"


def test_a_partial_corpus_vetoes_that_dimension_for_its_records(tmp_path):
    """End to end through the consumer: the veto is what actually protects the model."""
    from gliner2.training.negatives import NegativeLabels

    pools = {
        "derived": {
            "records_scanned": 10,
            # Entity gold IS present -- that is exactly the trap.
            "annotates": {"entities": False, "events": True,
                          "relations": False, "structures": False},
            "partial": ["entities"],
            "entities": ["Subject", "Equipment", "Date"],
            "events": {"Experiment": ["Subject", "Date"]},
            "relations": [], "structures": {},
        }
    }
    path = tmp_path / "pools.json"
    path.write_text(json.dumps({"config": "x", "pools": pools}), encoding="utf-8")

    neg = NegativeLabels.load(str(path), {"entities": 5})
    assert neg._usable_pool("entities", ["derived"]) == set(), (
        "a PARTIAL corpus must offer no entity negatives, even though it carries entity gold"
    )
    # Its other dimensions are untouched.
    assert neg._usable_pool("events", ["derived"]) == {"Experiment"}


def test_without_the_flag_the_same_corpus_would_offer_negatives(tmp_path):
    """The counterfactual, so the test above cannot pass for the wrong reason."""
    from gliner2.training.negatives import NegativeLabels

    pools = {
        "derived": {
            "records_scanned": 10,
            "annotates": {"entities": True, "events": True,      # <- inferred, unflagged
                          "relations": False, "structures": False},
            "entities": ["Subject", "Equipment", "Date"],
            "events": {"Experiment": ["Subject", "Date"]},
            "relations": [], "structures": {},
        }
    }
    path = tmp_path / "pools.json"
    path.write_text(json.dumps({"config": "x", "pools": pools}), encoding="utf-8")

    neg = NegativeLabels.load(str(path), {"entities": 5})
    assert neg._usable_pool("entities", ["derived"]) == {"Subject", "Equipment", "Date"}, (
        "unflagged, this corpus DOES offer entity negatives -- which is the hazard"
    )
