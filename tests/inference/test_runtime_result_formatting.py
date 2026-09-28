from __future__ import annotations

import json

import pytest

from gliner2.inference.runtime import format_results


def _round_trip(results):
    return json.loads(json.dumps(results))


@pytest.mark.parametrize("include_confidence", [False, True])
def test_single_label_tuple_and_json_pair_agree(include_confidence):
    results = {"sentiment": ("positive", 0.91)}
    from_tuple = format_results(
        results, include_confidence=include_confidence, classification_tasks=["sentiment"]
    )
    from_json = format_results(
        _round_trip(results),
        include_confidence=include_confidence,
        classification_tasks=["sentiment"],
    )
    expected = {"label": "positive", "confidence": 0.91} if include_confidence else "positive"
    assert from_tuple["sentiment"] == expected
    assert from_json["sentiment"] == expected


@pytest.mark.parametrize("include_confidence", [False, True])
def test_multi_label_pairs_survive_json(include_confidence):
    results = {"topics": [("tech", 0.9), ("finance", 0.8)]}
    formatted = format_results(
        _round_trip(results), include_confidence=include_confidence, classification_tasks=["topics"]
    )
    if include_confidence:
        assert formatted["topics"] == [
            {"label": "tech", "confidence": 0.9},
            {"label": "finance", "confidence": 0.8},
        ]
    else:
        assert formatted["topics"] == ["tech", "finance"]


@pytest.mark.parametrize("include_confidence", [False, True])
def test_two_label_list_is_not_decoded_as_label_score(include_confidence):
    formatted = format_results(
        {"topics": ["tech", "finance"]},
        include_confidence=include_confidence,
        classification_tasks=["topics"],
    )
    assert formatted["topics"] == ["tech", "finance"]


def test_single_label_list_without_score_is_unchanged():
    formatted = format_results(
        {"topics": ["tech"]}, include_confidence=True, classification_tasks=["topics"]
    )
    assert formatted["topics"] == ["tech"]


def test_bool_second_element_is_not_a_confidence():
    formatted = format_results(
        {"flags": ["urgent", True]}, include_confidence=True, classification_tasks=["flags"]
    )
    assert formatted["flags"] == ["urgent", True]


def test_non_classification_pairs_are_still_relations():
    formatted = format_results(
        {"founded": [("Sarah", "TechStart")]},
        include_confidence=False,
        requested_relations=["founded"],
    )
    assert formatted["relation_extraction"] == {"founded": [("Sarah", "TechStart")]}
    assert "founded" not in formatted


# Upstream's `format_results` has no event support. The merge of 2026-09-28 moved this
# function from a mixin method to module level, and taking upstream's version wholesale
# would have deleted `event_extraction` from every inference path -- silently, because
# events simply stop appearing rather than raising. These are the assertions that fail
# if the port is ever dropped again.

_EVENT = {"triggers": [("bombed", 0.9, 4, 10)], "arguments": {"Target": ["the market"]}}


def test_requested_events_land_under_event_extraction():
    formatted = format_results(
        {"Attack": [_EVENT]}, requested_events=["Attack", "Death"]
    )
    assert formatted["event_extraction"] == {"Attack": [_EVENT], "Death": []}
    assert "Attack" not in formatted, "an event must not also appear at the top level"


def test_event_shape_is_detected_without_being_requested():
    """The heuristic half: a list of dicts carrying both triggers and arguments."""
    formatted = format_results({"Attack": [_EVENT]})
    assert formatted["event_extraction"] == {"Attack": [_EVENT]}


def test_events_are_not_mistaken_for_relations():
    """Events are tested BEFORE the relation heuristic, which also matches dict lists."""
    formatted = format_results(
        {"Attack": [_EVENT], "founded": [("Sarah", "TechStart")]},
        requested_relations=["founded"],
        requested_events=["Attack"],
    )
    assert formatted["event_extraction"] == {"Attack": [_EVENT]}
    assert formatted["relation_extraction"] == {"founded": [("Sarah", "TechStart")]}


def test_no_events_means_no_event_key():
    formatted = format_results({"entities": [{"Person": ["Ada"]}]})
    assert "event_extraction" not in formatted
