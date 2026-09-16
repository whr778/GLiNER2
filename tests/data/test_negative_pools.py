"""Negative pools must be per-corpus, per-dimension, and post-label-map.

These three properties are the ones that make injected negatives SAFE. Getting any of them
wrong does not fail loudly -- it teaches the model that true labels are false, at loss
weight, for a whole run. See LABEL_NEGATIVES_PLAN.md section 3.
"""

import json

import pytest

from tools.data.build_negative_pools import scan


def _write(tmp_path, name, records):
    p = tmp_path / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                 encoding="utf-8")
    return p


def test_a_corpus_that_never_annotates_entities_gets_an_EMPTY_entity_pool(tmp_path):
    """The within-dimension rule, which is the difference between a negative and a lie.

    cmnee carries 8 event types and ZERO entity gold. If the pool reported entities for it,
    an injector would offer `Organization` to a document that does contain organisations and
    simply does not label them -- supervising a true positive as absent.
    """
    events_only = [{"input": "the army tested a missile",
                    "output": {"events": [{"event_type": "Experiment",
                                           "triggers": ["tested"],
                                           "arguments": [{"role": "Subject",
                                                          "entity": "the army"}]}]}}]
    info = scan(_write(tmp_path, "eventsonly", events_only), fns={}, limit=0)

    assert info["annotates"]["events"] is True
    assert info["annotates"]["entities"] is False, \
        "a corpus with no entity gold must not advertise an entity pool"
    assert info["entities"] == []
    assert info["events"] == {"Experiment": ["Subject"]}


def test_a_corpus_that_never_annotates_events_gets_an_EMPTY_event_pool(tmp_path):
    """The mirror case: biored has entities and relations but zero events."""
    ents_only = [{"input": "aspirin treats headache",
                  "output": {"entities": {"Chemical": ["aspirin"], "Disease": ["headache"]}}}]
    info = scan(_write(tmp_path, "entsonly", ents_only), fns={}, limit=0)

    assert info["annotates"]["entities"] is True
    assert info["annotates"]["events"] is False
    assert info["events"] == {}
    assert sorted(info["entities"]) == ["Chemical", "Disease"]


def test_the_pool_is_built_AFTER_the_label_map(tmp_path):
    """A pre-map alias in the pool is a direct contradiction of gold.

    With `LOC -> Location` in force, a pool holding BOTH would let an injector offer `LOC` as
    absent to a document whose gold says `Location` -- the same entity, asserted both ways.
    """
    from tools.train.train import _category_fns

    fns = _category_fns({"entities": {"map": {"LOC": "Location"}}})
    recs = [{"input": "he flew to Paris",
             "output": {"entities": {"LOC": ["Paris"]}}}]
    info = scan(_write(tmp_path, "aliased", recs), fns=fns, limit=0)

    assert info["entities"] == ["Location"], "the pool must hold canonical names only"
    assert "LOC" not in info["entities"]


def test_records_with_counts_are_reported_so_a_thin_dimension_is_visible(tmp_path):
    """`annotates` is a boolean; one stray record would flip it. The count is the check."""
    mixed = [{"input": "a", "output": {"entities": {"Person": ["a"]}}},
             {"input": "b", "output": {"entities": {"Person": ["b"]}}},
             {"input": "c", "output": {"relations": [{"works_at": {"head": "b", "tail": "c"}}]}}]
    info = scan(_write(tmp_path, "mixed", mixed), fns={}, limit=0)

    assert info["records_scanned"] == 3
    assert info["records_with"]["entities"] == 2
    assert info["records_with"]["relations"] == 1, \
        "a dimension carried by one record in three is annotated, but thinly -- say so"


def test_limit_is_honoured_so_a_huge_corpus_can_be_sampled(tmp_path):
    recs = [{"input": str(i), "output": {"entities": {f"T{i}": [str(i)]}}} for i in range(50)]
    info = scan(_write(tmp_path, "big", recs), fns={}, limit=10)
    assert info["records_scanned"] == 10
    assert len(info["entities"]) == 10
