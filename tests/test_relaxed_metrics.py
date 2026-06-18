"""Tests for strict vs relaxed (partial-overlap) metric matching.

Covers the overlap rule, the two-pass matcher's strict-dominance guarantee, and
end-to-end compute_metrics via a fake model for entities, relations, and the
trigger-link-dropping behaviour of relaxed events.
"""

from collections import Counter

from gliner2.training.metrics import (
    _items_argument,
    _items_trigger,
    _match_relaxed,
    _overlap,
    _pred_event_argument_set,
    _pred_event_trigger_set,
    _pred_relation_set,
    compute_metrics,
)


def test_pred_event_sets_handle_default_and_dict_formats():
    """The engine emits trigger/entity as plain strings by default and as
    {'text': ...} dicts under include_spans/confidence; both must score."""
    default = {"event_extraction": {"Conflict.Attack": [
        {"trigger": "bombed", "arguments": [{"role": "Target", "entity": "tower"}]}]}}
    nested = {"event_extraction": {"Conflict.Attack": [
        {"trigger": {"text": "bombed"}, "arguments": [{"role": "Target", "entity": {"text": "tower"}}]}]}}
    for pred in (default, nested):
        assert _pred_event_trigger_set(pred) == {("Conflict.Attack", "bombed")}
        assert _pred_event_argument_set(pred) == {("Conflict.Attack", "Target", "tower", "bombed")}


def test_pred_relation_set_handles_all_engine_output_formats():
    """The inference engine emits relations as (head, tail) tuples by default and
    as nested {'head': {'text': ...}} dicts under include_spans/confidence; the
    scorer must read all of them (a tuple-only run previously scored 0.0)."""
    expected = {("country", "Stockholm", "Sweden")}
    assert _pred_relation_set({"relation_extraction": {"country": [("Stockholm", "Sweden")]}}) == expected
    assert _pred_relation_set({"relation_extraction": {"country": [["Stockholm", "Sweden"]]}}) == expected
    assert _pred_relation_set(
        {"relation_extraction": {"country": [{"head": {"text": "Stockholm"}, "tail": {"text": "Sweden"}}]}}
    ) == expected
    assert _pred_relation_set({"relation_extraction": {"country": [{"head": "Stockholm", "tail": "Sweden"}]}}) == expected


# --- overlap rule ----------------------------------------------------------

def test_overlap_substring_and_token_and_norm():
    assert _overlap("New York City", "New York")       # substring containment
    assert _overlap("Bank of America", "America")       # shared content token
    assert _overlap("USA", "usa")                        # normalized-equal
    assert _overlap("  the   White House ", "White House")


def test_overlap_rejects_stopword_only_and_disjoint():
    assert not _overlap("the president", "the bombing")  # only 'the' shared
    assert not _overlap("Paris", "London")
    assert not _overlap("apple", "")


# --- two-pass matcher: relaxed dominates strict ----------------------------

def test_two_pass_exact_first_keeps_exact_pairs():
    # An exact pair must not be stolen by an overlapping one (order-independent).
    gold = sorted([(("PER",), ("Bank of America",), "PER"), (("PER",), ("America",), "PER")])
    pred = sorted([(("PER",), ("America",), "PER"), (("PER",), ("Bank of America",), "PER")])
    tp, fp, fn = Counter(), Counter(), Counter()
    _match_relaxed(gold, pred, tp, fp, fn)
    assert tp["PER"] == 2 and not fp and not fn


def test_relaxed_counts_overlap_as_tp():
    gold = sorted([(("PER",), ("Bank of America",), "PER")])
    pred = sorted([(("PER",), ("America",), "PER")])
    tp, fp, fn = Counter(), Counter(), Counter()
    _match_relaxed(gold, pred, tp, fp, fn)
    assert tp["PER"] == 1 and not fp and not fn


# --- relaxed event item builders drop the trigger link ---------------------

def test_relaxed_trigger_is_event_type_presence():
    strict = {("Conflict.Attack", "bombed"), ("Conflict.Attack", "struck"), ("Life.Die", "killed")}
    items = _items_trigger(strict)
    # two Conflict.Attack triggers collapse to one event-type presence item
    assert sorted(d[0] for d, _s, _k in items) == ["Conflict.Attack", "Life.Die"]
    assert all(s == () for _d, s, _k in items)


def test_relaxed_argument_drops_trigger():
    strict = {("Conflict.Attack", "Target", "tower", "bombed"),
              ("Conflict.Attack", "Target", "tower", "struck")}  # same arg, diff triggers
    items = _items_argument(strict)
    assert len(items) == 1                       # deduped once the trigger is dropped
    assert items[0] == (("Conflict.Attack", "Target"), ("tower",), "Target")


# --- end-to-end via a fake model -------------------------------------------

class _FakeModel:
    def __init__(self, preds):
        self.preds = preds

    def batch_extract(self, texts, schemas, batch_size=8, threshold=0.5):
        return self.preds


class _FakeDS:
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def _run(gold_output, pred):
    return compute_metrics(_FakeModel([pred]), _FakeDS([("the text", gold_output)]))


def test_entity_strict_zero_relaxed_one_on_overlap():
    m = _run({"entities": {"PER": ["Bank of America"]}}, {"entities": {"PER": ["America"]}})
    assert m["eval_entity_strict_micro_f1"] == 0.0
    assert m["eval_entity_relaxed_micro_f1"] == 1.0


def test_all_categories_score_with_real_engine_output_shapes():
    """Feed compute_metrics predictions in the inference engine's ACTUAL output
    shapes for every category at once -- entities as a {label: [str]} dict,
    relations as (head, tail) tuples, classifications under the task key, events
    as {trigger, arguments:[{role, entity}]} dicts. Every category must score
    (guards the whole pred-parsing surface against format drift)."""
    gold = {
        "entities": {"PER": ["Alice"], "ORG": ["Acme"]},
        "relations": [{"works_for": {"head": "Alice", "tail": "Acme"}}],
        "classifications": [{"task": "topic", "labels": ["business", "sports"], "true_label": "business"}],
        "events": [{"event_type": "Hire", "trigger": "joined",
                    "arguments": [{"role": "Employee", "entity": "Alice"}]}],
    }
    pred = {
        "entities": {"PER": ["Alice"], "ORG": ["Acme"]},          # engine entity dict
        "relation_extraction": {"works_for": [("Alice", "Acme")]},  # default tuple shape
        "topic": "business",                                        # classification under task key
        "event_extraction": {"Hire": [{"trigger": "joined",        # event dict, string leaves
                                        "arguments": [{"role": "Employee", "entity": "Alice"}]}]},
    }
    m = _run(pred=pred, gold_output=gold)
    for cat in ("entity", "relation", "classification", "event_trigger", "event_argument"):
        assert m[f"eval_{cat}_strict_micro_f1"] > 0.0, f"{cat} scored 0 (pred parsing/format issue)"
        assert m[f"eval_{cat}_relaxed_micro_f1"] >= m[f"eval_{cat}_strict_micro_f1"]


def test_relaxed_never_below_strict_across_categories():
    gold = {
        "entities": {"PER": ["Bank of America"], "LOC": ["New York City"]},
        "relations": [{"located_in": {"head": "Bank of America", "tail": "New York City"}}],
    }
    pred = {
        "entities": {"PER": ["America"], "LOC": ["New York"]},
        "relation_extraction": {"located_in": [{"head": "America", "tail": "New York"}]},
    }
    m = _run(gold, pred)
    for cat in ("entity", "relation"):
        assert m[f"eval_{cat}_relaxed_micro_f1"] >= m[f"eval_{cat}_strict_micro_f1"]
        assert m[f"eval_{cat}_relaxed_micro_recall"] >= m[f"eval_{cat}_strict_micro_recall"]
    # both regimes are emitted for every present category
    assert "eval_entity_strict_support" in m and "eval_entity_relaxed_support" in m
