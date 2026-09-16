"""Injected negatives must never contradict gold, and must be deterministic.

A wrong negative does not fail loudly. It teaches the model that a true label is false, at
loss weight, for a whole run -- so the properties below are the feature, not decoration.
See LABEL_NEGATIVES_PLAN.md section 3.
"""

import pytest

from gliner2.training.negatives import NegativeLabels

# cmnee annotates events and NO entities; biored annotates entities and NO events. That
# asymmetry is real (measured by tools/data/build_negative_pools.py) and is the whole reason
# pools are per-corpus.
POOLS = {
    "cmnee": {
        "annotates": {"entities": False, "events": True, "relations": False,
                      "structures": False},
        "entities": [],
        "events": {"Experiment": ["Subject"], "Accident": ["Subject"],
                   "Conflict": ["Subject"], "Deploy": ["Subject"]},
        "relations": [], "structures": {},
    },
    "biored": {
        "annotates": {"entities": True, "events": False, "relations": True,
                      "structures": False},
        "entities": ["Chemical", "Disease", "Gene", "Species"],
        "events": {},
        "relations": ["Association", "Positive_Correlation"], "structures": {},
    },
}

CMNEE_REC = {"events": [{"event_type": "Experiment", "triggers": ["tested"],
                         "arguments": [{"role": "Subject", "entity": "the army"}]}]}
BIORED_REC = {"entities": {"Chemical": ["aspirin"], "Disease": ["headache"]}}


def _neg(per_dim, seed=42):
    return NegativeLabels(POOLS, per_dim, seed=seed)


def test_off_by_default_so_existing_configs_reproduce():
    out = NegativeLabels(POOLS, {}).inject(dict(BIORED_REC), 0)
    assert out == BIORED_REC


def test_an_injected_entity_label_is_an_EMPTY_LIST():
    """The representation GuideScores already uses, hence already supported end to end."""
    out = _neg({"entities": 2}).inject(dict(BIORED_REC), 0)
    added = set(out["entities"]) - set(BIORED_REC["entities"])
    assert added, "nothing was injected"
    assert all(out["entities"][name] == [] for name in added)


def test_NEVER_injects_a_label_that_is_in_this_record_s_gold():
    """The contradiction the whole design exists to prevent."""
    for i in range(50):
        out = _neg({"entities": 4}).inject(dict(BIORED_REC), i)
        added = set(out["entities"]) - set(BIORED_REC["entities"])
        assert not (added & set(BIORED_REC["entities"]))


def test_WITHIN_DIMENSION_an_events_only_corpus_gets_no_entity_negatives():
    """cmnee annotates zero entities. Offering `Chemical` there would supervise a real
    entity as absent -- the false-negative trap, and the reason pools carry `annotates`."""
    out = _neg({"entities": 3, "events": 1}).inject(dict(CMNEE_REC), 0)
    assert "entities" not in out, "an events-only corpus must get no entity negatives"
    assert out.get("absent_events"), "but it should still get EVENT negatives"


def test_WITHIN_DIMENSION_an_entities_only_corpus_gets_no_event_negatives():
    out = _neg({"entities": 2, "events": 3}).inject(dict(BIORED_REC), 0)
    assert "absent_events" not in out
    assert set(out["entities"]) - set(BIORED_REC["entities"])


def test_within_corpus_a_record_matching_NO_pool_gets_nothing():
    """No candidate corpus means we cannot know what absence means here. Inject nothing."""
    alien = {"entities": {"Spacecraft": ["Voyager"]}}
    neg = _neg({"entities": 3})
    assert neg.inject(dict(alien), 0) == alien
    assert neg.stats["no_candidate"] == 1


def test_deterministic_for_the_same_seed_epoch_and_index():
    """Workers and DDP ranks must build the SAME menu -- they call this independently."""
    a, b = _neg({"entities": 2}), _neg({"entities": 2})
    a.set_epoch(3)
    b.set_epoch(3)
    assert a.inject(dict(BIORED_REC), 7) == b.inject(dict(BIORED_REC), 7)


def test_resamples_across_epochs():
    """Per-epoch resampling is the reason this lives in __getitem__ and not at load."""
    neg = _neg({"entities": 1})
    seen = set()
    for epoch in range(12):
        neg.set_epoch(epoch)
        out = neg.inject(dict(BIORED_REC), 0)
        seen |= set(out["entities"]) - set(BIORED_REC["entities"])
    assert len(seen) > 1, "the same negative every epoch is a frozen menu, not resampling"


def test_pool_exhaustion_is_not_an_error():
    """biored's pool has 4 entity labels, 2 are gold: asking for 5 must yield the 2 left."""
    out = _neg({"entities": 5}).inject(dict(BIORED_REC), 0)
    added = set(out["entities"]) - set(BIORED_REC["entities"])
    assert added == {"Gene", "Species"}


def test_composition_line_can_actually_fail_an_AB_gate():
    neg = _neg({"entities": 2})
    for i in range(10):
        neg.inject(dict(BIORED_REC), i)
    line = neg.composition_line()
    assert "entities k=2" in line and "10/10 records (100.0%)" in line
    off = NegativeLabels(POOLS, {})
    assert "NONE" in off.composition_line()


# ---------------------------------------------------------------------------------------
# END TO END: the gate for the whole feature. Injection is worthless unless the collator
# turns it into an ABSENT QUERY -- the positive class of abstention_loss and the selection
# pool of negative_query_ratio, both live and starved in every run trained so far.
# ---------------------------------------------------------------------------------------

def test_injected_negatives_become_REAL_QUERIES_through_the_collator(tiny_tokenizer):
    from gliner2.processor import SchemaTransformer

    proc = SchemaTransformer(tokenizer=tiny_tokenizer)
    text = "aspirin treats headache"

    def n_queries(schema):
        b = proc.collate_fn_train([(text, schema)], architecture="boundary",
                                  error_policy="skip", max_gold_per_query=256,
                                  on_capacity_exceeded="skip_sample")
        return b.query_layouts[0].extractive_count() if b.query_layouts else 0

    base = n_queries(dict(BIORED_REC))
    injected = _neg({"entities": 2}).inject(dict(BIORED_REC), 0)
    after = n_queries(injected)

    assert after == base + 2, (
        f"two absent entity labels must add two queries ({base} -> {after}); if this does not "
        "move, negative_query_ratio still has nothing to select and the feature is inert"
    )


def test_absent_EVENT_types_become_queries_too(tiny_tokenizer):
    """The events path needed real work: the training list skips an event with no triggers
    (processor.py), so absent types ride the `absent_events` key instead."""
    from gliner2.processor import SchemaTransformer

    proc = SchemaTransformer(tokenizer=tiny_tokenizer)
    text = "the army tested a missile"

    def n_queries(schema):
        b = proc.collate_fn_train([(text, schema)], architecture="boundary",
                                  error_policy="skip", max_gold_per_query=256,
                                  on_capacity_exceeded="skip_sample")
        return b.query_layouts[0].extractive_count() if b.query_layouts else 0

    base = n_queries(dict(CMNEE_REC))
    injected = _neg({"events": 2}).inject(dict(CMNEE_REC), 0)
    assert injected.get("absent_events"), "no absent event types were injected"
    after = n_queries(injected)

    assert after > base, f"absent event types must add queries ({base} -> {after})"


def test_an_absent_event_type_never_duplicates_a_gold_one(tiny_tokenizer):
    """`absent_events` is merged beside the gold list, so a collision would create two
    queries for one type -- one supervised present, one absent. Same type, both ways."""
    from gliner2.processor import SchemaTransformer

    proc = SchemaTransformer(tokenizer=tiny_tokenizer)
    schema = dict(CMNEE_REC, absent_events={"Experiment": ["Subject"]})   # Experiment IS gold
    b = proc.collate_fn_train([("the army tested a missile", schema)], architecture="boundary",
                              error_policy="skip", max_gold_per_query=256,
                              on_capacity_exceeded="skip_sample")
    gold_only = proc.collate_fn_train([("the army tested a missile", dict(CMNEE_REC))],
                                      architecture="boundary", error_policy="skip",
                                      max_gold_per_query=256,
                                      on_capacity_exceeded="skip_sample")
    assert b.query_layouts[0].extractive_count() == gold_only.query_layouts[0].extractive_count()
