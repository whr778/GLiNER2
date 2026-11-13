"""An argument span records the coreference cluster it matches (strict or relaxed), and the link score can fail."""
from gliner2.inference.argument_coref import attach_argument_clusters, score_argument_clusters
from gliner2.training import eval_metrics as M


def _sp(text, start):
    return {"text": text, "start": start, "end": start + len(text)}


def _pred(span=_sp("Paul", 0), clusters=None):
    return {
        "event_extraction": {"Leave": [{
            "triggers": [_sp("left", 10)],
            "arguments": [{"role": "Person", "entity": span}],
        }]},
        "coref_clusters": clusters if clusters is not None else [[_sp("Paul", 0), _sp("he", 5)]],
    }


def _gold():
    return {
        "events": [{"event_type": "Leave", "triggers": ["left"],
                    "arguments": [{"role": "Person", "entity": "Paul"}]}],
        "coref_clusters": [[_sp("Paul", 0), _sp("he", 5)]],
    }


def test_one_exact_span_writes_the_cluster_index_and_copies_the_other_mentions():
    pred = _pred()
    counts = attach_argument_clusters(pred)
    args = pred["event_extraction"]["Leave"][0]["arguments"]
    assert [(a["entity"]["text"], a["coref_cluster"]) for a in args] == [("Paul", 0), ("he", 0)]
    assert counts == {"linked": 1, "ambiguous": 0, "none": 0}
    attach_argument_clusters(pred)
    args = pred["event_extraction"]["Leave"][0]["arguments"]
    assert [(a["entity"]["text"], a["coref_cluster"]) for a in args] == [("Paul", 0), ("he", 0)]


def test_a_span_in_two_clusters_writes_nothing():
    paul = _sp("Paul", 0)
    pred = _pred(paul, [[paul, _sp("he", 5)], [paul, _sp("him", 20)]])
    counts = attach_argument_clusters(pred)
    args = pred["event_extraction"]["Leave"][0]["arguments"]
    assert "coref_cluster" not in args[0]
    assert len(args) == 1
    assert counts["ambiguous"] == 1 and counts["linked"] == 0


def test_no_shared_span_is_none():
    pred = _pred(_sp("Mary", 30))
    counts = attach_argument_clusters(pred)
    assert "coref_cluster" not in pred["event_extraction"]["Leave"][0]["arguments"][0]
    assert counts["none"] == 1


def test_missing_clusters_empty_clusters_and_a_string_entity_do_not_raise():
    missing = _pred()
    del missing["coref_clusters"]
    assert attach_argument_clusters(missing)["none"] == 1
    assert "coref_cluster" not in missing["event_extraction"]["Leave"][0]["arguments"][0]
    empty = _pred(clusters=[])
    assert attach_argument_clusters(empty)["none"] == 1
    string_arg = _pred()
    string_arg["event_extraction"]["Leave"][0]["arguments"][0]["entity"] = "Paul"
    assert attach_argument_clusters(string_arg)["none"] == 1
    assert "coref_cluster" not in string_arg["event_extraction"]["Leave"][0]["arguments"][0]


def test_the_copied_mention_is_an_argument_of_the_same_role():
    pred = _pred()
    before = M._pred_event_argument_set(pred)
    attach_argument_clusters(pred)
    after = M._pred_event_argument_set(pred)
    assert before < after
    assert any(item[2] == "he" for item in after)


def test_score_is_strict_micro_f1_when_gold_clusters_exist():
    pred = _pred()
    metrics = score_argument_clusters([_gold()], [pred])
    assert metrics["eval_argument_cluster_link_strict_micro_f1"] == 1.0
    assert metrics["eval_argument_cluster_link_strict_micro_precision"] == 1.0
    assert metrics["eval_argument_cluster_link_strict_micro_recall"] == 1.0
    assert metrics["eval_argument_cluster_link_relaxed_micro_f1"] == 1.0
    assert metrics["eval_argument_cluster_link_support"] == 2
    assert metrics["eval_argument_cluster_link_gold_documents"] == 1
    args = pred["event_extraction"]["Leave"][0]["arguments"]
    assert [(a["entity"]["text"], a["coref_cluster"]) for a in args] == [("Paul", 0), ("he", 0)]


def test_a_miss_scores_zero_recall():
    pred = _pred()
    del pred["coref_clusters"]
    metrics = score_argument_clusters([_gold()], [pred])
    assert metrics["eval_argument_cluster_link_strict_micro_recall"] == 0.0
    assert metrics["eval_argument_cluster_link_strict_micro_f1"] == 0.0
    assert metrics["eval_argument_cluster_link_relaxed_micro_recall"] == 0.0
    assert metrics["eval_argument_cluster_link_relaxed_micro_f1"] == 0.0
    assert metrics["eval_argument_cluster_link_support"] == 2


def test_a_copied_mention_outside_the_gold_cluster_is_a_false_positive():
    gold = _gold()
    gold["coref_clusters"] = [[_sp("Paul", 0), _sp("Mary", 30)]]
    metrics = score_argument_clusters([gold], [_pred()])
    assert metrics["eval_argument_cluster_link_strict_micro_precision"] == 0.5
    assert metrics["eval_argument_cluster_link_strict_micro_recall"] == 0.5
    assert metrics["eval_argument_cluster_link_strict_micro_f1"] == 0.5
    assert metrics["eval_argument_cluster_link_support"] == 2


def test_every_document_without_clusters_omits_f1():
    gold = {"events": _gold()["events"]}
    metrics = score_argument_clusters([gold], [_pred()])
    assert "eval_argument_cluster_link_strict_micro_f1" not in metrics
    assert "eval_argument_cluster_link_relaxed_micro_f1" not in metrics
    assert metrics["eval_argument_cluster_link_gold_documents"] == 0
    assert metrics["eval_argument_cluster_link_ungraded_documents"] == 1
    assert metrics["eval_argument_cluster_link_support"] == 0


def test_relaxed_match_links_overlapping_span_when_unambiguous_and_refuses_when_ambiguous():
    # "President Paul" (0..14) overlaps cluster mention "Paul" (10..14)
    clusters = [[_sp("Paul", 10), _sp("he", 25)]]
    pred_strict = _pred(_sp("President Paul", 0), clusters)
    assert attach_argument_clusters(pred_strict, match_mode="strict") == {"linked": 0, "ambiguous": 0, "none": 1}

    pred_rel = _pred(_sp("President Paul", 0), clusters)
    counts = attach_argument_clusters(pred_rel, match_mode="relaxed")
    assert counts == {"linked": 1, "ambiguous": 0, "none": 0}
    args = pred_rel["event_extraction"]["Leave"][0]["arguments"]
    assert args[0]["coref_cluster"] == 0 and args[0]["coref_match"] == "relaxed"
    assert [a["entity"]["text"] for a in args] == ["President Paul", "Paul", "he"]

    # Ambiguous overlap ("Paul and Mary" overlapping two clusters) must NOT link
    two_clusters = [[_sp("Paul", 0), _sp("he", 25)], [_sp("Mary", 9), _sp("she", 35)]]
    pred_amb = _pred(_sp("Paul and Mary", 0), two_clusters)
    assert attach_argument_clusters(pred_amb, match_mode="relaxed") == {"linked": 0, "ambiguous": 1, "none": 0}
    assert "coref_cluster" not in pred_amb["event_extraction"]["Leave"][0]["arguments"][0]


def test_copy_modes_canonical_gated_and_none():
    clusters = [[_sp("James Whitey Bulger", 0), _sp("Bulger", 25), _sp("he", 40), _sp("him", 60)]]
    # 1. copy_mode="canonical" on pronoun "he" copies only "James Whitey Bulger"
    pred_can = _pred(_sp("he", 40), clusters)
    attach_argument_clusters(pred_can, copy_mode="canonical")
    args_can = pred_can["event_extraction"]["Leave"][0]["arguments"]
    assert [a["entity"]["text"] for a in args_can] == ["he", "James Whitey Bulger"]

    # On non-pronoun "Bulger", copy_mode="canonical" does not flood with pronouns
    pred_nonpro = _pred(_sp("Bulger", 25), clusters)
    attach_argument_clusters(pred_nonpro, copy_mode="canonical")
    args_nonpro = pred_nonpro["event_extraction"]["Leave"][0]["arguments"]
    assert [a["entity"]["text"] for a in args_nonpro] == ["Bulger"]

    # 2. copy_mode="gated" copies only members meeting min_member_score
    pred_gated = _pred(_sp("he", 40), clusters)
    trig_sp = (10, 14)
    role_scores = {
        ("Leave", trig_sp, "Person", (0, 19)): 0.45,
        ("Leave", trig_sp, "Person", (25, 31)): 0.20,
        ("Leave", trig_sp, "Person", (60, 63)): 0.02,
    }
    attach_argument_clusters(
        pred_gated, copy_mode="gated", min_member_score=0.30, role_scores=role_scores
    )
    args_gated = pred_gated["event_extraction"]["Leave"][0]["arguments"]
    assert [a["entity"]["text"] for a in args_gated] == ["he", "James Whitey Bulger"]

    # 3. copy_mode="none" tags coref_cluster without copying
    pred_none = _pred(_sp("he", 40), clusters)
    attach_argument_clusters(pred_none, copy_mode="none")
    args_none = pred_none["event_extraction"]["Leave"][0]["arguments"]
    assert len(args_none) == 1 and args_none[0]["coref_cluster"] == 0
