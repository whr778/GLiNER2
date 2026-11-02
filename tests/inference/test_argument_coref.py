"""An argument span records the one coreference cluster it equals, and the link score can fail."""
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


def test_one_exact_span_writes_the_cluster_index():
    pred = _pred()
    counts = attach_argument_clusters(pred)
    assert pred["event_extraction"]["Leave"][0]["arguments"][0]["coref_cluster"] == 0
    assert counts == {"linked": 1, "ambiguous": 0, "none": 0}
    attach_argument_clusters(pred)
    assert pred["event_extraction"]["Leave"][0]["arguments"][0]["coref_cluster"] == 0


def test_a_span_in_two_clusters_writes_nothing():
    paul = _sp("Paul", 0)
    pred = _pred(paul, [[paul, _sp("he", 5)], [paul, _sp("him", 20)]])
    counts = attach_argument_clusters(pred)
    assert "coref_cluster" not in pred["event_extraction"]["Leave"][0]["arguments"][0]
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


def test_event_argument_sets_are_unchanged():
    pred = _pred()
    before = M._pred_event_argument_set(pred)
    attach_argument_clusters(pred)
    assert M._pred_event_argument_set(pred) == before
    assert len(pred["event_extraction"]["Leave"][0]["arguments"]) == 1


def test_score_is_strict_micro_f1_when_gold_clusters_exist():
    pred = _pred()
    metrics = score_argument_clusters([_gold()], [pred])
    assert metrics["eval_argument_cluster_link_strict_micro_f1"] == 1.0
    assert metrics["eval_argument_cluster_link_strict_micro_precision"] == 1.0
    assert metrics["eval_argument_cluster_link_strict_micro_recall"] == 1.0
    assert metrics["eval_argument_cluster_link_support"] == 1
    assert metrics["eval_argument_cluster_link_gold_documents"] == 1
    assert pred["event_extraction"]["Leave"][0]["arguments"][0]["coref_cluster"] == 0


def test_a_miss_scores_zero_recall():
    pred = _pred()
    del pred["coref_clusters"]
    metrics = score_argument_clusters([_gold()], [pred])
    assert metrics["eval_argument_cluster_link_strict_micro_recall"] == 0.0
    assert metrics["eval_argument_cluster_link_strict_micro_f1"] == 0.0
    assert metrics["eval_argument_cluster_link_support"] == 1


def test_every_document_without_clusters_omits_f1():
    gold = {"events": _gold()["events"]}
    metrics = score_argument_clusters([gold], [_pred()])
    assert "eval_argument_cluster_link_strict_micro_f1" not in metrics
    assert metrics["eval_argument_cluster_link_gold_documents"] == 0
    assert metrics["eval_argument_cluster_link_ungraded_documents"] == 1
    assert metrics["eval_argument_cluster_link_support"] == 0
