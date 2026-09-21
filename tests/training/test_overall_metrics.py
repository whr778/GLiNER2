"""`eval_overall_*`: the aggregate a general-purpose BASE can be selected on.

Until these keys existed a base could only be selected on ONE head or on `eval_loss`, and
loss selection ships a different epoch per arm -- it voided both the absneg and roles2
verdicts. Three numbers are emitted because none of them is trustworthy alone: `micro` pools
tp/fp/fn and is dominated by entity support, `head_macro` weights every head equally so a
single collapsing head is visible, and `head_min` cannot be improved by trading a head away.

The model here is a STUB returning fixed predictions. That is deliberate and is not a
simulation of the code under test: the code under test is the aggregation, and the gold,
the matcher and the counters are all real.
"""

from gliner2.training.eval_metrics import compute_metrics


class _StubModel:
    """Returns a canned prediction per record; that is all compute_metrics needs."""

    def __init__(self, preds):
        self._preds = preds

    def batch_extract(self, texts, schemas=None, **kwargs):
        return [self._preds[t] for t in texts]


def _run(records, preds):
    return compute_metrics(_StubModel(preds), records, batch_size=2,
                           threshold=0.3, report=False)


def test_overall_keys_appear_and_agree_with_the_heads():
    gold_a = {"entities": {"Person": ["Ada"]},
              "classifications": [{"task": "t", "labels": ["x", "y"], "true_label": ["x"]}]}
    gold_b = {"entities": {"Person": ["Grace"]},
              "classifications": [{"task": "t", "labels": ["x", "y"], "true_label": ["y"]}]}
    records = [("Ada Lovelace wrote notes .", gold_a),
               ("Grace Hopper wrote code .", gold_b)]
    # THE PREDICTION SHAPE IS NOT SYMMETRIC, and it is worth stating: entities are read
    # from `pred["entities"]` but a classification is read from a FLAT `pred[task]`.
    preds = {
        "Ada Lovelace wrote notes .": {"entities": {"Person": ["Ada"]}, "t": "x"},   # right
        "Grace Hopper wrote code .": {"entities": {"Person": ["Ada"]}, "t": "x"},    # wrong
    }
    m = _run(records, preds)

    assert "eval_overall_strict_micro_f1" in m
    assert "eval_overall_strict_head_macro_f1" in m
    assert "eval_overall_strict_head_min_f1" in m

    heads = {k: v for k, v in m.items()
             if k.endswith("_strict_micro_f1") and "overall" not in k and "_event_" not in k
             and k != "eval_event_strict_micro_f1"}
    assert m["eval_overall_strict_head_count"] == len(heads)
    assert abs(m["eval_overall_strict_head_macro_f1"]
               - sum(heads.values()) / len(heads)) < 1e-9
    assert abs(m["eval_overall_strict_head_min_f1"] - min(heads.values())) < 1e-9
    assert min(heads.values()) - 1e-9 <= m["eval_overall_strict_micro_f1"] <= max(heads.values()) + 1e-9


def test_head_min_catches_a_head_that_collapsed():
    """The point of head_min: a head at zero cannot be hidden by a strong one."""
    gold = {"entities": {"Person": ["Ada"]},
            "classifications": [{"task": "t", "labels": ["x", "y"], "true_label": ["x"]}]}
    records = [("Ada Lovelace wrote notes .", gold)]
    m = _run(records, {"Ada Lovelace wrote notes .": {"entities": {"Person": ["Ada"]}, "t": "y"}})

    assert m["eval_entity_strict_micro_f1"] == 1.0
    assert m["eval_classification_strict_micro_f1"] == 0.0
    assert m["eval_overall_strict_head_min_f1"] == 0.0
    assert m["eval_overall_strict_head_macro_f1"] == 0.5


def test_a_single_head_emits_no_overall():
    """Fewer than two live heads means there is nothing to aggregate."""
    records = [("Ada Lovelace wrote notes .", {"entities": {"Person": ["Ada"]}})]
    m = _run(records, {"Ada Lovelace wrote notes .": {"entities": {"Person": ["Ada"]}}})
    assert not any(k.startswith("eval_overall_") for k in m)


def test_relaxed_regime_is_emitted_too():
    gold = {"entities": {"Person": ["Ada Lovelace"]},
            "classifications": [{"task": "t", "labels": ["x", "y"], "true_label": ["x"]}]}
    records = [("Ada Lovelace wrote notes .", gold)]
    m = _run(records, {"Ada Lovelace wrote notes .": {"entities": {"Person": ["Ada"]}, "t": "x"}})
    assert "eval_overall_relaxed_micro_f1" in m
    # a partial span counts under relaxed and not under strict
    assert m["eval_overall_relaxed_micro_f1"] >= m["eval_overall_strict_micro_f1"]
