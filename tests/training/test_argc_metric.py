"""Arg-C: OneIE's argument criterion, for EXTERNAL comparison only.

WHY IT EXISTS. OneIE: "An argument is correctly identified (Arg-I) if its offsets and event
type match a reference argument mention. It is correctly classified (Arg-C) if its role
label also matches." Offsets + type + role, NO trigger identity. Our `event_argument`
strict key adds the trigger, so it is a LOWER bound on an OneIE-comparable number; our
relaxed drops exact spans, so it is an UPPER bound. The comparable figure sat between them
and was never computed, which left every comparison to the literature unsupported in BOTH
directions. See EVENT_ARGUMENT_DIAGNOSIS 4c-i for the five decisions these tests pin down.

WHAT THESE TESTS ARE REALLY GUARDING. Arg-C reads far higher than strict BY CONSTRUCTION,
because dropping the trigger means an argument bound to the WRONG INSTANCE of the right
type scores as correct -- the exact failure `event_records` exists to fix. The danger is
not that the metric is wrong; it is that it is flattering. So the last two tests assert it
stays OUT of the aggregates and keeps a name nothing prefix-matching `event_` can reach.
"""

from __future__ import annotations

from gliner2.training.eval_metrics import (
    _gold_event_argc_set,
    _gold_event_argument_set,
    _pred_event_argc_set,
    _pred_event_argument_set,
)

GOLD = {"events": [
    {"event_type": "Attack", "triggers": ["bombed"],
     "arguments": [{"role": "Target", "entity": "the market"}]},
    {"event_type": "Attack", "triggers": ["shelled"],
     "arguments": [{"role": "Target", "entity": "the bridge"}]},
    {"event_type": "Death", "triggers": [],
     "arguments": [{"role": "Victim", "entity": "three civilians"}]},
]}

# Right arguments, bound to the WRONG instances, plus a repeat of one already predicted.
PRED = {"event_extraction": {"Attack": [
    {"triggers": ["shelled"], "arguments": [{"role": "Target", "entity": "the market"}]},
    {"triggers": ["bombed"], "arguments": [{"role": "Target", "entity": "The Bridge"}]},
    {"triggers": ["struck"], "arguments": [{"role": "Target", "entity": "the market"}]},
]}}


def _prf(gold, pred):
    return len(gold & pred), len(pred - gold), len(gold - pred)


def test_a_trigger_less_event_is_included_and_counted():
    """The strict builder drops it before forming a key -- right there, wrong here."""
    keys, triggerless = _gold_event_argc_set(GOLD)
    assert ("death", "victim", "three civilians") in keys
    assert triggerless == 1
    strict = _gold_event_argument_set(GOLD)
    assert not any(k[0] == "Death" for k in strict), (
        "the trigger-keyed metric must still drop it; that is why Arg-C needs its own builder"
    )


def test_the_key_drops_the_trigger_and_keeps_the_type():
    keys, _ = _gold_event_argc_set(GOLD)
    assert all(len(k) == 3 for k in keys)
    assert ("attack", "target", "the market") in keys


def test_duplicates_collapse_because_it_is_a_set():
    """OneIE's scorer is `args.add(...)`. Three predicted mentions, two distinct keys."""
    keys, _ = _pred_event_argc_set(PRED)
    assert len(keys) == 2, sorted(keys)


def test_matching_is_case_insensitive():
    keys, _ = _pred_event_argc_set(PRED)
    assert ("attack", "target", "the bridge") in keys, "'The Bridge' must match 'the bridge'"


def test_argc_scores_higher_than_strict_on_identical_predictions():
    """The headline property, and the reason it must never be quoted as event_argument.

    These predictions find both arguments and bind both to the WRONG Attack instance.
    Strict scores zero. Arg-C scores well. Nothing about the model differs.
    """
    s_tp, _, _ = _prf(_gold_event_argument_set(GOLD), _pred_event_argument_set(PRED))
    c_tp, c_fp, c_fn = _prf(_gold_event_argc_set(GOLD)[0], _pred_event_argc_set(PRED)[0])
    assert s_tp == 0, "strict must refuse an argument bound to the wrong instance"
    assert c_tp == 2 and c_fp == 0 and c_fn == 1


def test_argc_is_not_a_scored_head():
    """It must not reach `primitive_heads`, which feeds head_macro and head_min -- and
    head_min is what eb17-best selects its checkpoint on. A flattering extra head would
    lift the selection metric for no change in the model."""
    import inspect

    from gliner2.training import eval_metrics

    src = inspect.getsource(eval_metrics.score_predictions)
    # Split on the CLOSING LINE, not on ")" -- the first ")" ends the first tuple row, so
    # `.split(")")[0]` inspects one line and can never fail. It was written that way, it
    # passed with argc deliberately planted in the block, and that is the whole reason
    # this comment exists.
    after = src.split("primitive_heads = (", 1)[1]
    block = after.split("\n    )", 1)[0]
    assert "entity" in block and "event_argument" in block, (
        "the block was not parsed -- this guard would pass on anything"
    )
    assert "argc" not in block, "Arg-C leaked into primitive_heads"


def test_the_metric_name_cannot_be_mistaken_for_event_argument():
    """`eval_argc_external_*`, so nothing prefix-matching `event_` picks it up."""
    from gliner2.training import eval_metrics
    import inspect

    src = inspect.getsource(eval_metrics.score_predictions)
    assert '_finalize("argc", "external"' in src
    assert 'eval_argc_external_triggerless_gold' in src


# ---- OneIE's other three criteria: Trig-I, Trig-C, Arg-I ------------------------------
#
# Added 2026-09-29. The LEVELS are for citing the literature; the DIFFERENCES are the
# reason to want them:
#   Trig-I - Trig-C  = trigger TYPE confusion, span already correct
#   Arg-I  - Arg-C   = ROLE confusion, span and event type already correct
# Neither decomposition exists in `event_trigger` / `event_argument`, which report each
# failure as a single number.

from gliner2.training.eval_metrics import _gold_oneie_sets, _pred_oneie_sets  # noqa: E402

# Right spans throughout; one wrong ROLE (Target -> Attacker) and one wrong event TYPE
# (a "bombed" trigger also emitted under Death). So every failure is classification.
_G = {"events": [
    {"event_type": "Attack", "triggers": ["bombed"],
     "arguments": [{"role": "Target", "entity": "the market"}]},
    {"event_type": "Attack", "triggers": ["shelled"],
     "arguments": [{"role": "Target", "entity": "the bridge"}]},
    {"event_type": "Death", "triggers": [],
     "arguments": [{"role": "Victim", "entity": "three civilians"}]},
]}
_P = {"event_extraction": {
    "Attack": [
        {"triggers": ["shelled"], "arguments": [{"role": "Target", "entity": "the market"}]},
        {"triggers": ["bombed"], "arguments": [{"role": "Attacker", "entity": "The Bridge"}]},
    ],
    "Death": [{"triggers": ["bombed"],
               "arguments": [{"role": "Victim", "entity": "three civilians"}]}],
}}


def _f1(gold, pred):
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _sets():
    return _gold_oneie_sets(_G)[0], _pred_oneie_sets(_P)[0]


def test_the_bracket_holds_by_construction():
    """Each metric is looser than the one it refines. If this ever inverts, a key is wrong."""
    g, p = _sets()
    assert _f1(g["trigi"], p["trigi"]) >= _f1(g["trigc"], p["trigc"])
    assert _f1(g["argi"], p["argi"]) >= _f1(g["argc"], p["argc"])


def test_the_gaps_isolate_classification_error():
    """Every span here is correct, so identification is perfect and the gaps are the
    whole failure -- which is exactly what the decomposition is for."""
    g, p = _sets()
    assert _f1(g["trigi"], p["trigi"]) == 1.0, "all trigger spans were found"
    assert _f1(g["argi"], p["argi"]) == 1.0, "all argument spans were found for the right type"
    assert _f1(g["trigc"], p["trigc"]) < 1.0, "but one trigger got the wrong event type"
    assert _f1(g["argc"], p["argc"]) < 1.0, "and one argument got the wrong role"


def test_trigger_keys_have_the_right_shape():
    g, _ = _sets()
    assert all(len(k) == 1 for k in g["trigi"]), "Trig-I is the trigger alone"
    assert all(len(k) == 2 for k in g["trigc"]), "Trig-C adds the event type"
    assert all(len(k) == 2 for k in g["argi"]), "Arg-I is (event_type, entity), no role"
    assert all(len(k) == 3 for k in g["argc"]), "Arg-C adds the role"


def test_none_of_the_four_is_a_scored_head():
    """All four are looser than our own metrics, so any of them in `primitive_heads` would
    lift head_min -- eb17's checkpoint-selection metric -- for no change in the model."""
    import inspect

    from gliner2.training import eval_metrics

    src = inspect.getsource(eval_metrics.score_predictions)
    block = src.split("primitive_heads = (", 1)[1].split("\n    )", 1)[0]
    assert "entity" in block and "event_argument" in block, "block not parsed"
    for name in ("argc", "argi", "trigc", "trigi"):
        assert f'"{name}"' not in block, f"{name} leaked into primitive_heads"


def test_all_four_emit_under_the_external_label():
    import inspect

    from gliner2.training import eval_metrics

    src = inspect.getsource(eval_metrics.score_predictions)
    for name in ("argc", "argi", "trigc", "trigi"):
        assert f'_finalize("{name}", "external"' in src
