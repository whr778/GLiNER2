"""Per-event argument completeness: mean Jaccard and exact-set rate.

These answer a question micro-F1 structurally cannot -- how completely is a TYPICAL
event recovered -- because micro pools every argument corpus-wide, so one 8-argument
mention counts eight times a 1-argument one. Every value below is computed by hand in
the docstring, so a change in behaviour fails loudly rather than silently re-baselining.
"""
from gliner2.training.eval_metrics import _events_by_key, _per_event_scores

BOMB = ("bombed",)


def arg(etype, role, entity, trig=BOMB):
    return (etype, role, entity, trig)


def test_events_group_by_type_and_trigger():
    """Two mentions of the same type with different triggers are different events."""
    args = {arg("Attack", "target", "school"), arg("Attack", "target", "clinic", ("shelled",))}
    assert len(_events_by_key(args)) == 2


def test_perfect_event_scores_one():
    g = {arg("Attack", "target", "school"), arg("Attack", "agent", "militia")}
    j, x, n = _per_event_scores(g, set(g))
    assert (j, x, n) == (1.0, 1.0, 1)


def test_three_of_four_is_jaccard_three_quarters():
    """3 of 4 gold arguments, nothing extra: J = 3/4 = 0.75.

    The matching micro-F1 for this document is 2*3/(2*3+0+1) = 0.857, and
    2J/(1+J) = 1.5/1.75 = 0.857 -- the two are monotone transforms, which is exactly
    why a corpus-level IoU would add no information over corpus-level F1.
    """
    g = {arg("Attack", "target", e) for e in ("a", "b", "c", "d")}
    p = {arg("Attack", "target", e) for e in ("a", "b", "c")}
    j, x, n = _per_event_scores(g, p)
    assert j == 0.75 and x == 0.0 and n == 1
    assert abs(2 * j / (1 + j) - 6 / 7) < 1e-12


def test_hallucinated_event_is_not_free():
    """An invented event scores 0 rather than being invisible.

    Averaging over GOLD events alone would make inventing events cost nothing; the
    denominator is the union of gold and predicted event keys instead.
    """
    g = {arg("Attack", "target", "school")}
    p = g | {arg("Flood", "target", "bridge", ("flooded",))}
    j, x, n = _per_event_scores(g, p)
    assert n == 2 and j == 1.0 and x == 1.0        # summed, not averaged
    assert j / n == 0.5 and x / n == 0.5


def test_micro_and_per_event_disagree_by_design():
    """The case the new metric exists for: one big perfect event, one small wrong one.

    Micro pools arguments: TP 8, FP 1, FN 1 -> F1 = 16/18 = 0.889, dominated by the
    8-argument mention. Per-event weights both equally: (1.0 + 0.0)/2 = 0.500.
    A 0.889 that is really 'one of two events is right' is the thing micro hides.
    """
    big_g = {arg("Attack", f"r{i}", f"e{i}") for i in range(8)}
    small_g = {arg("Flood", "target", "x", ("flooded",))}
    small_p = {arg("Flood", "target", "y", ("flooded",))}
    j, x, n = _per_event_scores(big_g | small_g, big_g | small_p)
    assert n == 2
    assert j / n == 0.5 and x / n == 0.5
    tp, fp, fn = 8, 1, 1
    assert abs(2 * tp / (2 * tp + fp + fn) - 0.8888888888888888) < 1e-12


def test_wrong_trigger_zeroes_the_event():
    """Consistent with strict scoring, where trigger_key is part of the key.

    Same type, same four arguments, different trigger: gold and prediction land under
    different event keys, so both are unmatched and the event scores 0.
    """
    g = {arg("Attack", "target", e) for e in ("a", "b", "c", "d")}
    p = {arg("Attack", "target", e, ("attacked",)) for e in ("a", "b", "c", "d")}
    j, x, n = _per_event_scores(g, p)
    assert n == 2 and j == 0.0 and x == 0.0


def test_empty_is_empty():
    assert _per_event_scores(set(), set()) == (0.0, 0.0, 0)
