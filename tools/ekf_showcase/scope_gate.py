"""The scope gate: is this figure the place's own toll, the national total, or neither?

Extracted from `scope_gate_test.py` so the pipeline and the experiment share ONE
implementation. The experiment measured it; `run_pipeline` now applies it.

Measured on Helene (106 `dead` observations, ratio 2.0): per-place RMSE **217.4 -> 21.9
deaths**, a 9.9x cut, and the national stream improves too (80.9 -> 63.5). A control that
removes the same number of observations at random scores 4.427 against the gate's 0.591
nRMSE, so it is selecting rather than thinning. Flat from ratio 1.5 to 2.5.

Three outcomes, not two -- that is load-bearing. A two-way version that rerouted every
reject wrecked the national stream (0.402 -> 2.110), because North Carolina's 1400 is not
a national total; it is not a casualty count at all.
"""
from __future__ import annotations

import math


def running_max(observations: list, predicate) -> list:
    """Running (t, max) over the observations matching ``predicate``, in time order."""
    out, run = [], 0.0
    for o in sorted((o for o in observations if predicate(o)), key=lambda o: o["t_hours"]):
        run = max(run, float(o["value"]))
        out.append((o["t_hours"], run))
    return out


def scale_at(scale: list, t: float) -> float:
    best = 0.0
    for tt, v in scale:
        if tt <= t:
            best = v
    return best


def clean_parts(observations: list, states: dict, ratio: float = 2.0) -> list:
    """Pass 1: drop the grossly-contaminated part observations, using the raw aggregate."""
    agg = running_max(observations, lambda o: str(o.get("event_key")) == "__aggregate__")
    out = []
    for o in observations:
        key = str(o.get("event_key"))
        natl = scale_at(agg, o["t_hours"])
        if key not in states or natl <= 0 or float(o["value"]) < natl / ratio:
            out.append(o)
    return out


def reference_for(observations: list, key: str, states: dict, mode: str) -> list:
    """The larger-scope series a stream is judged against.

    Gating a stream against its OWN running scale fails on its early history, where a toll
    legitimately jumps 6 -> 25 faster than any ratio tolerates. It has to be judged against
    something larger, and WHICH larger thing is the whole difficulty.

    ``aggregate``   the raw ``__aggregate__`` series. Works while a part is small relative
                    to the whole, and breaks down as the part approaches it.
    ``global-max``  running max across all streams. Needed where no aggregate stream exists,
                    but circular for the dominant stream: on Turkiye-Syria it judged Turkey
                    against a reference Turkey itself defines and rerouted Turkey's own true
                    1,014.
    ``implied``     **aggregate minus the other parts' running estimates** -- this part's
                    implied maximum, and the only one that survives a dominant part. Turkey
                    is 87.6% of its total, so 41,000 is indistinguishable from the whole by
                    magnitude alone; against an implied max of 46,800 - 5,800 = 41,000 it is
                    exactly at its ceiling and correctly kept, while the same 41,000 filed
                    under Syria faces an implied max of 5,800 and is correctly rerouted.
                    Requires a DECLARED hierarchy, which is what `rollup.json` now carries.
    """
    if mode == "global-max":
        return running_max(observations, lambda o: True)
    agg = running_max(observations, lambda o: str(o.get("event_key")) == "__aggregate__")
    if mode != "implied":
        return agg
    # The other parts must be CLEANED first, or the subtraction destroys the reference:
    # North Carolina's stream holds 250 and Florida's holds 300, so the raw sum of the
    # parts exceeds the whole and every implied maximum clamps to zero. Pass 1 gates
    # against the raw aggregate to remove the gross contamination; pass 2 computes the
    # implied maximum from what survived. Cheap two-pass rather than a fixed point --
    # a third pass moved nothing on either event.
    cleaned = clean_parts(observations, states)
    others = {k: running_max(cleaned, lambda o, k=k: str(o.get("event_key")) == k)
              for k in states if k != key}
    return [(t, max(0.0, v - sum(scale_at(o, t) for o in others.values())))
            for t, v in agg]


def gate(observations: list, ratio: float, warmup: int,
         states: dict, reference: str = "aggregate", down_ratio: float = 0.0):
    """Classify each state observation as its own toll, the national total, or neither.

    Three outcomes, not two. The earlier two-way version rerouted every reject to
    ``__aggregate__`` and that is what destroyed the national stream: 1400 filed under
    North Carolina is not a national total, it is not a casualty count at all, and moving
    it into the aggregate poisoned the one measurement that worked.

    ``keep``    below ``ratio`` of the running national total -- plausibly the state's own
    ``reroute`` within [1/ratio, ratio] of the national total -- it IS the national figure
    ``drop``    above the national total -- no scope in this event can exceed the whole

    ``down_ratio`` adds the SECOND side. The rule above only ever rejects upward, but
    measured on Helene 63% of the three-way oracle's rejects are readings BELOW the
    place's own truth -- an article still saying "three" dead in North Carolina at
    t=83.8h when the toll is 46 (``reject_headroom.py``). A death toll does not fall,
    so a reading far below this stream's own running maximum is stale, not a rival
    claim, and it drags the filter down. Rejecting it needs no national reference and
    no model. 0.0 disables, preserving the original one-sided behaviour exactly.
    """
    kept: dict[str, list] = {}
    moved: list = []
    dropped: list = []
    by_key: dict[str, list] = {}
    for o in observations:
        by_key.setdefault(str(o.get("event_key")), []).append(o)

    for key, obs in by_key.items():
        obs = sorted(obs, key=lambda o: o["t_hours"])
        if key not in states:                # only per-place streams are gated
            kept.setdefault(key, []).extend(obs)
            continue
        scale = reference_for(observations, key, states, reference)
        run = 0.0                      # this stream's own running max over KEPT values
        for i, o in enumerate(obs):
            v, natl = float(o["value"]), scale_at(scale, o["t_hours"])
            if down_ratio > 0 and i >= warmup and run > 0 and v < run / down_ratio:
                dropped.append(dict(o, _from=key, _stale=True))
                continue
            if ratio <= 0 or i < warmup or natl <= 0 or v < natl / ratio:
                kept.setdefault(key, []).append(o)
                run = max(run, v)
            elif v <= natl * ratio:
                moved.append(dict(o, event_key="__aggregate__", _from=key))
            else:
                dropped.append(dict(o, _from=key))
    for o in moved:
        kept.setdefault("__aggregate__", []).append(o)
    return kept, moved, dropped


# --------------------------------------------------------------------------- #
# Extracted scope: the field the ratio gate exists to reconstruct
# --------------------------------------------------------------------------- #
SCOPE_CLASSES = ("place", "national", "sub-place", "unclear")


def apply_extracted_scope(observations: list, states: dict, aggregate_key="__aggregate__"):
    """Route by the scope the EXTRACTOR emitted, before falling back to the ratio gate.

    The ratio gate infers extent from magnitude after the fact -- a figure near the
    running national total is probably the national total. That works, and it is a
    reconstruction of a field the model could have produced beside the number. Where the
    model states the extent, use it:

        national   -> reroute to the aggregate stream (it is the whole, not the part)
        sub-place  -> drop from the place stream: a county or town figure is not the
                      place's total, and filing it as one is what puts `"one"` against a
                      North Carolina truth of 123
        place      -> keep
        unclear    -> fall through to the ratio gate, which is why `unclear` is a real
                      class rather than a guess

    Returns ``(kept, moved, dropped, deferred)``; `deferred` are the `unclear` ones the
    caller should still pass through ``gate()``.
    """
    kept: dict[str, list] = {}
    moved: list = []
    dropped: list = []
    deferred: list = []
    for o in observations:
        key = str(o.get("event_key"))
        sc = str(o.get("scope") or "unclear").lower()
        if key not in states or sc not in SCOPE_CLASSES or sc == "unclear":
            deferred.append(o)
            kept.setdefault(key, []).append(o)
        elif sc == "national":
            moved.append(dict(o, event_key=aggregate_key, _from=key))
        elif sc == "sub-place":
            dropped.append(dict(o, _from=key))
        else:
            kept.setdefault(key, []).append(o)
    for o in moved:
        kept.setdefault(aggregate_key, []).append(o)
    return kept, moved, dropped, deferred


def scope_agreement(observations: list, states: dict, ratio: float, warmup: int,
                    reference: str = "aggregate") -> dict:
    """Do the EXTRACTED scope and the INFERRED (ratio) scope agree?

    This is the tunable signal to reach for, not a self-reported confidence. This
    architecture's confidence saturates -- every one of Helene's 106 `dead` observations
    carries exactly 1.000, contaminants included -- so a model-emitted `scope_confidence`
    would very likely be constant too. Agreement between two INDEPENDENT routes to the
    same field is checkable, and it degrades gracefully: high agreement means trust the
    extracted scope and skip the gate, low agreement means the corpus needs work before
    the field can be leaned on.

    Returns counts keyed ``"<extracted>/<inferred>"`` plus an ``agreement`` rate over the
    observations where both express an opinion.
    """
    gated, moved, dropped = gate(observations, ratio, warmup, states, reference)
    inferred = {}
    for o in moved:
        inferred[id(o.get("_orig", o))] = "national"
    for o in dropped:
        inferred[id(o.get("_orig", o))] = "drop"
    counts: dict[str, int] = {}
    agree = total = 0
    for o in observations:
        if str(o.get("event_key")) not in states:
            continue
        ex = str(o.get("scope") or "unclear").lower()
        # `gate` copies rejects, so match on value+time rather than identity.
        inf = "place"
        for m in moved:
            if m["t_hours"] == o["t_hours"] and m["value"] == o["value"]:
                inf = "national"
                break
        else:
            for dd in dropped:
                if dd["t_hours"] == o["t_hours"] and dd["value"] == o["value"]:
                    inf = "drop"
                    break
        counts[f"{ex}/{inf}"] = counts.get(f"{ex}/{inf}", 0) + 1
        if ex != "unclear":
            total += 1
            if (ex == "national" and inf == "national") or \
               (ex == "place" and inf == "place") or \
               (ex == "sub-place" and inf == "drop"):
                agree += 1
    counts["agreement"] = (agree / total) if total else float("nan")
    counts["_n_opinionated"] = total
    return counts


# --------------------------------------------------------------------------- #
# Viterbi scope decode
# --------------------------------------------------------------------------- #
OWN, AGG, REJ = 0, 1, 2
_STATE_NAME = {OWN: "own", AGG: "aggregate", REJ: "reject"}


def _emissions(v: float, m_own: float, natl: float, sigma: float, reject_cost: float,
               part_ratio: float = 2.0):
    """Log-likelihood of one reading under each hypothesis.

    ONE-SIDED for `own`, for the same reason the shipped gate and the Student-t model are:
    a rising toll may legitimately exceed the level established so far, so only a reading
    far BELOW that level is evidence against `own`. Exceeding the whole event's total is
    evidence against it in the other direction, and that side IS penalised.

    `reject` is the standard clutter model -- a flat density, so it wins exactly when both
    structured hypotheses have made the reading sufficiently improbable.
    """
    lv = math.log(max(v, 1.0))
    lo = math.log(max(m_own, 1.0))
    ln = math.log(max(natl, 1.0))
    pen_low = max(0.0, lo - lv)            # stale: below the level already established
    # A part is only credible as a part while it stays clear of the whole. The band top is
    # natl/part_ratio, the same place the shipped gate puts its keep/reroute boundary --
    # so this is a soft version of that rule, not a different one. Without it the one-sided
    # penalty scores a reading at 83% of the national total as a PERFECT part (measured on
    # the smoke case: 500 against a national 600 decoded as `own`).
    pen_high = max(0.0, lv - (ln - math.log(max(part_ratio, 1.0))))
    own = -(pen_low ** 2 + pen_high ** 2) / (2 * sigma ** 2)
    agg = -((lv - ln) ** 2) / (2 * sigma ** 2) if natl > 0 else -1e3
    return (own, agg, -reject_cost)


def _viterbi(rows, sigma, reject_cost, stay, part_ratio=2.0):
    """Decode the most likely state sequence. Returns a list of OWN/AGG/REJ."""
    n = len(rows)
    if not n:
        return []
    delta = [list(_emissions(*rows[0], sigma, reject_cost, part_ratio))]
    back = [[0, 0, 0]]
    for i in range(1, n):
        em = _emissions(*rows[i], sigma, reject_cost, part_ratio)
        cur, bk = [0.0] * 3, [0] * 3
        for s in range(3):
            best, arg = -1e18, 0
            for p in range(3):
                sc = delta[i - 1][p] + (stay if p == s else 0.0)
                if sc > best:
                    best, arg = sc, p
            cur[s] = best + em[s]
            bk[s] = arg
        delta.append(cur); back.append(bk)
    path = [max(range(3), key=lambda s: delta[-1][s])]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return path[::-1]


def viterbi_gate(observations: list, states: dict, reference: str = "aggregate",
                 sigma: float = 0.5, reject_cost: float = 2.0, stay: float = 0.5,
                 iters: int = 4, warmup: int = 0, part_ratio: float = 2.0):
    """Global scope decode. Same (kept, moved, dropped) contract as ``gate``.

    The shipped gate walks each stream in time order and commits per observation against
    a running reference. That is greedy, and it fails the way greedy fails: on Turkiye one
    large figure admitted early poisons the running maximum and every legitimate later
    reading looks stale (measured -- see two_sided_gate.txt). Viterbi decides the whole
    sequence jointly, so no single early reading can commit the rest.

    The `own` level is itself unknown, so this is hard-EM: decode, re-estimate the level
    from whatever the decode called `own`, repeat. `iters=1` is a single pass.

    The REJECT state is not optional. Measured, assignment headroom on Helene is ZERO --
    the two-way oracle scores exactly what the shipped gate does -- and the entire ~11.7
    death prize is in being able to say `none of these`.
    """
    kept: dict[str, list] = {}
    moved: list = []
    dropped: list = []
    by_key: dict[str, list] = {}
    for o in observations:
        by_key.setdefault(str(o.get("event_key")), []).append(o)

    for key, obs in by_key.items():
        obs = sorted(obs, key=lambda o: o["t_hours"])
        if key not in states:
            kept.setdefault(key, []).extend(obs)
            continue
        scale = reference_for(observations, key, states, reference)
        own_set = list(obs)                       # seed: assume everything is own
        path = [OWN] * len(obs)
        for _ in range(max(1, iters)):
            run = 0.0
            levels = []
            for o in obs:                          # causal running max over the OWN set
                levels.append(run)
                if o in own_set:
                    run = max(run, float(o["value"]))
            rows = [(float(o["value"]), levels[i], scale_at(scale, o["t_hours"]))
                    for i, o in enumerate(obs)]
            path = _viterbi(rows, sigma, reject_cost, stay, part_ratio)
            # NO warmup by default. The shipped gate needs one because it commits per
            # observation and must establish a scale before it can judge anything. Forcing
            # the first k readings to `own` here reintroduces exactly the greedy commitment
            # this decode exists to remove -- measured on Syria, whose first two readings
            # are contaminating Turkiye figures (9057, 17674 against a true peak of 5800).
            # Pinning them to `own` poisoned the level and the genuine 3317s were then
            # rejected as stale.
            for i in range(min(warmup, len(path))):
                path[i] = OWN
            new_own = [o for o, s in zip(obs, path) if s == OWN]
            if new_own == own_set:
                break
            own_set = new_own
        for o, s in zip(obs, path):
            if s == OWN:
                kept.setdefault(key, []).append(o)
            elif s == AGG:
                moved.append(dict(o, event_key="__aggregate__", _from=key))
            else:
                dropped.append(dict(o, _from=key, _viterbi=True))
    for o in moved:
        kept.setdefault("__aggregate__", []).append(o)
    return kept, moved, dropped
