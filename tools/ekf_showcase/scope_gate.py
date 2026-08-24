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
         states: dict, reference: str = "aggregate"):
    """Classify each state observation as its own toll, the national total, or neither.

    Three outcomes, not two. The earlier two-way version rerouted every reject to
    ``__aggregate__`` and that is what destroyed the national stream: 1400 filed under
    North Carolina is not a national total, it is not a casualty count at all, and moving
    it into the aggregate poisoned the one measurement that worked.

    ``keep``    below ``ratio`` of the running national total -- plausibly the state's own
    ``reroute`` within [1/ratio, ratio] of the national total -- it IS the national figure
    ``drop``    above the national total -- no scope in this event can exceed the whole
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
        for i, o in enumerate(obs):
            v, natl = float(o["value"]), scale_at(scale, o["t_hours"])
            if ratio <= 0 or i < warmup or natl <= 0 or v < natl / ratio:
                kept.setdefault(key, []).append(o)
            elif v <= natl * ratio:
                moved.append(dict(o, event_key="__aggregate__", _from=key))
            else:
                dropped.append(dict(o, _from=key))
    for o in moved:
        kept.setdefault("__aggregate__", []).append(o)
    return kept, moved, dropped
