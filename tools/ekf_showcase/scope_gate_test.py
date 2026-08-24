"""Can a magnitude gate tell a state's own toll from a larger-scope number filed under it?

Every state stream in the Helene run is contaminated, and always UPWARD: North Carolina
(true peak ~123) receives 200, 215, 227, 230 and 250; Florida (true peak 26) receives 150,
180, 230 and 300. Nothing ever leaks the other way. That one-directional signature is what a
larger scope leaking into a smaller one looks like, and it is what makes a gate feasible --
the separation is a factor of 2-10, not a few percent.

The gate needs no new model. Walking a stream in time order, the tracker's own running
estimate supplies the scale, and tolls start small, so a state pins its magnitude before the
national figures appear. An observation far above that scale is not a rival claim about the
state; it is an observation about a larger scope.

**Reclassify, do not discard.** A rejected figure is moved to ``__aggregate__``, where it is
a correct observation rather than a corrupting one. Discarding would throw away the national
signal that is the project's one honest measurement.

Two knobs, both reported across a sweep so the result is not one lucky setting:
  ratio   reject when value > ratio * running_estimate
  warmup  observations to accept unconditionally while the scale is established

    uv run python tools/ekf_showcase/scope_gate_test.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "datasets" / "disaster_streams"))
import evaluate as ekf  # noqa: E402

DATASETS = {
    "helene": {
        "onset": datetime(2024, 9, 26, 23, 10, tzinfo=timezone.utc),
        "tracked": "datasets/helene2024/_cache/tracked_rollup.json",
        "truth": "datasets/helene2024/ground_truth.json",
        "places": ("Florida", "Georgia", "South Carolina", "North Carolina",
                   "Tennessee", "Virginia"),
        # rollup has already collapsed the event type, so a key is a bare place
        "key_of": lambda name: name.lower(),
        "reference": "aggregate",
    },
    # Turkiye-Syria has NO aggregate stream, and its contaminant is the larger sibling
    # (Syria's stream carries Turkey's tolls), so it needs the global-max reference.
    "turkey": {
        "onset": datetime(2023, 2, 6, 1, 17, tzinfo=timezone.utc),
        "tracked": "datasets/turkey2023/_cache/tracked_perenvelope.json",
        "truth": "datasets/turkey2023/ground_truth.json",
        "places": ("turkey", "syria"),
        "key_of": lambda name: f"Earthquakes|{name}",
        "reference": "global-max",
    },
}


def truth(path: Path, onset: datetime) -> dict:
    """Per-place series plus Total. Helene ships a `deaths` map; Turkiye-Syria ships flat
    `turkey`/`syria` keys and no total, so the combined series is derived as their sum."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    series: dict[str, list] = {}
    for p in raw["points"]:
        ts = datetime.fromisoformat(p["snapshot"].replace("Z", "+00:00"))
        t = (ts - onset).total_seconds() / 3600.0
        if "deaths" in p:
            for name, v in p["deaths"].items():
                if not name.startswith("_"):
                    series.setdefault(name, []).append((t, float(v)))
        else:
            total = 0.0
            for name in ("turkey", "syria"):
                series.setdefault(name, []).append((t, float(p[name])))
                total += float(p[name])
            series.setdefault("Total", []).append((t, total))
    return series


def at(seq, t):
    best = None
    for tt, v in seq:
        if tt <= t:
            best = v
    return best


def nrmse(pred, seq, grid):
    r = errors(pred, seq, grid)
    return None if r is None else r[0]


def errors(pred, seq, grid):
    """(nRMSE, RMSE in deaths) -- both, deliberately.

    nRMSE divides by the stream's OWN range, so a place whose toll never moves far has a
    tiny denominator and can dominate a macro-average over streams. That is not
    hypothetical here: the same normalisation reversed the aggregate-constraint verdict in
    EKF_MHT_DESIGN 6.2, where Virginia (range 1->2) outvoted North Carolina (6->123).
    Every per-place mean in this file is such a macro-average, so the absolute column is
    what says whether a gain is deaths or arithmetic.
    """
    pairs = [(p, at(seq, t)) for p, t in zip(pred, grid)]
    pairs = [(p, g) for p, g in pairs if g is not None]
    if not pairs:
        return None
    vals = [g for _, g in pairs]
    rng = max(vals) - min(vals)
    err = sqrt(sum((p - g) ** 2 for p, g in pairs) / len(pairs))
    return (err / rng if rng > 0 else None), err


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


def random_control(observations: list, n_removed: int, trials: int, series: dict,
                   grid: list, states: dict, seed: int = 0) -> float:
    """Remove the SAME NUMBER of state observations at random, and score that instead.

    The control the result needs: gating removes 29 of 106 observations, and a filter fed
    fewer observations can look better simply by drifting less. If random removal buys the
    same improvement then the gate is not selecting anything -- it is just thinning.
    """
    import random
    rng = random.Random(seed)
    state_obs = [o for o in observations if str(o.get("event_key")) in states]
    means = []
    for _ in range(trials):
        drop = set(map(id, rng.sample(state_obs, min(n_removed, len(state_obs)))))
        kept: dict[str, list] = {}
        for o in observations:
            if id(o) not in drop:
                kept.setdefault(str(o.get("event_key")), []).append(o)
        sc = score(kept, series, grid, states)
        vals = [sc[s][0] for s in states.values() if sc.get(s, (None,))[0] is not None]
        if vals:
            means.append(sum(vals) / len(vals))
    return sum(means) / len(means) if means else float("nan")


def oracle_gate(observations: list, states: dict, series: dict) -> dict:
    """The best any hard association could do: assign each figure to the scope it FITS.

    Uses ground truth, so it is not a method -- it is the CEILING. Every observation is
    assigned to whichever scope (its own place, or Total) its value is closest to at that
    time, in relative terms. No association scheme, MHT included, can beat a perfect
    assignment by much, so the gap between this and the shipped gate is the entire prize
    available to better association. If that gap is small, MHT is not the bottleneck.
    """
    kept: dict[str, list] = {}
    for o in observations:
        key = str(o.get("event_key"))
        place = states.get(key)
        if place is None or place not in series:
            kept.setdefault(key, []).append(o)
            continue
        v = float(o["value"])
        here = at(series[place], o["t_hours"])
        whole = at(series.get("Total", []), o["t_hours"])
        def err(g):
            return abs(v - g) / max(g, 1.0) if g is not None else float("inf")
        kept.setdefault(key if err(here) <= err(whole) else "__aggregate__",
                        []).append(o)
    return kept


def oracle_gate_three_way(observations: list, states: dict, series: dict,
                          tol: float) -> dict:
    """The ceiling for an association layer that can also say NONE OF THE ABOVE.

    ``oracle_gate`` is two-way -- every observation goes to its own place or to Total --
    so it has no home for a figure belonging to no Helene scope at all, and it scores
    Katrina's 1,400 as badly as the shipped gate does. That understates the prize for a
    real attribution layer, because MHT's track birth/death IS a null hypothesis: an
    unassignable observation starts its own track and leaves these streams. It also
    explains why the shipped gate BEATS the two-way oracle on Florida and South Carolina --
    the gate can drop and the oracle cannot.

    This oracle gets the third option. An observation is dropped when its relative error
    against BOTH candidate scopes exceeds ``tol``; otherwise it goes to the better one.
    Uses ground truth, so it is a ceiling and not a method. ``tol`` is swept by the caller
    rather than fixed, because there is no principled value and one lucky setting is not
    a result.
    """
    kept: dict[str, list] = {}
    for o in observations:
        key = str(o.get("event_key"))
        place = states.get(key)
        if place is None or place not in series:
            kept.setdefault(key, []).append(o)
            continue
        v = float(o["value"])
        here = at(series[place], o["t_hours"])
        whole = at(series.get("Total", []), o["t_hours"])

        def err(g):
            return abs(v - g) / max(g, 1.0) if g is not None else float("inf")

        e_here, e_whole = err(here), err(whole)
        if min(e_here, e_whole) > tol:
            continue                      # belongs to no scope in this event -- reject
        kept.setdefault(key if e_here <= e_whole else "__aggregate__", []).append(o)
    return kept


def tail_cut(observations: list, k: float, warmup: int = 8) -> float | None:
    """The upper-tail cut, DERIVED from the event's own observations. No ground truth.

    An absolute ceiling has to be told the event's scale, and gets it from the answer: 2,000
    is defensible on Helene only because Helene killed ~230. Held out on Türkiye that same
    ceiling deletes 80 of 91 observations including Turkey's true 41,000s, empties Syria's
    stream, and the reported mean *improves* because it is then an average over one stream
    instead of two. A ceiling is not a method; it is the answer smuggled in.

    This is scale-free instead. Three choices, each measured rather than assumed:

    ``log10``       values span 1 .. 129,933, and the false positives are *orders of
                    magnitude* out, not a few sigma out.
    ``median/MAD``  robust to the outliers being hunted. Measured against mean/stdev on this
                    data the two are nearly identical -- the log transform does most of the
                    work -- but MAD cannot be masked in principle and costs nothing.
    ``one-sided``   contamination here is documented as one-directional, and a toll of 1 or 2
                    is legitimate, so trimming the low tail would delete real readings.

    Derived cuts: Helene 516, Türkiye 47,622. Both are above their event's true peak and
    below its junk, and neither was chosen.
    """
    if k <= 0 or len(observations) < warmup:
        return None
    lv = [math.log10(max(float(o["value"]), 1.0)) for o in observations]
    centre = median(lv)
    mad = median([abs(x - centre) for x in lv])
    if mad <= 0:
        return None
    return 10 ** (centre + k * 1.4826 * mad)      # 1.4826: sigma-consistent for normal data


def tail_filter(observations: list, k: float, warmup: int = 8) -> tuple[list, list]:
    """Reject the upper tail at ``k`` robust sigmas, pooled over the whole event.

    **Pool over every observation, not over the ones already accepted.** Recomputing on the
    accepted set is self-reinforcing and collapses: a death toll starts small, so the early
    cut is small, and it then rejects the legitimate growth that follows -- measured, 88 of
    106 rejected on Helene with the cut stuck at 3. That is the same self-reference that
    defeated M5 track birth. Pooled over all observations the streaming estimate converges
    to the batch one exactly (Helene 3 -> 84 -> 272 -> 351, batch 351).
    """
    cut = tail_cut(observations, k, warmup)
    if cut is None:
        return list(observations), []
    keep, drop = [], []
    for o in observations:
        (drop if float(o["value"]) > cut else keep).append(o)
    return keep, drop


def plausibility_filter(observations: list, ceiling: float) -> tuple[list, list]:
    """Drop observations above the largest credible toll FOR THIS EVENT, whatever the stream.

    Prior knowledge, declared per event, not fitted: Hurricane Helene killed on the order of
    230 people, so a five- or six-figure "death toll" in any of its streams is not a casualty
    figure at all. This is the cheapest response to the false-positive audit -- 129,933 is
    FEMA flood-insurance policies, 94,000 is Asheville's population, 15,000 is wellness checks
    -- and it needs no model change.

    Two things it CANNOT do, both measured in `muting_arm_results/FALSE_POSITIVES.md`:

    - It cannot see 1,500 active-duty troops or 8,000 power crews. Those are counts of living
      people in the affected area, and they are plausible MAGNITUDES for a casualty figure;
      only casualty-role semantics separate them.
    - Set low enough to catch Katrina's 1,400 it is no longer a plausibility test, it is the
      magnitude gate again, rejecting a cross-event toll for being large rather than for
      belonging to another storm. The sweep below shows exactly where that line is.

    Returns ``(kept, dropped)``. ``ceiling <= 0`` disables it.
    """
    if ceiling <= 0:
        return list(observations), []
    kept, dropped = [], []
    for o in observations:
        (dropped if float(o["value"]) > ceiling else kept).append(o)
    return kept, dropped


def score(kept: dict, series: dict, grid: list, states: dict,
          role: str = "dead") -> dict:
    """Per-state nRMSE plus the national stream, using the shipped estimators."""
    out = {}
    for key, obs in kept.items():
        state = states.get(key) or ("Total" if key == "__aggregate__" else None)
        if state is None or state not in series:
            continue
        rows = [o for o in obs if o["role"] == role]
        if not rows:
            continue
        e = ekf.est_ekf(rows, grid, role)
        r = errors(e, series[state], grid)
        out[state] = (None, len(rows)) if r is None else (r[0], len(rows), r[1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="helene")
    ap.add_argument("--tracked", default=None)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--reference", choices=("aggregate", "global-max", "implied"),
                    default=None)
    ap.add_argument("--mode", default="heuristic")
    ap.add_argument("--max-plausible", type=float, default=0.0,
                    help="drop any observation above this value before gating -- the largest "
                         "credible toll for this event. 0 disables. Swept in the report.")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    states = {cfg["key_of"](p): p for p in cfg["places"]}
    reference = args.reference or cfg["reference"]
    res = json.loads(Path(args.tracked or cfg["tracked"]).read_text(encoding="utf-8"))
    series = truth(Path(args.truth or cfg["truth"]), cfg["onset"])
    grid = res["grid"]
    obs = [o for a in res["articles"] for o in a["observations"]
           if o["mode"] == args.mode and o["role"] == "dead"]
    obs_all = obs
    obs, culled = plausibility_filter(obs, args.max_plausible)
    print(f"[{args.dataset}] {len(obs)} 'dead' observations over {res['n_articles']} "
          f"articles, reference={reference}"
          + (f", plausibility ceiling {args.max_plausible:.0f} dropped {len(culled)}"
             if args.max_plausible > 0 else "") + "\n")

    cols = ("Total",) + tuple(cfg["places"])
    print(f"{'ratio':>7}{'moved':>7}{'drop':>6}" +
          "".join(f"{c.split()[-1][:6]:>9}" for c in cols) +
          f"{'mean':>9} | {'deaths':>8}{'Total_d':>9}")
    print(f"{'':>20}{'(per-place nRMSE, then the same streams in DEATHS)':<60}")
    for ratio in (0.0, 4.0, 3.0, 2.5, 2.0, 1.5):
        kept, moved, dropped = gate(obs, ratio, args.warmup, states, reference)
        sc = score(kept, series, grid, states)
        cells = []
        for c in cols:
            v = sc.get(c, (None, 0))[0]
            cells.append(f"{v:>9.3f}" if v is not None else f"{'-':>9}")
        vals = [sc[p][0] for p in cfg["places"] if sc.get(p, (None,))[0] is not None]
        mean = sum(vals) / len(vals) if vals else float("nan")
        # Same streams, absolute. A macro-averaged nRMSE hides which places the gain is in.
        avals = [sc[p][2] for p in cfg["places"]
                 if len(sc.get(p, ())) > 2 and sc[p][2] is not None]
        amean = sum(avals) / len(avals) if avals else float("nan")
        tot = sc.get("Total", (None, 0, None))
        atot = tot[2] if len(tot) > 2 and tot[2] is not None else float("nan")
        tag = "off" if ratio == 0.0 else f"{ratio:.1f}"
        print(f"{tag:>7}{len(moved):>7}{len(dropped):>6}" + "".join(cells) +
              f"{mean:>9.3f} | {amean:>8.1f}{atot:>9.1f}")

    # Where the residual actually sits, in deaths, at the shipped setting. The macro
    # -averaged nRMSE above cannot show this: a place with a small range dominates it.
    kept2, _, _ = gate(obs, 2.0, args.warmup, states, reference)
    sc2 = score(kept2, series, grid, states)
    print(f"\n[residual @ratio 2.0, in DEATHS]  {'stream':<18}{'RMSE':>8}{'nRMSE':>8}{'n_obs':>7}")
    for c in cols:
        r = sc2.get(c)
        if r and len(r) > 2 and r[2] is not None:
            print(f"{'':<34}{c:<18}{r[2]:>8.1f}{r[0]:>8.3f}{r[1]:>7}")

    # ratio 2.0 is the setting chosen on Helene; it is NOT retuned per dataset.
    kept, moved, dropped = gate(obs, 2.0, args.warmup, states, reference)
    n = len(moved) + len(dropped)
    ctrl = random_control(obs, n, 40, series, grid, states)
    sc = score(kept, series, grid, states)
    vals = [sc[p][0] for p in cfg["places"] if sc.get(p, (None,))[0] is not None]
    gated = sum(vals) / len(vals) if vals else float("nan")
    print(f"\n[control] removing {n} per-place observations AT RANDOM (40 trials): "
          f"mean {ctrl:.3f}")
    print(f"[control] the gate removing the same number: {gated:.3f}  "
          f"({'gate is selecting' if gated < ctrl else 'NO BETTER THAN THINNING'})")

    orc = score(oracle_gate(obs, states, series), series, grid, states)
    ovals = [orc[p][0] for p in cfg["places"] if orc.get(p, (None,))[0] is not None]
    omean = sum(ovals) / len(ovals) if ovals else float("nan")
    print(f"\n[ORACLE] perfect hard association (uses ground truth -- a CEILING, not a "
          f"method): per-place mean {omean:.3f}")
    print(f"[ORACLE] shipped gate {gated:.3f} vs ceiling {omean:.3f} -> "
          f"headroom for better association = {gated - omean:+.3f}")
    for c in ("Total",) + tuple(cfg["places"]):
        v = orc.get(c, (None, 0))[0]
        if v is not None:
            print(f"           {c:<16}{v:>8.3f}")

    # Baseline stream count: a filter that EMPTIES a stream removes it from the mean, and
    # the score then "improves" for the wrong reason. Flagged in both sweeps below.
    base = score(gate(obs_all, 0.0, args.warmup, states, reference)[0], series, grid, states)
    base_streams = sum(1 for pl in cfg["places"] if base.get(pl, (None,))[0] is not None)

    print("\n[TAIL CUT] scale-free, DERIVED from the event -- median + k*MAD on log10, "
          "upper tail only:")
    print(f"{'k':>5}{'cut':>10}{'dropped':>9}{'kept':>7}{'streams':>9}{'ungated':>10}"
          f"{'gated@2.0':>11}")
    for k in (0.5, 1.0, 1.5, 2.0, 3.0):
        sub, cut = tail_filter(obs_all, k)
        thr = tail_cut(obs_all, k)
        ung = score(gate(sub, 0.0, args.warmup, states, reference)[0], series, grid, states)
        gat = score(gate(sub, 2.0, args.warmup, states, reference)[0], series, grid, states)
        def _mk(sc):
            v = [sc[pl][0] for pl in cfg["places"] if sc.get(pl, (None,))[0] is not None]
            return (sum(v) / len(v) if v else float("nan")), len(v)
        mu, ns = _mk(ung)
        mg, _ = _mk(gat)
        flag = "" if ns == base_streams else f"  <-- {base_streams - ns} STREAM(S) LOST"
        print(f"{k:>5.1f}{thr or 0:>10.0f}{len(cut):>9}{len(sub):>7}{ns:>9}{mu:>10.3f}"
              f"{mg:>11.3f}{flag}")

    print("\n[PLAUSIBILITY] a hand-set per-event ceiling, for comparison -- it needs to be "
          "told\n               the event's scale, and gets it from the answer:")
    print(f"{'ceiling':>9}{'dropped':>9}{'kept':>7}{'streams':>9}{'ungated':>10}"
          f"{'gated@2.0':>11}")
    for ceil in (0.0, 20000.0, 5000.0, 2000.0, 1000.0, 500.0, 250.0):
        sub, cut = plausibility_filter(obs_all, ceil)
        ung = score(gate(sub, 0.0, args.warmup, states, reference)[0], series, grid, states)
        gat = score(gate(sub, 2.0, args.warmup, states, reference)[0], series, grid, states)
        def _m(sc):
            v = [sc[p][0] for p in cfg["places"] if sc.get(p, (None,))[0] is not None]
            return sum(v) / len(v) if v else float("nan")
        # Thinning control, and it is not optional here. On Turkiye the Helene-tuned
        # ceiling deletes 80 of 91 observations INCLUDING Turkey's true 41,000s, and the
        # score still IMPROVES -- nRMSE on a near-empty stream looks good. Without this
        # column the sweep would recommend destroying the event.
        # The mean is only comparable across rows if it is a mean over the SAME streams.
        # A ceiling that empties a stream removes it from the average and the score
        # "improves" -- on Turkiye, Helene's ceiling of 2,000 deletes Syria entirely and
        # every one of Turkey's true 41,000s, and the reported mean falls from 1.815 to
        # 0.703. The random-removal control does NOT catch this; it reports "selecting".
        n_streams = sum(1 for pl in cfg["places"] if ung.get(pl, (None,))[0] is not None)
        tag = "off" if ceil == 0 else f"{ceil:.0f}"
        kept_n = len(obs_all) - len(cut)
        flag = "" if n_streams == base_streams else f"  <-- {base_streams - n_streams} STREAM(S) LOST, mean not comparable"
        print(f"{tag:>9}{len(cut):>9}{kept_n:>7}{n_streams:>9}{_m(ung):>10.3f}"
              f"{_m(gat):>11.3f}{flag}")

    print("\n[ORACLE-3] adding a REJECT option -- the ceiling for association that can say "
          "'none of these':")
    print(f"{'tol':>7}{'kept':>7}" + "".join(f"{c.split()[-1][:6]:>9}"
                                             for c in ("Total",) + tuple(cfg["places"]))
          + f"{'mean':>9}")
    for tol in (2.0, 1.0, 0.5, 0.25):
        k3 = oracle_gate_three_way(obs, states, series, tol)
        s3 = score(k3, series, grid, states)
        n3 = sum(len(v) for v in k3.values())
        cells = []
        for c in ("Total",) + tuple(cfg["places"]):
            v = s3.get(c, (None, 0))[0]
            cells.append(f"{v:>9.3f}" if v is not None else f"{'-':>9}")
        v3 = [s3[p][0] for p in cfg["places"] if s3.get(p, (None,))[0] is not None]
        m3 = sum(v3) / len(v3) if v3 else float("nan")
        print(f"{tol:>7.2f}{n3:>7}" + "".join(cells) + f"{m3:>9.3f}")
    print(f"[ORACLE-3] two-way oracle {omean:.3f} -- the gap between these prices a null "
          f"hypothesis (MHT track birth/death), which the two-way oracle cannot express")

    print(f"\n[detail at ratio=2.0] {len(moved)} rerouted, {len(dropped)} dropped:")
    for o in sorted(moved, key=lambda o: (o["_from"], o["t_hours"]))[:14]:
        print(f"    reroute {o['_from']:<22} value={int(o['value']):>6}  t={o['t_hours']:>7.1f}h")
    for o in sorted(dropped, key=lambda o: (o["_from"], o["t_hours"]))[:8]:
        print(f"    DROP    {o['_from']:<22} value={int(o['value']):>6}  t={o['t_hours']:>7.1f}h")


if __name__ == "__main__":
    main()
