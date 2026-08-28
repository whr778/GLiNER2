"""Track birth as a null hypothesis: the smallest piece of MHT, priced before it was built.

Why this and not the whole subsystem. Perfect hard association was measured at +0.055 over
the shipped scope gate, which priced MHT out. That oracle is TWO-WAY -- every observation
goes to its own place or to the national total -- so a figure belonging to no Helene scope
has no correct home and the oracle scores Katrina's 1,400 as badly as the gate does. Adding
a reject option moves the ceiling 0.537 -> 0.480, so the real headroom is 0.111 (18.8%),
double the number MHT was rejected on. Most of that prize is the reject option itself, which
is M5 track birth and not the cost matrix, so M5 is what this builds.

How it differs from the shipped scope gate, which is the point:

    scope gate   value > ratio * running_max(aggregate)      magnitude, one reference,
                                                             streams filtered independently
    this         (z - mu)^2 / S  against every candidate     the filter's OWN innovation,
                 track, reject when it gates out of ALL      tracks advanced JOINTLY in time

The magnitude test cannot see a contaminant that is the right size -- Tennessee receives 32,
32, 32, 36 and 50 against a truth of 18, too large for the state and too small to look
national, and the gate keeps every one. An innovation test can, because 32 against a track
sitting at 18 with a tight covariance is a large NIS regardless of how 32 compares to the
national total.

Three outcomes per observation, and the third is the new one:

    assign    smallest normalized innovation among candidate tracks, inside the gate
    reroute   the aggregate track fits better than the observation's own place
    BIRTH     it gates out of EVERY candidate -- it belongs to no track here, so it starts
              its own and leaves these streams

Association is nearest-neighbour, not deferred: one hard choice per observation, taken in
time order. Deferred assignment (M4) keeps rival hypotheses alive and is NOT built here --
this measures how much of the 0.111 the null hypothesis alone recovers, so that M4 is priced
on what it adds rather than on what the two together are worth.

RESULT, 2026-08-19: NEGATIVE. Innovation gating with track birth does not beat the fixed
magnitude ratio it was meant to replace, at any setting swept.

    arm                                          Total   per-place   (sigma 4.0)
    no gate                                      0.402       5.247
    symmetric birth, own+aggregate               0.555       1.059   q_rel 0.20 (the filter's)
    symmetric birth, tuned                       2.115       0.636   q_rel 2.00
    one-sided birth, tuned                       2.115       0.608   q_rel 2.00
    aggregate-only reference                     0.387       0.624   q_rel 0.20
    SHIPPED magnitude scope gate                 0.316       0.591
    three-way oracle (ground truth)              0.308       0.480

Read the PAIR, never the per-place number alone. The two tuned arms buy their per-place
improvement by DUMPING JUNK INTO THE AGGREGATE -- Total 2.115 against the gate's 0.316, a
6.7x degradation of the one stream this project calls its honest measurement. That is the
failure the shipped gate's three-outcome design was invented to prevent: its docstring
records that an earlier two-way version rerouted every reject to ``__aggregate__`` and
destroyed the national stream. This associator reproduced it.

Two causes, both measured rather than argued.

1. **Judging a stream against its own track is CIRCULAR.** Every contaminant the track
   accepts moves the reference the next test uses. Removing the self-reference -- testing
   only against the aggregate -- takes 1.059 -> 0.624 at the filter's native dynamics, and
   that single change is worth more than every other knob combined. This is the same failure
   the implied-maximum reference hit on Turkiye, where Turkey was judged against a reference
   Turkey itself defines, so it is now two independent mechanisms defeated by one cause.

2. **The innovation is not informative about scope on a rising toll.** Even with the
   self-reference removed it loses to a fixed ratio. At the filter's own q_rel = 0.20 the
   tracks are far too tight to admit real growth -- Georgia keeps only [2, 3] against a truth
   peak of 34 -- and the sweep has to push q_rel to 2.00 before real rises survive, at which
   point the track constrains so little that only 1-4 observations are ever born. Birth is
   never the lever; the rerouting is.

**What this does NOT refute, and the argument is now stronger than before.** Deferred
assignment (M4) is the one piece that addresses cause (1) directly: hard assignment commits
early and poisons its own reference, which is exactly what keeping rival hypotheses alive
exists to prevent. That was previously a design preference. It is now the specific mechanism
implicated by a measurement.

**SUPERSEDED 2026-08-25 -- the paragraph above is the conclusion as it stood on 2026-08-19,
kept because the diagnosis is still right and only the remedy changed.** The global decode
over {own, aggregate, reject} (M4', `scope_gate.hmm_gate`) was built six days later and
SHIPS: pooled RMSE in deaths Helene -29.4%, Turkiye -7.6%, Aegean -78.8%. It takes most of
the +0.111 without a hypothesis tree, because that headroom splits almost evenly and the
halves are not equally hard -- 0.591 -> 0.537 is reassignment (+0.055, mostly Tennessee) and
0.537 -> 0.480 is the reject option (+0.057). A tree buys the reassignment half;
PIPELINES.md's verdict on it is "Hungarian assignment solves the half worth nothing."

So cause (1) is unchanged and still real: hard assignment does poison its own reference. What
changed is that deferring the decision is no longer the cheapest way to stop it. Read this
docstring as the record of an argument, not as a live recommendation to build M4.

    uv run python tools/ekf_showcase/mht_associate.py
    uv run python tools/ekf_showcase/mht_associate.py --birth one-sided --q-rel 2.0
    uv run python tools/ekf_showcase/mht_associate.py --candidates aggregate-only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "datasets" / "disaster_streams"))
sys.path.insert(0, str(REPO / "tools" / "ekf_showcase"))
import evaluate as ekf  # noqa: E402
from scope_gate_test import DATASETS, at, nrmse, score, truth  # noqa: E402

Q_REL = 0.20        # process noise, matching est_ekf_rise
INIT_REL = 0.40     # initial state uncertainty, matching est_ekf_rise


class Track:
    """One stream's filter state, advanced on demand rather than over a fixed grid.

    Association needs the predicted state at an arbitrary observation time, which the
    grid-based estimators in evaluate.py do not expose -- they consume a whole stream and
    return a series. Same dynamics and same measurement noise, driven observation by
    observation so several tracks can be advanced together.
    """

    def __init__(self) -> None:
        self.mu: float | None = None
        self.P: float = 0.0
        self.t: float = 0.0

    def predict(self, t: float) -> tuple[float, float]:
        dt = max(0.0, (t - self.t) / 24.0)
        return self.mu, self.P + (Q_REL * max(self.mu, 1.0) * max(dt, 1e-3)) ** 2

    def nis(self, o: dict) -> float:
        """Normalized innovation squared -- how surprising this reading is to this track."""
        mu, P = self.predict(o["t_hours"])
        S = P + ekf._R_at(o, mu)
        return (float(o["value"]) - mu) ** 2 / S

    def update(self, o: dict) -> None:
        z = float(o["value"])
        if self.mu is None:
            self.mu, self.P, self.t = z, (INIT_REL * max(z, 1.0)) ** 2, o["t_hours"]
            return
        mu, P = self.predict(o["t_hours"])
        self.t = o["t_hours"]
        if o["qualifier"] == "at_least" and z <= mu:      # uninformative lower bound
            self.mu, self.P = mu, P
            return
        S = P + ekf._R_at(o, mu)
        K = P / S
        self.mu, self.P = mu + K * (z - mu), (1 - K) * P


def _rejects(candidates: dict, best: str, o: dict, gate_sigma: float, birth: str) -> bool:
    """Does this observation gate out of every candidate track?

    ``symmetric``  the plain chi-square test. Rejects readings that are surprisingly LOW as
                   well as high, which on a monotone rising toll throws away real growth.
    ``one-sided``  born only when ABOVE every candidate. Contamination on this feed is
                   documented as one-directional -- nothing ever leaks downward -- so a low
                   reading is a bad reading, not evidence of another event. Better than
                   symmetric (0.608 against 0.636) and still short of the ratio it replaces.
    """
    if birth != "one-sided":
        return candidates[best].nis(o) > gate_sigma ** 2
    z = float(o["value"])
    for track in candidates.values():
        mu, P = track.predict(o["t_hours"])
        if (z - mu) <= gate_sigma * math.sqrt(P + ekf._R_at(o, mu)):
            return False
    return True


def associate(observations: list, states: dict, gate_sigma: float, warmup: int = 2,
              birth: str = "symmetric", candidates_mode: str = "own+aggregate"):
    """Assign each observation to a track, or let it be born into its own.

    Tracks advance JOINTLY in time order, which is what makes this association rather than
    per-stream filtering: an observation rerouted to the aggregate updates the aggregate,
    and the next observation is tested against that updated state.
    """
    tracks: dict[str, Track] = {}
    kept: dict[str, list] = {}
    born: list = []
    seen: dict[str, int] = {}

    for o in sorted(observations, key=lambda o: o["t_hours"]):
        key = str(o.get("event_key"))
        if key not in states:                     # aggregate and ungated streams pass through
            tracks.setdefault(key, Track()).update(o)
            kept.setdefault(key, []).append(o)
            continue

        seen[key] = seen.get(key, 0) + 1
        pool = ("__aggregate__",) if candidates_mode == "aggregate-only" else (key, "__aggregate__")
        candidates = {k: tracks[k] for k in pool
                      if k in tracks and tracks[k].mu is not None}

        if seen[key] <= warmup or not candidates:
            # No track to test against yet, or still establishing this stream's scale.
            tracks.setdefault(key, Track()).update(o)
            kept.setdefault(key, []).append(o)
            continue

        if candidates_mode == "aggregate-only":
            # The magnitude gate's SHAPE -- judged only against the larger scope -- with the
            # filter's innovation in place of a fixed ratio. Routing has to be explicit here:
            # the aggregate is the only candidate, so an argmin over candidates would send
            # every observation to it.
            agg = candidates["__aggregate__"]
            mu, _ = agg.predict(o["t_hours"])
            if float(o["value"]) > mu and agg.nis(o) > gate_sigma ** 2:
                born.append(dict(o, _from=key))
                continue
            best = key if float(o["value"]) < mu else "__aggregate__"
        else:
            best = min(candidates, key=lambda k: candidates[k].nis(o))
            if _rejects(candidates, best, o, gate_sigma, birth):
                born.append(dict(o, _from=key))   # gates out of every track: a new one
                continue
        tracks.setdefault(best, Track()).update(o)
        kept.setdefault(best, []).append(dict(o, event_key=best,
                                              **({"_from": key} if best != key else {})))
    return kept, born


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="helene")
    ap.add_argument("--mode", default="heuristic")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--birth", choices=("symmetric", "one-sided"), default="symmetric")
    ap.add_argument("--candidates", choices=("own+aggregate", "aggregate-only"),
                    default="own+aggregate")
    ap.add_argument("--q-rel", type=float, default=0.20,
                    help="process noise. The filter's own value is 0.20; the sweep has to "
                         "reach 2.00 before real growth survives the gate, which is itself "
                         "the finding.")
    args = ap.parse_args()

    global Q_REL
    Q_REL = args.q_rel
    cfg = DATASETS[args.dataset]
    states = {cfg["key_of"](p): p for p in cfg["places"]}
    res = json.loads((REPO / cfg["tracked"]).read_text(encoding="utf-8"))
    series = truth(REPO / cfg["truth"], cfg["onset"])
    grid = res["grid"]
    obs = [o for a in res["articles"] for o in a["observations"]
           if o["mode"] == args.mode and o["role"] == "dead"]
    print(f"[{args.dataset}] {len(obs)} 'dead' observations over {res['n_articles']} "
          f"articles, birth={args.birth}, candidates={args.candidates}, "
          f"q_rel={Q_REL}\n")

    cols = ("Total",) + tuple(cfg["places"])
    print(f"{'sigma':>7}{'born':>6}{'moved':>7}" +
          "".join(f"{c.split()[-1][:6]:>9}" for c in cols) + f"{'mean':>9}")
    for sigma in (6.0, 5.0, 4.0, 3.0, 2.5, 2.0):
        kept, born = associate(obs, states, sigma, args.warmup,
                               birth=args.birth, candidates_mode=args.candidates)
        sc = score(kept, series, grid, states)
        moved = sum(1 for v in kept.values() for o in v if o.get("_from"))
        cells = []
        for c in cols:
            v = sc.get(c, (None, 0))[0]
            cells.append(f"{v:>9.3f}" if v is not None else f"{'-':>9}")
        vals = [sc[p][0] for p in cfg["places"] if sc.get(p, (None,))[0] is not None]
        mean = sum(vals) / len(vals) if vals else float("nan")
        print(f"{sigma:>7.1f}{len(born):>6}{moved:>7}" + "".join(cells) + f"{mean:>9.3f}")

    print("\n  reference points on this feed (Total / per-place):")
    print("    no gate                                  0.402 / 5.247")
    print("    shipped magnitude scope gate             0.316 / 0.591")
    print("    two-way oracle (no reject, ground truth)   --   / 0.537")
    print("    three-way oracle (reject, ground truth)  0.308 / 0.480")
    print("  Read the PAIR: a per-place gain bought by poisoning Total is not a gain.")


if __name__ == "__main__":
    main()
