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
import sys
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

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
    pairs = [(p, at(seq, t)) for p, t in zip(pred, grid)]
    pairs = [(p, g) for p, g in pairs if g is not None]
    if not pairs:
        return None
    vals = [g for _, g in pairs]
    rng = max(vals) - min(vals)
    err = sqrt(sum((p - g) ** 2 for p, g in pairs) / len(pairs))
    return err / rng if rng > 0 else None


def national_scale(observations: list, reference: str = "aggregate") -> list:
    """Running (t, largest-scope value) in time order -- what the gate judges against.

    Gating against a stream's OWN running scale fails on its early history, where a toll
    legitimately jumps 6 -> 25 faster than any ratio tolerates. It has to be judged against
    something larger.

    ``aggregate``   the ``__aggregate__`` stream. Correct where one exists: on Helene it is
                    the only stream whose scope is known right (nRMSE 0.402 vs Wikipedia's
                    Total).
    ``global-max``  the running maximum across ALL streams. Needed because Turkiye-Syria has
                    no aggregate stream at all, and its contaminant is not a national total
                    but the larger SIBLING -- Syria's stream carries Turkey's 41,000. This
                    reduces to the aggregate wherever an aggregate is the largest scope
                    reported, so it generalizes rather than replaces.
    """
    rows = observations if reference == "global-max" else [
        o for o in observations if str(o.get("event_key")) == "__aggregate__"]
    out, run = [], 0.0
    for o in sorted(rows, key=lambda o: o["t_hours"]):
        run = max(run, float(o["value"]))
        out.append((o["t_hours"], run))
    return out


def scale_at(scale: list, t: float) -> float:
    best = 0.0
    for tt, v in scale:
        if tt <= t:
            best = v
    return best


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
    scale = national_scale(observations, reference)
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
        out[state] = (nrmse(e, series[state], grid), len(rows))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="helene")
    ap.add_argument("--tracked", default=None)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--reference", choices=("aggregate", "global-max"), default=None)
    ap.add_argument("--mode", default="heuristic")
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
    print(f"[{args.dataset}] {len(obs)} 'dead' observations over {res['n_articles']} "
          f"articles, reference={reference}\n")

    cols = ("Total",) + tuple(cfg["places"])
    print(f"{'ratio':>7}{'moved':>7}{'drop':>6}" +
          "".join(f"{c.split()[-1][:6]:>9}" for c in cols) + f"{'mean':>9}")
    for ratio in (0.0, 4.0, 3.0, 2.5, 2.0, 1.5):
        kept, moved, dropped = gate(obs, ratio, args.warmup, states, reference)
        sc = score(kept, series, grid, states)
        cells = []
        for c in cols:
            v = sc.get(c, (None, 0))[0]
            cells.append(f"{v:>9.3f}" if v is not None else f"{'-':>9}")
        vals = [sc[p][0] for p in cfg["places"] if sc.get(p, (None,))[0] is not None]
        mean = sum(vals) / len(vals) if vals else float("nan")
        tag = "off" if ratio == 0.0 else f"{ratio:.1f}"
        print(f"{tag:>7}{len(moved):>7}{len(dropped):>6}" + "".join(cells) + f"{mean:>9.3f}")

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

    print(f"\n[detail at ratio=2.0] {len(moved)} rerouted, {len(dropped)} dropped:")
    for o in sorted(moved, key=lambda o: (o["_from"], o["t_hours"]))[:14]:
        print(f"    reroute {o['_from']:<22} value={int(o['value']):>6}  t={o['t_hours']:>7.1f}h")
    for o in sorted(dropped, key=lambda o: (o["_from"], o["t_hours"]))[:8]:
        print(f"    DROP    {o['_from']:<22} value={int(o['value']):>6}  t={o['t_hours']:>7.1f}h")


if __name__ == "__main__":
    main()
