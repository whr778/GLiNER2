"""Synthetic disaster event-stream generator for the EKF/MHT tracker.

Each stream = one disaster whose ground-truth state evolves (dead/injured approach an
asymptote, missing decays, salience/attention decays), observed by a sequence of noisy,
*hedged* reports (the tracker's input). Parametric + deterministic (seeded) -> free,
with exact ground truth. Parameters are scaled to look like real disasters (the
Venezuela 2026 toll ran ~920 -> ~1,719 -> ~6,000+). See
tools/events_working_papers/EKF_MHT_DESIGN.md sec 11.

  uv run python datasets/disaster_streams/generate.py \
      --out datasets/disaster_streams --n-train 400 --n-val 60 --n-test 60 \
      --seed 42 [--text template]

Outputs per split: observations.jsonl (tracker input) + trajectory.jsonl (eval target),
plus config.json at the root.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROLES = ("dead", "injured", "missing")
SOURCES = ("official", "major_outlet", "preliminary")
# source -> (multiplicative-noise sigma, early under-report bias factor)
SOURCE_NOISE = {"official": (0.05, 0.97), "major_outlet": (0.12, 1.0), "preliminary": (0.25, 1.0)}
GRID_HOURS = 6  # ground-truth trajectory sampling interval

# Regime knobs: rate0 (reports/day), source weights (official, major, preliminary),
# qualifier weights for rising roles (point, at_least, about, interval) and for
# missing (feared, about, interval, at_least). "hard" = sparse + unreliable + hedged.
_NORMAL = dict(rate0=(8.0, 30.0), src_w=(0.50, 0.35, 0.15),
               qual_rise=(0.40, 0.25, 0.20, 0.15), qual_miss=(0.40, 0.30, 0.20, 0.10))
_HARD_KNOBS = dict(rate0=(2.0, 6.0), src_w=(0.20, 0.30, 0.50),
                   qual_rise=(0.15, 0.40, 0.15, 0.30), qual_miss=(0.50, 0.15, 0.25, 0.10))
# "hard" flips all three knobs; the single-knob regimes isolate each cause. Trajectory
# params draw before rate0, so ALL regimes share byte-identical trajectories (paired eval).
REGIMES = {
    "normal": dict(_NORMAL),
    "hard":   dict(_HARD_KNOBS),
    "sparse":     {**_NORMAL, "rate0": _HARD_KNOBS["rate0"]},
    "unreliable": {**_NORMAL, "src_w": _HARD_KNOBS["src_w"]},
    "censored":   {**_NORMAL, "qual_rise": _HARD_KNOBS["qual_rise"],
                              "qual_miss": _HARD_KNOBS["qual_miss"]},
}
REGIME = REGIMES["normal"]  # set in main()


@dataclass
class StreamParams:
    d_final: int   # settled dead toll (asymptote)
    i_final: int   # settled injured
    m0: int        # initial missing
    beta: float    # dead/injured approach rate (per day)
    gamma: float   # missing decay rate (per day)
    lam: float     # salience/attention decay (per day)
    days: float    # stream duration
    rate0: float   # base report intensity (reports/day at t=0)


def _sample_params(rng: random.Random) -> StreamParams:
    d_final = int(round(math.exp(rng.uniform(math.log(30), math.log(6000)))))
    i_final = int(round(d_final * rng.uniform(3.0, 12.0)))
    m0 = int(round(d_final * rng.uniform(0.5, 5.0)))
    return StreamParams(
        d_final=d_final, i_final=i_final, m0=m0,
        beta=rng.uniform(0.15, 0.6),   # dead half-life ~1.2-4.6 days
        gamma=rng.uniform(0.10, 0.40),
        lam=rng.uniform(0.08, 0.25),
        days=rng.uniform(14.0, 45.0),
        rate0=rng.uniform(*REGIME["rate0"]),
    )


def _true_state(p: StreamParams, t_days: float) -> Dict[str, float]:
    """Ground-truth continuous state at t (days). Linear ODE closed forms (design sec 3)."""
    return {
        "dead": p.d_final * (1.0 - math.exp(-p.beta * t_days)),
        "injured": p.i_final * (1.0 - math.exp(-p.beta * t_days)),
        "missing": p.m0 * math.exp(-p.gamma * t_days),
        "salience": math.exp(-p.lam * t_days),
    }


def _report_times(p: StreamParams, rng: random.Random) -> List[float]:
    """Inhomogeneous Poisson report times (days): intensity rate0*salience(t) via thinning."""
    times: List[float] = []
    t = 0.0
    while True:
        t += rng.expovariate(p.rate0)  # candidate at base rate
        if t >= p.days:
            break
        if rng.random() < math.exp(-p.lam * t):  # accept with prob = salience(t)
            times.append(t)
    return times


def _bucket(v: float) -> Tuple[str, int]:
    for lo, hi, name in ((10, 100, "dozens"), (100, 1000, "hundreds"), (1000, 10**12, "thousands")):
        if lo <= v < hi:
            mid = int(round(math.sqrt(lo * hi))) if hi < 10**12 else int(round(v / 1000.0) * 1000)
            return name, mid
    return "few", max(1, int(round(v)))


def _observe(role: str, true_val: float, rng: random.Random) -> Dict:
    """One hedged, source-attributed observation of a role's true value."""
    src = rng.choices(SOURCES, weights=REGIME["src_w"])[0]
    sigma, bias = SOURCE_NOISE[src]
    if role == "missing":
        q = rng.choices(["feared", "about", "interval", "at_least"], REGIME["qual_miss"])[0]
    else:
        q = rng.choices(["point", "at_least", "about", "interval"], REGIME["qual_rise"])[0]
    base = max(0.0, true_val * bias)
    bucket: Optional[str] = None
    if q == "point":
        val = base * (1.0 + rng.gauss(0, sigma))
    elif q == "at_least":
        val = base * rng.uniform(0.5, 0.9)             # a genuine lower bound (censored)
    elif q == "about":
        val = base * (1.0 + rng.gauss(0, sigma * 1.5))
    elif q == "feared":
        val = base * (1.0 + abs(rng.gauss(0, sigma * 2.0)))  # often an over-estimate
    else:  # interval
        bucket, val = _bucket(base)
    return {"role": role, "qualifier": q, "value": max(0, int(round(val))),
            "bucket": bucket, "source": src}


_ROLE_WORD = {"dead": "dead", "injured": "injured", "missing": "missing"}


def _templated_text(obs: Dict) -> str:
    """Placeholder realizer (free). --text sonnet5 (LLM) is the higher-quality upgrade."""
    role, q, v = _ROLE_WORD[obs["role"]], obs["qualifier"], obs["value"]
    src = {"official": "Authorities", "major_outlet": "Reports", "preliminary": "Early reports"}[obs["source"]]
    if q == "point":
        return f"{src} say the confirmed {role} toll has risen to {v}."
    if q == "at_least":
        return f"{src} say at least {v} people are {role}."
    if q == "about":
        return f"{src} put the number {role} at about {v}."
    if q == "feared":
        return f"{src} say some {v} people are feared {role}."
    return f"{src} say {obs['bucket']} of people are {role}."


def _sample_stream(stream_id: str, rng: random.Random, with_text: bool) -> Tuple[List[Dict], List[Dict]]:
    p = _sample_params(rng)
    # ground-truth trajectory on a fixed grid
    trajectory: List[Dict] = []
    n_grid = int(p.days * 24 / GRID_HOURS) + 1
    for g in range(n_grid):
        t_days = g * GRID_HOURS / 24.0
        st = _true_state(p, t_days)
        trajectory.append({"stream_id": stream_id, "t_hours": round(t_days * 24, 1),
                           **{k: round(v, 3) for k, v in st.items()}})
    # observations
    observations: List[Dict] = []
    for t in sorted(_report_times(p, rng)):
        st = _true_state(p, t)
        roles = [r for r in ROLES if st[r] >= 1.0]
        if not roles:
            continue
        k = rng.choices([1, 2, 3], [0.6, 0.3, 0.1])[0]
        for role in rng.sample(roles, min(k, len(roles))):
            obs = _observe(role, st[role], rng)
            obs = {"stream_id": stream_id, "t_hours": round(t * 24, 1), **obs}
            if with_text:
                obs["text"] = _templated_text(obs)
            observations.append(obs)
    return observations, trajectory


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic disaster event streams.")
    ap.add_argument("--out", default="datasets/disaster_streams")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regime", choices=list(REGIMES), default="normal",
                    help="'hard' = sparse reporting + unreliable sources + heavy censoring")
    ap.add_argument("--text", choices=["none", "template"], default="none",
                    help="attach a 'text' field per observation (template = free placeholder; "
                         "the sonnet5 LLM realizer is a separate step)")
    args = ap.parse_args(argv)

    global REGIME
    REGIME = REGIMES[args.regime]
    out = Path(args.out)
    with_text = args.text == "template"
    counts = {}
    for split, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        if n <= 0:
            continue
        obs_all: List[Dict] = []
        traj_all: List[Dict] = []
        for i in range(n):
            sid = f"{split}-{i:05d}"
            rng = random.Random(f"{args.seed}-{split}-{i}")  # per-stream reproducible
            obs, traj = _sample_stream(sid, rng, with_text)
            obs_all.extend(obs)
            traj_all.extend(traj)
        _write_jsonl(out / split / "observations.jsonl", obs_all)
        _write_jsonl(out / split / "trajectory.jsonl", traj_all)
        counts[split] = {"streams": n, "observations": len(obs_all), "trajectory_points": len(traj_all)}
        print(f"[{split}] {n} streams, {len(obs_all)} observations, {len(traj_all)} traj points")

    config = {"seed": args.seed, "regime": args.regime, "text": args.text,
              "grid_hours": GRID_HOURS, "roles": list(ROLES), "sources": list(SOURCES),
              "regime_params": REGIME, "counts": counts}
    (out).mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[config] wrote {out / 'config.json'}")


if __name__ == "__main__":
    main()
