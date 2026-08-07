"""EKF tracker (v1, per-role) + baseline harness on synthetic disaster streams.

Reads <split>/observations.jsonl (hedged, censored, source-attributed reports),
runs each estimator causally (obs with t <= grid_t only), and scores its state
trajectory against trajectory.jsonl (ground truth). Inference-only, hand-set
dynamics (EKF_MHT_DESIGN.md sec 3/6); no GPU/API.

  uv run python datasets/disaster_streams/evaluate.py --split val

The point: does the EKF beat recursive fusion (weighted-average) and the naive
last-value / running-max heuristics on trajectory RMSE?
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

RISE_ROLES = ("dead", "injured")
DECAY_ROLES = ("missing",)
ROLES = RISE_ROLES + DECAY_ROLES

SRC_REL_SIGMA = {"official": 0.06, "major_outlet": 0.12, "preliminary": 0.25}
QUAL_FACTOR = {"point": 1.0, "about": 1.6, "interval": 2.0, "feared": 2.5, "at_least": 1.2}
BETA = 0.15   # rise-rate prior (per day) -- gentle; the asymptote x* is estimated
GAMMA = 0.25  # decay-rate prior (per day)


def _R(o: Dict) -> float:
    sig = SRC_REL_SIGMA[o["source"]] * QUAL_FACTOR[o["qualifier"]]
    return (sig * max(o["value"], 1.0)) ** 2


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def est_last_value(obs: List[Dict], grid: List[float]) -> List[float]:
    out, j, cur = [], 0, 0.0
    for t in grid:
        while j < len(obs) and obs[j]["t_hours"] <= t:
            cur = obs[j]["value"]; j += 1
        out.append(cur)
    return out


def est_weighted_avg(obs: List[Dict], grid: List[float]) -> List[float]:
    out, j, sw, swv = [], 0, 0.0, 0.0
    for t in grid:
        while j < len(obs) and obs[j]["t_hours"] <= t:
            w = 1.0 / SRC_REL_SIGMA[obs[j]["source"]]
            sw += w; swv += w * obs[j]["value"]; j += 1
        out.append(swv / sw if sw > 0 else 0.0)
    return out


def est_running_max(obs: List[Dict], grid: List[float]) -> List[float]:
    out, j, mx = [], 0, 0.0
    for t in grid:
        while j < len(obs) and obs[j]["t_hours"] <= t:
            mx = max(mx, obs[j]["value"]); j += 1
        out.append(mx)
    return out


# --------------------------------------------------------------------------- #
# EKF
# --------------------------------------------------------------------------- #
def _predict_rise(mu, P, t_cur, t_new):
    dt = max(0.0, (t_new - t_cur) / 24.0)
    F = np.array([[1 - BETA * dt, BETA * dt], [0.0, 1.0]])
    q = max(dt, 1e-3)
    Q = np.diag([(0.10 * max(mu[0], 1.0) * q) ** 2, (0.04 * max(mu[1], 1.0) * q) ** 2])
    return F @ mu, F @ P @ F.T + Q, t_new


def _update_rise(mu, P, o):
    z, R = o["value"], _R(o)
    H = np.array([[1.0, 0.0]])
    pred = float(mu[0])                    # H = [1,0]
    if o["qualifier"] == "at_least" and z <= pred:
        return mu, P                      # a lower bound below our estimate is uninformative
    S = float(P[0, 0]) + R
    K = (P @ H.T) / S
    mu = mu + K.flatten() * (z - pred)
    P = (np.eye(2) - K @ H) @ P
    mu[1] = max(mu[1], mu[0])             # asymptote >= current (rise, not a hard monotone clamp on x)
    return mu, P


def est_ekf_rise(obs: List[Dict], grid: List[float]) -> List[float]:
    """1D random-walk smoother, censoring-aware. Holds between reports (no rise
    over-shoot); fuses reports weighted by source/qualifier; ignores 'at least'
    bounds below the estimate (one-sided). q_rel sets smoothing strength."""
    obs = sorted(obs, key=lambda o: o["t_hours"])
    q_rel = 0.20
    mu = P = None; t_cur = 0.0; j = 0; out = []

    def grow(mu, P, t0, t1):
        dt = max(0.0, (t1 - t0) / 24.0)
        return P + (q_rel * max(mu, 1.0) * max(dt, 1e-3)) ** 2

    for gt in grid:
        while j < len(obs) and obs[j]["t_hours"] <= gt:
            o = obs[j]; z = o["value"]
            if mu is None:
                mu = float(z); P = (0.4 * max(z, 1.0)) ** 2; t_cur = o["t_hours"]; j += 1
                continue
            P = grow(mu, P, t_cur, o["t_hours"]); t_cur = o["t_hours"]
            if not (o["qualifier"] == "at_least" and z <= mu):   # else: uninformative lower bound
                R = _R(o); K = P / (P + R); mu = mu + K * (z - mu); P = (1 - K) * P
            j += 1
        if mu is None:
            out.append(0.0)
        else:
            P = grow(mu, P, t_cur, gt); t_cur = gt
            out.append(float(mu))
    return out


def est_ekf_decay(obs: List[Dict], grid: List[float]) -> List[float]:
    obs = sorted(obs, key=lambda o: o["t_hours"])
    mu = P = None; t_cur = 0.0; j = 0; out = []
    for gt in grid:
        while j < len(obs) and obs[j]["t_hours"] <= gt:
            o = obs[j]
            if mu is None:
                mu = float(o["value"]); P = (0.5 * max(o["value"], 1.0)) ** 2; t_cur = o["t_hours"]
            else:
                dt = max(0.0, (o["t_hours"] - t_cur) / 24.0); f = math.exp(-GAMMA * dt)
                mu = f * mu; P = f * f * P + (0.10 * max(mu, 1.0) * max(dt, 1e-3)) ** 2; t_cur = o["t_hours"]
            R = _R(o); S = P + R; K = P / S
            mu = mu + K * (o["value"] - mu); P = (1 - K) * P; j += 1
        if mu is None:
            out.append(0.0)
        else:
            dt = max(0.0, (gt - t_cur) / 24.0); f = math.exp(-GAMMA * dt)
            mu = f * mu; P = f * f * P; t_cur = gt
            out.append(float(mu))
    return out


def est_ekf(obs, grid, role):
    return est_ekf_decay(obs, grid) if role in DECAY_ROLES else est_ekf_rise(obs, grid)


METHODS = {"last_value": None, "weighted_avg": None, "running_max": None, "EKF": None}


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _load(split_dir: Path):
    obs = defaultdict(lambda: defaultdict(list))
    for line in (split_dir / "observations.jsonl").open(encoding="utf-8"):
        o = json.loads(line); obs[o["stream_id"]][o["role"]].append(o)
    traj = defaultdict(list)
    for line in (split_dir / "trajectory.jsonl").open(encoding="utf-8"):
        t = json.loads(line); traj[t["stream_id"]].append(t)
    for s in traj:
        traj[s].sort(key=lambda x: x["t_hours"])
    return obs, traj


def _estimate(method: str, obs, grid, role):
    if method == "last_value": return est_last_value(obs, grid)
    if method == "weighted_avg": return est_weighted_avg(obs, grid)
    if method == "running_max": return est_running_max(obs, grid)
    return est_ekf(obs, grid, role)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/disaster_streams")
    ap.add_argument("--split", default="val")
    args = ap.parse_args(argv)

    obs, traj = _load(Path(args.data) / args.split)
    # per-(method, role) list of per-stream normalized RMSE (by peak true value)
    nrmse = {m: {r: [] for r in ROLES} for m in METHODS}
    final_err = {m: {r: [] for r in ROLES} for m in METHODS}

    for s, tps in traj.items():
        grid = [tp["t_hours"] for tp in tps]
        for role in ROLES:
            true = np.array([tp[role] for tp in tps])
            peak = max(true.max(), 1.0)
            obs_r = sorted(obs[s].get(role, []), key=lambda o: o["t_hours"])
            for m in METHODS:
                est = np.array(_estimate(m, obs_r, grid, role))
                nrmse[m][role].append(float(np.sqrt(np.mean((est - true) ** 2)) / peak))
                final_err[m][role].append(abs(est[-1] - true[-1]) / peak)

    print(f"\n== {args.split}: {len(traj)} streams == (normalized RMSE, lower is better)\n")
    hdr = f"{'method':14s}" + "".join(f"{r:>11s}" for r in ROLES) + f"{'overall':>11s}"
    print(hdr); print("-" * len(hdr))
    for m in METHODS:
        per_role = {r: float(np.mean(nrmse[m][r])) for r in ROLES}
        overall = float(np.mean([per_role[r] for r in ROLES]))
        print(f"{m:14s}" + "".join(f"{per_role[r]:11.4f}" for r in ROLES) + f"{overall:11.4f}")
    print("\nfinal-value normalized error:")
    for m in METHODS:
        overall = float(np.mean([np.mean(final_err[m][r]) for r in ROLES]))
        print(f"  {m:14s} {overall:.4f}")


if __name__ == "__main__":
    main()
