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
GAMMA = 0.25  # missing-decay rate prior (per day)

# MoE gate (Reading B): weight on the tracked state = 1 - trust(last report)*recency.
SRC_TRUST = {"official": 1.0, "major_outlet": 0.7, "preliminary": 0.4}
QUAL_TRUST = {"point": 1.0, "about": 0.8, "at_least": 0.5, "interval": 0.4, "feared": 0.3}
GATE_TAU = 12.0  # hours; recency half-scale


def _R_at(o: Dict, ref: float) -> float:
    """Measurement-noise variance scaled by a reference LEVEL (the current estimate),
    not the observed value. Multiplicative noise -> linearize R around the state; scaling
    by the raw report over-trusts a low reading and drags a rising estimate downward
    (the failure the harder-regime ablation exposed under unreliable sources)."""
    sig = SRC_REL_SIGMA[o["source"]] * QUAL_FACTOR[o["qualifier"]]
    return (sig * max(ref, 1.0)) ** 2


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
                R = _R_at(o, mu); K = P / (P + R); mu = mu + K * (z - mu); P = (1 - K) * P
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
            R = _R_at(o, mu); S = P + R; K = P / S
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


def est_moe(obs: List[Dict], grid: List[float], role: str) -> List[float]:
    """Gate the local read (last_value) and the tracked state (EKF): alpha =
    weight on the tracked state = 1 - trust(last report) * recency. Fresh+reliable
    -> local read; stale/censored/unreliable -> tracked state."""
    ekf = est_ekf(obs, grid, role)
    lv = est_last_value(obs, grid)
    obs = sorted(obs, key=lambda o: o["t_hours"])
    out, j, last = [], 0, None
    for i, t in enumerate(grid):
        while j < len(obs) and obs[j]["t_hours"] <= t:
            last = obs[j]; j += 1
        if last is None:
            out.append(ekf[i]); continue
        tau = SRC_TRUST[last["source"]] * QUAL_TRUST[last["qualifier"]]
        alpha = 1.0 - tau * math.exp(-(t - last["t_hours"]) / GATE_TAU)
        out.append(alpha * ekf[i] + (1.0 - alpha) * lv[i])
    return out


# --------------------------------------------------------------------------- #
# Learned gate: a logistic router alpha = sigmoid(w.x) that replaces the hand-set
# SRC_TRUST/QUAL_TRUST/GATE_TAU tables. Trained on the train split by gradient
# descent to minimize the blend's normalized MSE against ground truth.
# --------------------------------------------------------------------------- #
GATE = None  # fitted {"w","mu","sd"}, set by --learn-gate


def _gate_feats(obs: List[Dict], grid: List[float], role: str,
                ekf: List[float], lv: List[float]):
    """One feature row per grid point (None before the first report). Same builder
    used at fit and apply time -> single source of truth."""
    obs = sorted(obs, key=lambda o: o["t_hours"])
    rows, j, last, n = [], 0, None, 0
    for i, t in enumerate(grid):
        while j < len(obs) and obs[j]["t_hours"] <= t:
            last = obs[j]; j += 1; n += 1
        if last is None:
            rows.append(None); continue
        gap = (ekf[i] - lv[i]) / max(abs(lv[i]), 1.0)
        rows.append(np.array([
            1.0,                                           # bias
            math.log1p(max(t - last["t_hours"], 0.0)),     # staleness (hours)
            SRC_REL_SIGMA[last["source"]],                 # source unreliability
            QUAL_FACTOR[last["qualifier"]],                # qualifier coarseness
            1.0 if last["qualifier"] == "at_least" else 0.0,  # censored lower bound
            1.0 if role in DECAY_ROLES else 0.0,           # decay vs rise role
            max(-3.0, min(3.0, gap)),                      # ekf-vs-lastvalue disagreement
            math.log1p(n),                                 # reports seen so far
        ]))
    return rows


def fit_gate(train_dir: Path, steps: int = 2000, lr: float = 0.5):
    """Fit the router on peak-normalized blend MSE (matches the eval metric)."""
    obs, traj = _load(train_dir)
    X, E, L, Y = [], [], [], []
    for s, tps in traj.items():
        grid = [tp["t_hours"] for tp in tps]
        for role in ROLES:
            true = np.array([tp[role] for tp in tps]); peak = max(true.max(), 1.0)
            obs_r = sorted(obs[s].get(role, []), key=lambda o: o["t_hours"])
            ekf = est_ekf(obs_r, grid, role); lv = est_last_value(obs_r, grid)
            for i, r in enumerate(_gate_feats(obs_r, grid, role, ekf, lv)):
                if r is None:
                    continue
                X.append(r); E.append(ekf[i] / peak); L.append(lv[i] / peak); Y.append(true[i] / peak)
    X = np.array(X); E = np.array(E); L = np.array(L); Y = np.array(Y)
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-6] = 1.0; mu[0] = 0.0; sd[0] = 1.0  # keep bias
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1])
    for _ in range(steps):
        a = 1.0 / (1.0 + np.exp(-(Xs @ w)))
        p = a * E + (1 - a) * L
        w -= lr * (Xs.T @ (2.0 * (p - Y) * (E - L) * a * (1 - a))) / len(Y)
    return {"w": w, "mu": mu, "sd": sd, "n": len(Y)}


def est_moe_learned(obs: List[Dict], grid: List[float], role: str) -> List[float]:
    ekf = est_ekf(obs, grid, role); lv = est_last_value(obs, grid)
    w, mu, sd = GATE["w"], GATE["mu"], GATE["sd"]
    out = []
    for i, r in enumerate(_gate_feats(obs, grid, role, ekf, lv)):
        if r is None:
            out.append(ekf[i]); continue
        a = 1.0 / (1.0 + math.exp(-float(((r - mu) / sd) @ w)))
        out.append(a * ekf[i] + (1.0 - a) * lv[i])
    return out


METHODS = {"last_value": None, "weighted_avg": None, "running_max": None,
           "EKF": None, "MoE_gate": None}


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
    if method == "MoE_gate": return est_moe(obs, grid, role)
    if method == "MoE_learned": return est_moe_learned(obs, grid, role)
    return est_ekf(obs, grid, role)


def main(argv=None) -> None:
    global GATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/disaster_streams")
    ap.add_argument("--split", default="val")
    ap.add_argument("--learn-gate", action="store_true",
                    help="fit the logistic router on <data>/train and add MoE_learned")
    args = ap.parse_args(argv)

    methods = list(METHODS)
    if args.learn_gate:
        GATE = fit_gate(Path(args.data) / "train")
        methods.append("MoE_learned")
        print(f"[gate] fit on {GATE['n']} points; w={np.round(GATE['w'], 3).tolist()}")

    obs, traj = _load(Path(args.data) / args.split)
    # per-(method, role) list of per-stream normalized RMSE (by peak true value)
    nrmse = {m: {r: [] for r in ROLES} for m in methods}
    final_err = {m: {r: [] for r in ROLES} for m in methods}

    for s, tps in traj.items():
        grid = [tp["t_hours"] for tp in tps]
        for role in ROLES:
            true = np.array([tp[role] for tp in tps])
            peak = max(true.max(), 1.0)
            obs_r = sorted(obs[s].get(role, []), key=lambda o: o["t_hours"])
            for m in methods:
                est = np.array(_estimate(m, obs_r, grid, role))
                nrmse[m][role].append(float(np.sqrt(np.mean((est - true) ** 2)) / peak))
                final_err[m][role].append(abs(est[-1] - true[-1]) / peak)

    print(f"\n== {args.split}: {len(traj)} streams == (normalized RMSE, lower is better)\n")
    hdr = f"{'method':14s}" + "".join(f"{r:>11s}" for r in ROLES) + f"{'overall':>11s}"
    print(hdr); print("-" * len(hdr))
    for m in methods:
        per_role = {r: float(np.mean(nrmse[m][r])) for r in ROLES}
        overall = float(np.mean([per_role[r] for r in ROLES]))
        print(f"{m:14s}" + "".join(f"{per_role[r]:11.4f}" for r in ROLES) + f"{overall:11.4f}")
    print("\nfinal-value normalized error:")
    for m in methods:
        overall = float(np.mean([np.mean(final_err[m][r]) for r in ROLES]))
        print(f"  {m:14s} {overall:.4f}")


if __name__ == "__main__":
    main()
