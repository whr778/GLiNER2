"""Toy dev-vs-blind check of fg-GEGLU against plain GELU (and dropout 0.1 as the familiar yardstick).

A 2-hidden-layer MLP (width 64) fits a noisy nonlinear target from 20 inputs on 400 training points; dev (1,000) and a
blind test (5,000) are drawn from the same distribution without the label noise. Over 8 seeds: train MSE at the end,
dev MSE, blind MSE, and the dev-blind gap. The claim under test is "fg-GEGLU trains like GELU and generalizes better".
"""
import sys
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2")
from gliner2.models.boundary.flipped_gate import FlippedGateGELU  # noqa: E402

D, H, N_TRAIN, N_DEV, N_TEST, STEPS, NOISE = 20, 64, 400, 1000, 5000, 1500, 0.3


def target(x):
    return torch.sin(x[:, :5].sum(1)) + 0.5 * (x[:, 5:10] ** 2).sum(1) - torch.tanh(x[:, 10:15]).prod(1) + 0.3 * x[:, 15:].sum(1)


def make(seed, act):
    torch.manual_seed(seed)
    a1 = {"gelu": nn.GELU(), "fg": FlippedGateGELU(4, 2, 1), "fg3": FlippedGateGELU(4, 3, 1), "fg_s0": FlippedGateGELU(4, 2, 0), "drop": nn.Sequential(nn.GELU(), nn.Dropout(0.1))}[act]
    a2 = {"gelu": nn.GELU(), "fg": FlippedGateGELU(4, 2, 1), "fg3": FlippedGateGELU(4, 3, 1), "fg_s0": FlippedGateGELU(4, 2, 0), "drop": nn.Sequential(nn.GELU(), nn.Dropout(0.1))}[act]
    return nn.Sequential(nn.Linear(D, H), a1, nn.Linear(H, H), a2, nn.Linear(H, 1))


def run(seed, act):
    g = torch.Generator().manual_seed(1000 + seed)
    xtr = torch.randn(N_TRAIN, D, generator=g); ytr = target(xtr) + NOISE * torch.randn(N_TRAIN, generator=g)
    xdev = torch.randn(N_DEV, D, generator=g); ydev = target(xdev)
    xte = torch.randn(N_TEST, D, generator=g); yte = target(xte)
    m = make(seed, act); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    for _ in range(STEPS):
        m.train(); opt.zero_grad(); F.mse_loss(m(xtr).squeeze(-1), ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return (F.mse_loss(m(xtr).squeeze(-1), ytr).item(), F.mse_loss(m(xdev).squeeze(-1), ydev).item(), F.mse_loss(m(xte).squeeze(-1), yte).item())


if __name__ == "__main__":
    print(f"MLP {D}->{H}->{H}->1, {N_TRAIN} noisy train points (noise sd {NOISE}), {STEPS} full-batch AdamW steps, 8 seeds; MSE")
    print(f"{'act':5s} {'train':>8s} {'dev':>8s} {'blind':>8s} {'dev-blind gap':>14s}   per-seed blind")
    for act in ("gelu", "drop", "fg", "fg3", "fg_s0"):
        rows = [run(s, act) for s in range(8)]
        tr = sum(r[0] for r in rows) / 8; dv = sum(r[1] for r in rows) / 8; te = sum(r[2] for r in rows) / 8
        gap = sum(abs(r[1] - r[2]) for r in rows) / 8
        print(f"{act:5s} {tr:8.4f} {dv:8.4f} {te:8.4f} {gap:14.4f}   {[round(r[2], 3) for r in rows]}")
