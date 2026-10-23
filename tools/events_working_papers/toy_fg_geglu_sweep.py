"""fg-GEGLU parameter sweep on the toy (width 72, divisible by 4, 6, 8, 12): (chunks, gates, skipped) grid, both
inference rules (plain GELU; mean of 8 gate patterns), two budgets. Baselines GELU and dropout 0.1. MSE.
Output: one line per configuration per budget, plus a ranking by blind MSE under the MC rule."""
import sys
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/private/tmp/claude-501/-Users-williamroe-development-git-GLiNER2/53137fe2-d0b1-48f7-953e-b33bc19fd0a1/scratchpad")
import toy_fg_geglu as T  # noqa: E402
from gliner2.models.boundary.flipped_gate import FlippedGateGELU  # noqa: E402

H = 72
GRID = [(4, 2, 1), (4, 2, 2), (6, 2, 2), (6, 3, 2), (6, 2, 3), (6, 2, 1), (8, 2, 2), (8, 2, 4), (8, 4, 2), (8, 3, 3), (8, 4, 4),
        (12, 2, 4), (12, 4, 4), (12, 4, 2), (12, 3, 3), (12, 6, 4), (12, 2, 2)]


def build(cfg):
    if cfg == "gelu":
        mk = lambda: nn.GELU()
    elif cfg == "drop":
        mk = lambda: nn.Sequential(nn.GELU(), nn.Dropout(0.1))
    else:
        c, g, s = cfg
        mk = lambda: FlippedGateGELU(c, g, s, eval_mode="mc", mc_samples=8)
    return nn.Sequential(nn.Linear(T.D, H), mk(), nn.Linear(H, H), mk(), nn.Linear(H, 1))


def data(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    xtr = torch.randn(T.N_TRAIN, T.D, generator=g); ytr = T.target(xtr) + T.NOISE * torch.randn(T.N_TRAIN, generator=g)
    xdev = torch.randn(T.N_DEV, T.D, generator=g); xte = torch.randn(T.N_TEST, T.D, generator=g)
    return g, xtr, ytr, xdev, T.target(xdev), xte, T.target(xte)


def set_eval_mode(m, mode):
    for mod in m.modules():
        if isinstance(mod, FlippedGateGELU):
            mod.eval_mode = mode


def measure(m, cfg, xdev, ydev, xte, yte):
    """{rule: (dev, blind)}; fg gets both rules, baselines one."""
    out = {}
    with torch.no_grad():
        m.eval()
        rules = ("gelu", "mc") if isinstance(cfg, tuple) else ("plain",)
        for rule in rules:
            if isinstance(cfg, tuple):
                set_eval_mode(m, rule)
            torch.manual_seed(12345)
            out[rule] = (F.mse_loss(m(xdev).squeeze(-1), ydev).item(), F.mse_loss(m(xte).squeeze(-1), yte).item())
    return out


def run(cfg, seeds, steps, minibatch):
    rows = []
    for seed in seeds:
        g, xtr, ytr, xdev, ydev, xte, yte = data(seed)
        torch.manual_seed(seed); m = build(cfg); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if minibatch else None
        for _ in range(steps):
            idx = torch.randint(0, T.N_TRAIN, (64,), generator=g) if minibatch else slice(None)
            m.train(); opt.zero_grad(); F.mse_loss(m(xtr[idx]).squeeze(-1), ytr[idx]).backward(); opt.step()
            if sched: sched.step()
        rows.append(measure(m, cfg, xdev, ydev, xte, yte))
    return rows


def summarize(title, seeds, steps, minibatch):
    print(title, flush=True)
    print(f"{'config':10s} {'rule':6s} {'dev':>8s} {'blind':>8s} {'|dev-blind|':>12s}  per-seed blind", flush=True)
    ranking = []
    for cfg in ["gelu", "drop"] + GRID:
        rows = run(cfg, seeds, steps, minibatch)
        name = cfg if isinstance(cfg, str) else f"fg{cfg[0]}-{cfg[1]}-{cfg[2]}"
        for rule in rows[0]:
            dev = sum(r[rule][0] for r in rows) / len(rows); te = sum(r[rule][1] for r in rows) / len(rows)
            gap = sum(abs(r[rule][0] - r[rule][1]) for r in rows) / len(rows)
            print(f"{name:10s} {rule:6s} {dev:8.4f} {te:8.4f} {gap:12.4f}  {[round(r[rule][1], 3) for r in rows]}", flush=True)
            if rule in ("mc", "plain"):
                ranking.append((te, name, rule, gap))
    print("ranking by blind MSE (fg under the MC rule):", flush=True)
    for te, name, rule, gap in sorted(ranking)[:8]:
        print(f"  {name:10s} {rule:6s} blind {te:.4f} gap {gap:.4f}", flush=True)


if __name__ == "__main__":
    summarize(f"width {H}: 1,500 full-batch steps, 6 seeds", range(6), 1500, False)
    summarize(f"width {H}: 10,000 mini-batch steps (batch 64), cosine LR, 4 seeds", range(4), 10000, True)
