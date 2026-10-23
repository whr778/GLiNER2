"""fg-GEGLU sweep to 24 and 32 chunks at width 96 (divisible by 4, 6, 8, 12, 16, 24, 32), MC inference (K=8), plus a
zero-skip variant (skipped chunks output 0 instead of passing through: the 'stop dendrite' reading). Both budgets. MSE."""
import sys
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/private/tmp/claude-501/-Users-williamroe-development-git-GLiNER2/53137fe2-d0b1-48f7-953e-b33bc19fd0a1/scratchpad")
import toy_fg_geglu as T  # noqa: E402
from gliner2.models.boundary.flipped_gate import FlippedGateGELU  # noqa: E402

H = 96


class ZeroSkip(FlippedGateGELU):
    """Experiment-only variant: skipped chunks output zero (propagation stopped) instead of passing through."""
    def gated(self, x):
        pieces = list(x.chunk(self.chunks, dim=-1)); activated, skipped = self.draw(x.device)
        product = F.gelu(pieces[activated[0]])
        for i in activated[1:]:
            product = product * F.gelu(pieces[i])
        return torch.cat([product if i in activated else torch.zeros_like(p) if i in skipped else F.gelu(p) for i, p in enumerate(pieces)], dim=-1)


GRID = [("gelu",), ("drop",), (12, 3, 3), (12, 2, 4), (16, 4, 4), (16, 2, 2), (24, 6, 6), (24, 4, 4), (24, 3, 3), (24, 2, 2), (24, 8, 8),
        (24, 2, 4), (32, 8, 8), (32, 4, 4), (12, 3, 3, "zero"), (24, 6, 6, "zero")]


def name(cfg):
    if cfg[0] in ("gelu", "drop"): return cfg[0]
    return f"fg{cfg[0]}-{cfg[1]}-{cfg[2]}" + ("z" if len(cfg) > 3 else "")


def build(cfg):
    if cfg[0] == "gelu": mk = lambda: nn.GELU()
    elif cfg[0] == "drop": mk = lambda: nn.Sequential(nn.GELU(), nn.Dropout(0.1))
    elif len(cfg) > 3: mk = lambda: ZeroSkip(cfg[0], cfg[1], cfg[2], eval_mode="mc", mc_samples=8)
    else: mk = lambda: FlippedGateGELU(cfg[0], cfg[1], cfg[2], eval_mode="mc", mc_samples=8)
    return nn.Sequential(nn.Linear(T.D, H), mk(), nn.Linear(H, H), mk(), nn.Linear(H, 1))


def data(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    xtr = torch.randn(T.N_TRAIN, T.D, generator=g); ytr = T.target(xtr) + T.NOISE * torch.randn(T.N_TRAIN, generator=g)
    xdev = torch.randn(T.N_DEV, T.D, generator=g); xte = torch.randn(T.N_TEST, T.D, generator=g)
    return g, xtr, ytr, xdev, T.target(xdev), xte, T.target(xte)


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
        with torch.no_grad():
            m.eval(); torch.manual_seed(12345)
            rows.append((F.mse_loss(m(xdev).squeeze(-1), ydev).item(), F.mse_loss(m(xte).squeeze(-1), yte).item()))
    return rows


def summarize(title, seeds, steps, minibatch):
    print(title, flush=True); print(f"{'config':11s} {'dev':>8s} {'blind':>8s} {'|dev-blind|':>12s}  per-seed blind", flush=True)
    rank = []
    for cfg in GRID:
        rows = run(cfg, seeds, steps, minibatch)
        dev = sum(r[0] for r in rows) / len(rows); te = sum(r[1] for r in rows) / len(rows); gap = sum(abs(r[0] - r[1]) for r in rows) / len(rows)
        print(f"{name(cfg):11s} {dev:8.4f} {te:8.4f} {gap:12.4f}  {[round(r[1], 3) for r in rows]}", flush=True); rank.append((te, name(cfg), gap))
    print("ranking by blind MSE:", ", ".join(f"{n} {te:.3f} (gap {g:.3f})" for te, n, g in sorted(rank)[:8]), flush=True)


if __name__ == "__main__":
    summarize(f"width {H}: 1,500 full-batch steps, 6 seeds (MC K=8 for fg)", range(6), 1500, False)
    summarize(f"width {H}: 10,000 mini-batch steps (batch 64), cosine LR, 4 seeds (MC K=8 for fg)", range(4), 10000, True)
