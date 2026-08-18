"""Gated and stochastic FFN activation variants, on a parameter-matched toy harness.

Standalone: imports only torch, touches no checkpoint, config or corpus. It lives in
``prototypes/`` rather than ``tools/train/`` because it is architecture research, not
part of any pipeline here. Findings are written up in
``tools/events_working_papers/TODO.md`` item 2d.

**What it answers.** GeGLU's ``v * gelu(g)`` reintroduces an unbounded gradient path
that a pointwise activation cannot have: ``dy/dv = gelu(g)`` and
``dy/dg = v * gelu'(g)`` each scale with the OTHER branch. The variants here try to
keep the multiplicative expressiveness while bounding that path.

**Headline results** (``--grad-scan`` and the default sweep reproduce both):

* Max |grad| through the activation, by input scale -- plain GELU is pinned at 1.13 at
  every scale; GeGLU reaches 37.8 at 8x; a hybrid gating ONE chunk in six still reaches
  32.7. Partial gating does not partially protect: only 2.6% of channels exceed the
  bound, but the max is what explodes.
* Randomising WHICH slots get the nonlinearity costs ~10x (0.43-0.80 MSE vs 0.049-0.070
  for every deterministic variant). Randomising only which chunk PARTNERS a gated slot
  is free (hybrid5, 0.0623). The rule is that a draw changing a slot's function CLASS is
  fatal, because ``fc2`` reads a fixed slot and one weight cannot serve both GELU output
  and identity output. Dropout escapes this only by being linear in the mask.
* ``hybrid4`` with a fixed assignment is the keeper: width preserved, parameter-identical
  to plain GELU, MSE tied with it, and the lowest gradient max in the study.

**Prior art.** Noam Shazeer, *GLU Variants Improve Transformer*, arXiv:2002.05202 (2020)
defines this family: GLU (sigmoid gate), GEGLU (GELU gate -- what mmBERT/ModernBERT use),
SwiGLU (Swish gate), and **Bilinear**, which drops the nonlinearity and is just the
component-wise product of two projections. ``hybrid4``'s product chunk IS Bilinear,
applied to 1/8 of the channels rather than all of them; every variant in that paper gates
all hidden units. The two-thirds ``d_ff`` rule used here for parameter matching is also
from that paper. No published study of gating only a FRACTION of channels turned up in a
search, but that is weak evidence, not a novelty claim.

**Harness limits, which bound every claim.** 4-block residual MLP, D=64, H=192, AdamW
3e-3, 3000 steps, synthetic regression with multiplicative interactions. Plain GELU alone
ranges 0.043-0.074 across seeds. This separates 0.06 from 0.65 reliably and cannot
separate 0.048 from 0.066 at all.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 64
H = 192


def geglu(h):
    """Standard GeGLU: halve the width, gating one half by the GELU of the other."""
    value, gate = h.chunk(2, dim=-1)
    return value * F.gelu(gate)


def stoch(h, n, k, randomize, training):
    """GELU applied to k of n chunks; width preserved.

    Randomised, a slot is GELU on some steps and identity on others. The eval rule is
    the expectation ``p*gelu(x) + (1-p)*x``, which is an APPROXIMATION -- GELU does not
    commute with the expectation, so no scalar can correct it the way dropout's can.
    A fixed mask is the same function at train and eval, so it must not be blended.
    """
    chunks = list(h.chunk(n, dim=-1))
    if randomize and not training:
        p = k / n
        return torch.cat([p * F.gelu(c) + (1 - p) * c for c in chunks], dim=-1)
    slots = torch.randperm(n)[:k].tolist() if (randomize and training) else range(k)
    for i in slots:
        chunks[i] = F.gelu(chunks[i])
    return torch.cat(chunks, dim=-1)


def hybrid4(h, randomize, training):
    """8 slots: 6 GELU, one holding the product of the two linear chunks, one a passthrough.

    Width preserved and slot positions fixed, so ``fc2`` always reads the same feature in
    the same place -- only the function applied to a slot varies under randomisation.
    """
    c = list(h.chunk(8, dim=-1))
    perm = torch.randperm(8).tolist() if (randomize and training) else list(range(8))
    gelu_slots, a, b = perm[:6], perm[6], perm[7]
    out = [None] * 8
    for i in gelu_slots:
        out[i] = F.gelu(c[i])
    out[a] = c[a] * c[b]
    out[b] = c[a] if (randomize and training and torch.rand(1).item() < 0.5) else c[b]
    return torch.cat(out, dim=-1)


def hybrid5(h, randomize, training):
    """8 slots: 6 always GELU, the 2 linear slots each gated by a chosen GELU'd chunk.

    Every slot's function CLASS is fixed; only the operand is drawn. Because ``c[6]`` is
    independent of the draw, ``E[c6 * g_j] = c6 * mean(g)`` is an EXACT eval rule.
    """
    c = list(h.chunk(8, dim=-1))
    g = [F.gelu(x) for x in c[:6]]
    if randomize and training:
        ja, jb = int(torch.randint(6, (1,))), int(torch.randint(6, (1,)))
        pair = (c[6] * g[ja], c[7] * g[jb])
    elif randomize:
        mean_g = sum(g) / 6.0
        pair = (c[6] * mean_g, c[7] * mean_g)
    else:
        pair = (c[6] * g[0], c[7] * g[1])
    return torch.cat(g + list(pair), dim=-1)


def parse_variant(spec: str):
    """Turn a spec like 'stoch:8:12:random' into (name, kwargs, fc1_width, fc2_width)."""
    parts = spec.split(":")
    name = parts[0]
    if name == "gelu":
        return name, {}, H, H
    if name == "geglu":
        return name, {}, 4 * H // 3, 2 * H // 3
    randomize = parts[-1] != "fixed"
    if name == "stoch":
        return name, {"k": int(parts[1]), "n": int(parts[2]), "randomize": randomize}, H, H
    if name in ("hybrid4", "hybrid5"):
        return name, {"randomize": randomize}, H, H
    raise SystemExit(f"unknown variant: {spec}")


def apply_act(name, h, kwargs, training):
    if name == "gelu":
        return F.gelu(h)
    if name == "geglu":
        return geglu(h)
    if name == "stoch":
        return stoch(h, kwargs["n"], kwargs["k"], kwargs["randomize"], training)
    if name == "hybrid4":
        return hybrid4(h, kwargs["randomize"], training)
    return hybrid5(h, kwargs["randomize"], training)


class Block(nn.Module):
    def __init__(self, name, kwargs, w_in, w_out):
        super().__init__()
        self.name, self.kwargs = name, kwargs
        self.norm = nn.LayerNorm(D)
        self.fc1, self.fc2 = nn.Linear(D, w_in), nn.Linear(w_out, D)

    def forward(self, x):
        h = apply_act(self.name, self.fc1(self.norm(x)), self.kwargs, self.training)
        return x + self.fc2(h)


class Net(nn.Module):
    def __init__(self, name, kwargs, w_in, w_out, depth=4):
        super().__init__()
        self.inp = nn.Linear(8, D)
        self.blocks = nn.ModuleList([Block(name, kwargs, w_in, w_out) for _ in range(depth)])
        self.out = nn.Linear(D, 1)

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def target(x):
    """Nonlinear with multiplicative interactions, so gating has something to earn."""
    return (torch.sin(3 * x[:, :1]) * x[:, 1:2]
            + (x[:, 2:3] ** 2) * torch.tanh(x[:, 3:4])
            + x[:, 4:5] * x[:, 5:6] * x[:, 6:7]
            - 0.5 * torch.cos(2 * x[:, 7:8]))


def train_once(spec, seed, steps):
    name, kwargs, w_in, w_out = parse_variant(spec)
    torch.manual_seed(seed)
    net = Net(name, kwargs, w_in, w_out)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    grad_max = 0.0
    for _ in range(steps):
        x = torch.randn(256, 8)
        loss = F.mse_loss(net(x), target(x))
        opt.zero_grad()
        loss.backward()
        grad_max = max(grad_max, torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9).item())
        opt.step()
    net.eval()
    x_test = torch.randn(4000, 8)
    with torch.no_grad():
        mse = F.mse_loss(net(x_test), target(x_test)).item()
    return sum(p.numel() for p in net.parameters()), mse, grad_max


def grad_scan(n_elem=320_000):
    """Max |grad| through each activation as the input scale grows."""
    print(f"{'scale':>6s} {'gelu':>8s} {'geglu':>9s} {'stoch':>9s} {'hybrid4':>9s} {'hybrid5':>9s}")
    for scale in (1.0, 2.0, 4.0, 8.0):
        row = []
        for fn in (lambda t: F.gelu(t),
                   geglu,
                   lambda t: stoch(t, 4, 2, True, True),
                   lambda t: hybrid4(t, True, True),
                   lambda t: hybrid5(t, True, True)):
            x = (torch.randn(1, n_elem) * scale).requires_grad_(True)
            fn(x).sum().backward()
            row.append(x.grad.abs().max().item())
        print(f"{scale:6.0f} " + " ".join(f"{v:9.2f}" for v in row))


DEFAULT = ["gelu", "geglu", "stoch:2:4:random", "stoch:2:4:fixed",
           "hybrid4:random", "hybrid4:fixed", "hybrid5:random", "hybrid5:fixed"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", metavar="SPEC",
                    help="Repeatable. gelu | geglu | stoch:K:N:{random,fixed} | "
                         "hybrid4:{random,fixed} | hybrid5:{random,fixed}")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--grad-scan", action="store_true",
                    help="Run the gradient-vs-input-scale scan instead of training.")
    args = ap.parse_args()

    if args.grad_scan:
        grad_scan()
        return 0

    print(f"{'variant':22s} {'params':>8s} {'test MSE':>10s} {'+/-':>7s} {'grad max':>9s}")
    for spec in (args.variant or DEFAULT):
        runs = [train_once(spec, s, args.steps) for s in range(args.seeds)]
        mses = [r[1] for r in runs]
        print(f"{spec:22s} {runs[0][0]:8,d} {sum(mses)/len(mses):10.4f} "
              f"{(max(mses)-min(mses))/2:7.4f} {max(r[2] for r in runs):9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
