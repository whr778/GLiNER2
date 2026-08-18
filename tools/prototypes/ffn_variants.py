"""Drop-in FFN blocks: plain GELU, GeGLU, and fixed partial gating (hybrid4).

Transformer-shaped versions of the activations explored in `activation_variants.py`,
packaged as `nn.Module` so they can be substituted into a real encoder. The toy study
is in `PARTIAL_GATING.md`; this module exists so the claim can be tested at depth.

**Parameter matching is the whole point of the comparison, so it is done here rather
than left to the caller.** Use `matched_d_ff()` to size each variant against a plain
GELU FFN of the given `d_ff`:

* plain GELU  -- `d_model -> d_ff -> d_model`, 2*d_model*d_ff params.
* GeGLU       -- needs `d_model -> 2*d_ff' -> d_ff' -> d_model`; matching requires
  `d_ff' = 2/3 d_ff`, the convention from Shazeer, arXiv:2002.05202.
* hybrid4     -- preserves width, so `d_ff' = d_ff` unchanged and it is exactly
  parameter-identical to plain GELU with no adjustment at all.

`Hybrid4FFN`'s product chunk IS Shazeer's `Bilinear` variant (a GLU with the
nonlinearity dropped), applied to one slot in eight rather than to every channel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def matched_d_ff(kind: str, d_ff: int, n_chunks: int = 8) -> int:
    """Inner width that makes `kind` parameter-match a plain GELU FFN of `d_ff`."""
    if kind == "geglu":
        return int(round(d_ff * 2 / 3))
    if kind == "hybrid4":
        return d_ff - (d_ff % n_chunks)      # only needs to divide evenly
    return d_ff


class GeluFFN(nn.Module):
    """Baseline: `fc2(gelu(fc1(x)))`."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class GeGLUFFN(nn.Module):
    """ModernBERT/mmBERT-style gated FFN: `fc2(v * gelu(g))`, fused up-projection."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        value, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(value * F.gelu(gate))


class Hybrid4FFN(nn.Module):
    """Fixed partial gating: 6 of 8 slots GELU, one `a*b` product, one passthrough.

    Width is preserved, so no up-projection is needed and the block is
    parameter-identical to `GeluFFN`. Slot positions are fixed, so `fc2` always reads
    the same feature in the same place.

    The assignment is deliberately NOT randomised. Randomising which slots receive the
    nonlinearity cost ~10x in the toy study (`PARTIAL_GATING.md` section 5): a slot whose
    FUNCTION CLASS changes between steps cannot have a stable downstream weight, and
    unlike dropout there is no scalar correction, because a nonlinearity does not commute
    with the expectation.
    """

    def __init__(self, d_model: int, d_ff: int, n_chunks: int = 8, n_gelu: int = 6,
                 bias: bool = False):
        super().__init__()
        if d_ff % n_chunks:
            raise ValueError(f"d_ff={d_ff} must be divisible by n_chunks={n_chunks}")
        if n_gelu != n_chunks - 2:
            raise ValueError("layout needs exactly two non-GELU slots: the product and "
                             f"the passthrough (got n_gelu={n_gelu}, n_chunks={n_chunks})")
        self.n_chunks, self.n_gelu = n_chunks, n_gelu
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

    def activate(self, h):
        chunks = list(h.chunk(self.n_chunks, dim=-1))
        out = [F.gelu(c) for c in chunks[:self.n_gelu]]
        gate, keep = chunks[self.n_gelu], chunks[self.n_gelu + 1]
        out.append(gate * keep)       # Bilinear (Shazeer 2020), one slot only
        out.append(keep)              # passthrough
        return torch.cat(out, dim=-1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


BUILDERS = {"gelu": GeluFFN, "geglu": GeGLUFFN, "hybrid4": Hybrid4FFN}


def build_ffn(kind: str, d_model: int, d_ff: int, bias: bool = False) -> nn.Module:
    """Build `kind` already sized to parameter-match a plain GELU FFN of `d_ff`."""
    if kind not in BUILDERS:
        raise ValueError(f"unknown FFN kind {kind!r}; choose from {sorted(BUILDERS)}")
    return BUILDERS[kind](d_model, matched_d_ff(kind, d_ff), bias=bias)
