"""Flipped-gate GEGLU (fg-GEGLU): a train-time stochastic gated activation that is plain GELU at inference.

The activation's input (a layer's pre-activation, width D) is split along the last dimension into `chunks` equal
pieces (an even number). On each training forward, `gates` of those pieces are drawn at random from every piece
except a fixed passive one (the dendrite that stops propagation: it is never gated and passes through unchanged).
The drawn pieces go through GELU and are multiplied together elementwise; that product replaces each drawn piece in
its own slot, so the output keeps the input's width. Undrawn pieces pass through unchanged. In eval mode the module
is exactly `F.gelu(x)`.

Zero parameters. The random gate pattern acts like dropout on activations: measured in the user's earlier work as a
normalizing effect that reduced the gap between dev and blind test.
"""
import torch
import torch.nn.functional as F
from torch import nn


class FlippedGateGELU(nn.Module):
    """fg-GEGLU over the last dimension; `chunks` even, `1 <= gates <= chunks - 1`, `passive` never gated."""

    def __init__(self, chunks: int = 4, gates: int = 2, passive: int = 0) -> None:
        super().__init__()
        if chunks < 2 or chunks % 2:
            raise ValueError(f"chunks must be an even number >= 2, got {chunks}")
        if not 1 <= gates <= chunks - 1:
            raise ValueError(f"gates must be in [1, chunks - 1], got {gates} for {chunks} chunks")
        if not 0 <= passive < chunks:
            raise ValueError(f"passive chunk {passive} out of range for {chunks} chunks")
        self.chunks, self.gates, self.passive = chunks, gates, passive

    def draw_gates(self, device) -> torch.Tensor:
        """Indices of this forward's gated chunks: `gates` drawn without replacement, never the passive one."""
        eligible = torch.tensor([i for i in range(self.chunks) if i != self.passive], device=device)
        return eligible[torch.randperm(len(eligible), device=device)[: self.gates]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return F.gelu(x)
        if x.shape[-1] % self.chunks:
            raise ValueError(f"last dimension {x.shape[-1]} is not divisible by {self.chunks} chunks")
        pieces = list(x.chunk(self.chunks, dim=-1))
        gated = self.draw_gates(x.device).tolist()
        product = F.gelu(pieces[gated[0]])
        for i in gated[1:]:
            product = product * F.gelu(pieces[i])
        for i in gated:
            pieces[i] = product
        return torch.cat(pieces, dim=-1)

    def extra_repr(self) -> str:
        return f"chunks={self.chunks}, gates={self.gates}, passive={self.passive}"
