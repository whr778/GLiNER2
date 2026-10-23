"""Flipped-gate GEGLU (fg-GEGLU): a stochastic gated activation with a choice of inference rule.

The activation's input (a layer's pre-activation, width D) is split along the last dimension into `chunks` equal
pieces (an even number). On each training forward two disjoint random sets of pieces are drawn: `gates` activated
pieces and `skipped` pieces. The activated pieces go through GELU and are multiplied together elementwise; that
product replaces each activated piece in its own slot. The skipped pieces pass through unchanged (the dendrite that
stops propagation). Every other piece gets plain GELU. The output keeps the input's width.

Inference (`eval_mode`): "gelu" is plain F.gelu(x) (the original rule; measured on the toy to leave a train/eval gap
the network does not close); "mc" averages `mc_samples` random gate patterns, the Monte Carlo estimate of the
training-time expectation (measured on the toy to beat GELU for 6 chunks / 2 gates / 2 skipped).

Zero parameters. The random gate pattern acts like dropout on activations: measured in the user's earlier work as a
normalizing effect that reduced the gap between dev and blind test, at the cost of more training steps.
"""
import torch
import torch.nn.functional as F
from torch import nn


class FlippedGateGELU(nn.Module):
    """fg-GEGLU over the last dimension; `chunks` even, `gates >= 1`, `skipped >= 0`, `gates + skipped <= chunks`."""

    def __init__(self, chunks: int = 12, gates: int = 3, skipped: int = 3, eval_mode: str = "mc", mc_samples: int = 8) -> None:
        # defaults = the toy sweep's best stable configuration under the MC rule (FG_GEGLU.md section 3.3); the layer
        # width must be divisible by `chunks` (768 is; a 64-wide MLP takes chunks=8)
        super().__init__()
        if chunks < 2 or chunks % 2:
            raise ValueError(f"chunks must be an even number >= 2, got {chunks}")
        if gates < 1 or skipped < 0 or gates + skipped > chunks:
            raise ValueError(f"need gates >= 1, skipped >= 0 and gates + skipped <= chunks; got {gates}, {skipped}, {chunks}")
        if eval_mode not in ("gelu", "mc"):
            raise ValueError(f"eval_mode must be 'gelu' or 'mc', got {eval_mode!r}")
        if mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {mc_samples}")
        self.chunks, self.gates, self.skipped = chunks, gates, skipped
        self.eval_mode, self.mc_samples = eval_mode, mc_samples

    def draw(self, device) -> tuple[list[int], list[int]]:
        """This forward's (activated, skipped) chunk indices: disjoint draws without replacement."""
        order = torch.randperm(self.chunks, device=device).tolist()
        return order[: self.gates], order[self.gates: self.gates + self.skipped]

    def gated(self, x: torch.Tensor) -> torch.Tensor:
        """One random gate pattern applied to x."""
        if x.shape[-1] % self.chunks:
            raise ValueError(f"last dimension {x.shape[-1]} is not divisible by {self.chunks} chunks")
        pieces = list(x.chunk(self.chunks, dim=-1))
        activated, skipped = self.draw(x.device)
        product = F.gelu(pieces[activated[0]])
        for i in activated[1:]:
            product = product * F.gelu(pieces[i])
        out = []
        for i, piece in enumerate(pieces):
            if i in activated:
                out.append(product)
            elif i in skipped:
                out.append(piece)
            else:
                out.append(F.gelu(piece))
        return torch.cat(out, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.gated(x)
        if self.eval_mode == "gelu":
            return F.gelu(x)
        return torch.stack([self.gated(x) for _ in range(self.mc_samples)]).mean(0)

    def extra_repr(self) -> str:
        return f"chunks={self.chunks}, gates={self.gates}, skipped={self.skipped}, eval_mode={self.eval_mode}, mc_samples={self.mc_samples}"
