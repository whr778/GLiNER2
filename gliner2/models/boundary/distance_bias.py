"""The pairwise distance term of a coreference link (TriggerLink, MentionLink).

``scalar`` (historical): ``Linear(1, 1)`` on log1p(distance), zero-initialised, parameters ``dist.weight`` /
``dist.bias``. ``mlp``: ``Linear(1, dim) -> GELU -> Linear(dim, 1)`` with the LAST layer zero-initialised, so either
kind is exactly 0 at init and a warm start is bit-identical. Measured 2026-10-22 on the eb22c control at 30k: the
scalar term is 66-68% of the coref score and REMOVING it raises the coref pair AUC (CASIE 0.932 -> 0.944, CMNEE
0.743 -> 0.814) -- one monotone slope in log distance is wrong often enough to cost ranking.

RNG parity: the scalar ``Linear(1, 1)`` is always built first, consuming the global RNG exactly as the historical
link did, and the mlp is built on the meta device and drawn from a generator forked off the seed -- so every module
built after the link keeps its init at any kind or width (the one-change property an A/B needs).
"""
import math

import torch
from torch import nn


def build_distance_bias(kind: str, dim: int) -> nn.Module:
    """``scalar``: the historical zero-initialised Linear(1, 1). ``mlp``: Linear(1, dim) -> GELU -> Linear(dim, 1)."""
    scalar = nn.Linear(1, 1)
    nn.init.zeros_(scalar.weight)
    nn.init.zeros_(scalar.bias)
    if kind == "scalar":
        return scalar
    if kind != "mlp":
        raise ValueError(f"distance bias kind must be 'scalar' or 'mlp', got {kind!r}")
    fork = torch.Generator().manual_seed(torch.initial_seed() % (2 ** 63) + 7 * dim)
    dev = torch.get_default_device()
    first = nn.Linear(1, dim, device="meta").to_empty(device=dev)
    nn.init.kaiming_uniform_(first.weight, a=math.sqrt(5), generator=fork)
    nn.init.uniform_(first.bias, -1.0, 1.0, generator=fork)   # fan_in 1: the default Linear bound
    last = nn.Linear(dim, 1, device="meta").to_empty(device=dev)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(first, nn.GELU(), last)
