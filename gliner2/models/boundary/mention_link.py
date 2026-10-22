"""Entity-mention COREFERENCE link: are two mentions in one document the same entity? (COREF_DATA_SOURCES.md)

s_ij = <W m_i, W m_j> / sqrt(d) + w * log1p(word distance) + b, symmetric, where m is the MEAN of the
encoder's word states over the mention -- the representation the separability probe measured (eb21,
VAL pair AUC: CMNEE 0.92 untrained; English 0.66, pronouns 0.58). W is randomly initialised: English
cosine barely beats string match, so there is no state worth starting from. The distance term starts
at zero. Nothing else reads the link, so with it off or on every other loss and score is unchanged.

Trained on gold ``coref_clusters`` (processing/coref.py): every positive pair (same cluster) and, per
mention, the ``negatives`` highest-scoring mentions of OTHER clusters. Its own mean, weighted.
"""
from __future__ import annotations

import logging
import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gliner2.models.boundary.distance_bias import build_distance_bias

logger = logging.getLogger(__name__)


class MentionLink(nn.Module):
    def __init__(self, hidden_size: int, dim: int, distance: str = "scalar", distance_dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(hidden_size, dim)
        # "scalar" = the historical zero-initialised Linear(1, 1) under the name `dist`; "mlp" = a zero-at-init MLP
        # on log1p(distance) (boundary_head.coref_link_distance, shared with TriggerLink; distance_bias.py).
        self.distance, self.distance_dim = distance, distance_dim
        self.dist = build_distance_bias(distance, distance_dim)

    def forward(self, states: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
        """[M, M] logits for mention states [M, H] at word midpoints [M]."""
        z = self.w(states)
        dist = torch.log1p((mid[:, None] - mid[None, :]).abs()).unsqueeze(-1)
        return z @ z.t() / math.sqrt(self.dim) + self.dist(dist).squeeze(-1)


def mention_states(text_states: torch.Tensor, mentions: Sequence[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean word state per mention [M, H] and its word midpoint [M], for one sample's [W, H] states."""
    idx = torch.tensor([(s, e) for s, e, _ in mentions], device=text_states.device)
    cum = torch.cat([text_states.new_zeros(1, text_states.shape[1]), text_states.cumsum(0)])
    sums = cum[idx[:, 1] + 1] - cum[idx[:, 0]]
    return sums / (idx[:, 1] - idx[:, 0] + 1).unsqueeze(-1).to(sums.dtype), idx.float().mean(-1)


def pair_loss(logits: torch.Tensor, clusters: torch.Tensor, k: int) -> Tuple[torch.Tensor, int, int, int]:
    """(summed BCE, n pairs, wins, ranked) over all positives + each mention's k hardest negatives.
    ``wins/ranked`` is the in-run pair AUC: positives scoring above negatives."""
    m = logits.shape[0]
    same = clusters[:, None] == clusters[None, :]
    upper = torch.ones(m, m, dtype=torch.bool, device=logits.device).triu(1)
    pos = same & upper
    neg = torch.zeros_like(pos)
    if k > 0 and m > 1:
        masked = logits.detach().masked_fill(same, float("-inf"))
        top = masked.topk(min(k, m - 1), dim=-1)
        neg.scatter_(1, top.indices, torch.isfinite(top.values))
        neg = (neg | neg.t()) & upper
    p, n = logits[pos], logits[neg]
    if p.numel() + n.numel() == 0:
        return logits.new_zeros(()), 0, 0, 0
    scores = torch.cat([p, n])
    target = torch.cat([torch.ones_like(p), torch.zeros_like(n)])
    loss = F.binary_cross_entropy_with_logits(scores.float(), target.float(), reduction="sum")
    wins = ranked = 0
    if p.numel() and n.numel():
        pd, nd = p.detach().float(), n.detach().float()
        wins = int((pd[:, None] > nd[None, :]).sum())
        ranked = pd.numel() * nd.numel()
    return loss, int(scores.numel()), wins, ranked


def batch_loss(link: MentionLink, text_states: torch.Tensor, coref_mentions: Sequence[List[Tuple[int, int, int]]],
               k: int) -> Tuple[torch.Tensor, int]:
    """Mean pair BCE over the batch's samples that carry >= 2 placed mentions; (loss, n pairs)."""
    total, count, wins, ranked = text_states.new_zeros(()), 0, 0, 0
    for b, mentions in enumerate(coref_mentions):
        if len(mentions) < 2:
            continue
        states, mid = mention_states(text_states[b], mentions)
        clusters = torch.tensor([c for _, _, c in mentions], device=states.device)
        loss, n, w, r = pair_loss(link(states, mid), clusters, k)
        total, count, wins, ranked = total + loss, count + n, wins + w, ranked + r
    _note(count, wins, ranked)
    return (total / count if count else total), count


_NOTE = {"pairs": 0, "wins": 0, "ranked": 0, "calls": 0}


def _note(n_pairs: int, wins: int, ranked: int) -> None:
    """In-run proof the link trains, and its pair AUC on training batches, with backoff."""
    if not n_pairs:
        return
    _NOTE["pairs"] += n_pairs
    _NOTE["wins"] += wins
    _NOTE["ranked"] += ranked
    _NOTE["calls"] += 1
    n = _NOTE["calls"]
    if n in (1, 200, 1000, 5000) or (n > 5000 and n % 25000 == 0):
        logger.info("entity coref link: %d mention pairs trained; coreferent pairs outscore non-coreferent in "
                    "%.3f of %d (pair AUC, cumulative over %d batches)",
                    _NOTE["pairs"], _NOTE["wins"] / max(_NOTE["ranked"], 1), _NOTE["ranked"], n)
