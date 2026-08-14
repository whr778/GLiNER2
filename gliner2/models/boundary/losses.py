"""Masked boundary/pair/inside losses.

Start and end objectives are multi-label BCE (nested spans may share a
boundary — never a softmax over positions). All losses are masking-aware and
empty-query safe: denominators use ``clamp_min(1)`` so a query with no positive
span still contributes finite negative supervision without NaNs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from gliner2.models.boundary.constants import MASK_LOGIT


def _to_query_candidate(
    tensor: torch.Tensor, query_axis: int, candidate_axis: int
) -> torch.Tensor:
    """Canonicalize candidate scores to ``[B,Q,C]`` without copying."""
    return torch.movedim(tensor, (query_axis, candidate_axis), (1, 2))


def _from_query_candidate(
    tensor: torch.Tensor, query_axis: int, candidate_axis: int
) -> torch.Tensor:
    return torch.movedim(tensor, (1, 2), (query_axis, candidate_axis))


def _safe_bce(logits: torch.Tensor, targets: torch.Tensor, keep: torch.BoolTensor) -> torch.Tensor:
    """Elementwise BCE-with-logits with extreme masked logits neutralized."""
    safe_logits = torch.where(keep, logits, torch.zeros_like(logits))
    safe_targets = torch.where(keep, targets, torch.zeros_like(targets))
    return F.binary_cross_entropy_with_logits(safe_logits, safe_targets, reduction="none")


def _reduce(
    elementwise: torch.Tensor,
    keep: torch.BoolTensor,
    query_mask: Optional[torch.BoolTensor],
    mode: str,
    query_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce a masked ``[B, Q, N]`` loss globally or per active query.

    ``query_weights`` is an optional ``[B, Q]`` per-query scale, used to rebalance
    the loss across task types (entities vs relations vs events) — the boundary
    analogue of the phase-2 span model's per-task event loss.

    The denominator stays **unweighted** on purpose. A weighted mean would make
    the loss invariant to scaling every weight, so raising the event weight would
    do nothing; dividing by the plain active count means a weight of 2.0 really
    does double that task's share of the gradient. At all-ones this reduces
    exactly to the unweighted form.
    """
    keep_f = keep.to(elementwise.dtype)
    if query_weights is not None:
        query_weights = query_weights.to(elementwise.dtype)
    if mode == "global":
        weighted = elementwise * keep_f
        if query_weights is not None:
            weighted = weighted * query_weights.unsqueeze(-1)
        return weighted.sum() / keep_f.sum().clamp_min(1)
    if mode != "per_query":
        raise ValueError(f"unknown reduction mode {mode!r}")
    numerator = (elementwise * keep_f).sum(-1)
    denominator = keep_f.sum(-1).clamp_min(1)
    per_query = numerator / denominator
    active = keep.any(-1)
    if query_mask is not None:
        active = active & query_mask
    active_f = active.to(per_query.dtype)
    scaled = per_query * active_f
    if query_weights is not None:
        scaled = scaled * query_weights
    return scaled.sum() / active_f.sum().clamp_min(1)


def balanced_multilabel_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.BoolTensor,
    *,
    negative_weight: float = 1.0,
    query_mask: Optional[torch.BoolTensor] = None,
    reduction: str = "global",
    query_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean multi-label BCE over valid positions.

    Negative (non-boundary) positions are scaled by ``negative_weight``. The
    default of ``1.0`` applies **no** down-weighting; pass a value in ``(0, 1)``
    (wired from ``BoundaryHeadSettings.boundary_negative_weight``) to counter the
    negative-dominance of sparse boundary targets.
    """
    bce = _safe_bce(logits, targets, valid_mask)
    weight = torch.where(targets > 0.5, torch.ones_like(targets), torch.full_like(targets, negative_weight))
    return _reduce(bce * weight, valid_mask, query_mask, reduction, query_weights)


def asymmetric_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.BoolTensor,
    *,
    gamma_positive: float = 0.0,
    gamma_negative: float = 2.0,
    clip: float = 0.05,
    negative_weight: float = 1.0,
    query_mask: Optional[torch.BoolTensor] = None,
    reduction: str = "global",
    query_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Asymmetric focal loss for imbalanced multi-label boundary targets."""
    keep = valid_mask
    safe_logits = torch.where(keep, logits, torch.zeros_like(logits))
    p = torch.sigmoid(safe_logits)
    if clip > 0:
        p_neg = (1 - p + clip).clamp(max=1.0)
    else:
        p_neg = 1 - p
    los_pos = targets * torch.log(p.clamp_min(1e-8)) * ((1 - p) ** gamma_positive)
    los_neg = (
        (1 - targets)
        * torch.log(p_neg.clamp_min(1e-8))
        * (p ** gamma_negative)
        * negative_weight
    )
    return _reduce(-(los_pos + los_neg), valid_mask, query_mask, reduction, query_weights)


def build_candidate_labels(
    candidate_indices: torch.LongTensor,   # [B,Q,C,2] or pooled [B,C,2]
    candidate_mask: torch.BoolTensor,      # [B,Q,C] or pooled [B,C]
    gold_pairs: torch.LongTensor,          # [B, Q, G, 2]
    gold_mask: torch.BoolTensor,           # [B, Q, G]
    *,
    return_iou: bool = False,
    query_axis: int = 1,
    candidate_axis: int = 2,
):
    """Build labels in the caller's score-axis order.

    For a shared pool pass ``candidate_indices=[B,C,2]``,
    ``candidate_mask=[B,C]``, ``candidate_axis=1`` and ``query_axis=2``; the
    result is ``[B,C,Q]``.
    """
    pooled = candidate_indices.dim() == 3
    if pooled:
        cand = candidate_indices.unsqueeze(2).unsqueeze(3)  # [B,C,1,1,2]
        gold = gold_pairs.unsqueeze(1)                       # [B,1,Q,G,2]
        same = (cand == gold).all(dim=-1) & gold_mask.unsqueeze(1)
        labels = same.any(-1).to(torch.float) * candidate_mask.unsqueeze(-1)
        candidate_weight = candidate_mask.unsqueeze(-1).to(labels.dtype)
    else:
        canonical_indices = torch.movedim(
            candidate_indices, (query_axis, candidate_axis), (1, 2)
        )
        canonical_mask = _to_query_candidate(
            candidate_mask, query_axis, candidate_axis
        )
        cand = canonical_indices.unsqueeze(3)       # [B,Q,C,1,2]
        gold = gold_pairs.unsqueeze(2)              # [B,Q,1,G,2]
        same = (cand == gold).all(dim=-1) & gold_mask.unsqueeze(2)
        canonical_labels = same.any(dim=-1).to(torch.float)
        canonical_labels = canonical_labels * canonical_mask.to(canonical_labels.dtype)
        labels = _from_query_candidate(
            canonical_labels, query_axis, candidate_axis
        )
        candidate_weight = canonical_mask.to(canonical_labels.dtype)
    if not return_iou:
        return labels

    candidate_start, candidate_end = cand[..., 0], cand[..., 1]
    gold_start, gold_end = gold[..., 0], gold[..., 1]
    intersection = (
        torch.minimum(candidate_end, gold_end)
        - torch.maximum(candidate_start, gold_start)
    ).clamp_min(0)
    union = (
        candidate_end - candidate_start
        + gold_end - gold_start
        - intersection
    )
    iou = intersection.to(torch.float) / union.clamp_min(1).to(torch.float)
    iou = iou * (
        gold_mask.unsqueeze(1) if pooled else gold_mask.unsqueeze(2)
    ).to(iou.dtype)
    canonical_soft = iou.amax(-1) * candidate_weight
    soft_targets = (
        canonical_soft
        if pooled
        else _from_query_candidate(
            canonical_soft, query_axis, candidate_axis
        )
    )
    return labels, soft_targets


def apply_guide_veto(
    selected: torch.BoolTensor,       # [B, Q, C] the mined hard-negative mask
    guide_logits: torch.Tensor,       # [B, Q, C] scores from a FROZEN guide model
    labels: torch.Tensor,             # [B, Q, C]
    valid_mask: torch.BoolTensor,     # [B, Q, C]
    *,
    reference: Optional[torch.Tensor] = None,   # [B, Q, C]
    margin: float = 0.0,
    floor: float = 0.0,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.BoolTensor:
    """Drop mined negatives that a guide model judges to be positives here (GIST).

    Query-axis mining asks "which sibling type queries score this span highly?". Many of the
    answers are genuine confusions to train on, but some are simply CORRECT: `quantity` beats
    `death toll` on 64 of 250 gold casualty figures and is simultaneously the right answer for
    "1.2 million homes". Penalizing it unconditionally teaches the model that the type never
    applies, which is worse than the confusion. So a negative is removed from the loss --
    neither pushed down nor treated as positive -- when the guide ranks it above this
    candidate's own positive query.

    **The guide must be FROZEN and independent of the model being trained.** Self-guiding with
    the live model vetoes precisely the hard negatives it is meant to select, since a cell is
    mined *because* the live model scores it highly. A frozen pretrained checkpoint is the
    right source: measured on 40 gold records that name both a casualty type and a competing
    count type, `fastino/gliner2-base-v1` gets both directions right 82.5% of the time against
    a 25% chance baseline (EKF_MHT_DESIGN sec 27.7).

    ``margin`` raises the bar for a veto: a negative survives unless the guide prefers it by
    more than ``margin``. 0.0 reproduces GIST's rule.

    ``floor`` is the guide's abstention threshold, and it matters more than it looks. A
    rival can "outrank" a positive while both sit at zero -- measured on real cache entries,
    `President Bush` scored 0.00 under its gold type and 0.00 under a randomly drawn Chinese
    industry type. That is not the guide endorsing the rival, it is the guide having no
    opinion, and vetoing on it deletes negatives for no reason. Require the guide to actually
    score the rival above ``floor`` before its vote counts.

    **Axis warning.** Query-axis MINING is done by handing
    :func:`select_hard_negative_candidates` its axes *swapped*, which turns its "top-k
    candidates per query" into "top-k queries per candidate". This function takes the axes
    **as they really are** -- it has to reduce over queries to find each candidate's own
    positive, so a swapped call here silently compares along the wrong dimension and vetoes
    nothing. The two calls therefore take different axis arguments on the same tensors, and
    that is deliberate.

    Candidates with no positive query are left alone -- there is no reference to compare a
    guide score against, so vetoing there would be arbitrary.

    ``reference`` supplies that comparison point per cell instead of deriving it from
    ``labels``, and is REQUIRED whenever candidate columns are not shared across queries.
    Deriving it takes the max over the query axis at a fixed column, which silently assumes
    column *c* denotes the same span for every query -- true for ``candidate_pool="shared"``
    and false for the default ``"per_query"``, where each query proposes its own list. Under
    a per-query pool the derived path finds no positive at the rival's column and vetoes
    nothing at all, so the wiring in :mod:`gliner2.models.boundary.guide` resolves the
    reference by span identity and passes it here. A non-positive ``reference`` means the
    guide never scored that span under its own gold type, and nothing is vetoed there.
    """
    sel = _to_query_candidate(selected, query_axis, candidate_axis)
    guide = _to_query_candidate(guide_logits, query_axis, candidate_axis)
    lab = _to_query_candidate(labels, query_axis, candidate_axis)
    valid = _to_query_candidate(valid_mask, query_axis, candidate_axis)

    positive = (lab > 0.5) & valid
    if reference is None:
        # `neg_inf` masks non-positives out of the max; do NOT call it `floor`, which is
        # the caller's abstention threshold -- shadowing it made `guide > floor` always
        # true and silently disabled the abstention check.
        neg_inf = torch.finfo(guide.dtype).min
        # Best guide score among this candidate's OWN positive queries, per (batch, candidate).
        pos_guide = guide.masked_fill(~positive, neg_inf).amax(dim=1, keepdim=True)
        has_reference = positive.any(dim=1, keepdim=True)
    else:
        pos_guide = _to_query_candidate(reference, query_axis, candidate_axis)
        has_reference = pos_guide > 0

    vetoed = (sel & has_reference & (guide > pos_guide + margin) & (guide > floor)
              & ~positive)
    return _from_query_candidate(sel & ~vetoed, query_axis, candidate_axis)


def select_hard_negative_candidates(
    pair_logits: torch.Tensor,        # [B, Q, C]
    labels: torch.Tensor,             # [B, Q, C]
    valid_mask: torch.BoolTensor,     # [B, Q, C]
    *,
    negatives_per_positive: int,
    minimum_negatives: int,
    keep_all_when_no_positive: bool = False,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.BoolTensor:
    """Select the highest-scoring negatives per query (positives always kept)."""
    original_query_axis, original_candidate_axis = query_axis, candidate_axis
    pair_logits = _to_query_candidate(pair_logits, query_axis, candidate_axis)
    labels = _to_query_candidate(labels, query_axis, candidate_axis)
    valid_mask = _to_query_candidate(valid_mask, query_axis, candidate_axis)
    positive = (labels > 0.5) & valid_mask
    negative = (labels <= 0.5) & valid_mask
    n_positive = positive.sum(-1, keepdim=True)
    floor = torch.finfo(pair_logits.dtype).min
    negative_scores = pair_logits.masked_fill(~negative, floor)
    order = torch.argsort(negative_scores, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    n_keep = (negatives_per_positive * n_positive).clamp_min(minimum_negatives)
    if keep_all_when_no_positive:
        n_keep = torch.where(
            n_positive == 0,
            torch.full_like(n_keep, pair_logits.shape[-1]),
            n_keep,
        )
    selected = positive | (negative & (rank < n_keep))
    return _from_query_candidate(
        selected, original_query_axis, original_candidate_axis
    )


def candidate_pair_loss(
    logits: torch.Tensor,             # [B, Q, C]
    labels: torch.Tensor,             # [B, Q, C]
    valid_mask: torch.BoolTensor,     # [B, Q, C]
    hard_negative_mask: Optional[torch.BoolTensor] = None,
    *,
    query_mask: Optional[torch.BoolTensor] = None,
    reduction: str = "global",
    query_axis: int = 1,
    candidate_axis: int = 2,
    query_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCE over candidates; optionally restrict negatives to a hard subset."""
    logits = _to_query_candidate(logits, query_axis, candidate_axis)
    labels = _to_query_candidate(labels, query_axis, candidate_axis)
    valid_mask = _to_query_candidate(valid_mask, query_axis, candidate_axis)
    if hard_negative_mask is not None:
        hard_negative_mask = _to_query_candidate(
            hard_negative_mask, query_axis, candidate_axis
        )
    if hard_negative_mask is not None:
        effective = valid_mask & ((labels > 0.5) | hard_negative_mask)
    else:
        effective = valid_mask
    bce = _safe_bce(logits, labels, effective)
    return _reduce(bce, effective, query_mask, reduction, query_weights)


def inside_consistency_loss(
    inside_logits: torch.Tensor,      # [B, Q, L]
    inside_targets: torch.Tensor,     # [B, Q, L]
    text_mask: torch.BoolTensor,      # [B, L]
    query_mask: torch.BoolTensor,     # [B, Q]
    *,
    negative_weight: float = 1.0,
    reduction: str = "global",
) -> torch.Tensor:
    keep = text_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
    return balanced_multilabel_bce(
        inside_logits,
        inside_targets,
        keep,
        negative_weight=negative_weight,
        query_mask=query_mask,
        reduction=reduction,
    )


def proposal_listwise_loss(
    proposal_logits: torch.Tensor,
    gold_mask: torch.BoolTensor,
    valid_mask: torch.BoolTensor,
    query_mask: torch.BoolTensor,
    *,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.Tensor:
    """Rank injected gold candidates above other valid proposals."""
    proposal_logits = _to_query_candidate(
        proposal_logits, query_axis, candidate_axis
    )
    gold_mask = _to_query_candidate(gold_mask, query_axis, candidate_axis)
    valid_mask = _to_query_candidate(valid_mask, query_axis, candidate_axis)
    floor = MASK_LOGIT
    logits = proposal_logits.masked_fill(~valid_mask, floor)
    all_lse = torch.logsumexp(logits, dim=-1)
    gold_lse = torch.logsumexp(logits.masked_fill(~gold_mask, floor), dim=-1)
    has_gold = gold_mask.any(-1) & query_mask
    loss = torch.where(
        has_gold, all_lse - gold_lse, torch.zeros_like(all_lse)
    )
    return loss.sum() / has_gold.to(loss.dtype).sum().clamp_min(1)


def reranker_listwise_loss(
    pair_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.BoolTensor,
    query_mask: torch.BoolTensor,
    *,
    query_axis: int = 1,
    candidate_axis: int = 2,
) -> torch.Tensor:
    """Listwise gold-mass loss over reranked candidates, empty-query safe."""
    gold_mask = (labels > 0.5) & valid_mask
    return proposal_listwise_loss(
        pair_logits,
        gold_mask,
        valid_mask,
        query_mask,
        query_axis=query_axis,
        candidate_axis=candidate_axis,
    )


def marginal_pair_consistency_loss(
    pair_logits: torch.Tensor,
    indices: torch.LongTensor,
    valid_mask: torch.BoolTensor,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    boundary_keep: torch.BoolTensor,
) -> torch.Tensor:
    """Match boundary marginals to candidate-level noisy-OR probabilities."""
    probabilities = torch.sigmoid(pair_logits) * valid_mask.to(pair_logits.dtype)
    log_survival = torch.log1p(-probabilities.clamp(max=1.0 - 1e-6))
    b, q, n = start_logits.shape

    def accumulate(index: torch.LongTensor):
        total = torch.zeros(
            b, q, n, dtype=log_survival.dtype, device=log_survival.device
        )
        total.scatter_add_(2, index, log_survival)
        count = torch.zeros_like(total)
        count.scatter_add_(2, index, valid_mask.to(total.dtype))
        return 1.0 - torch.exp(total), count > 0

    predicted_start, reached_start = accumulate(indices[..., 0])
    predicted_end, reached_end = accumulate(indices[..., 1])
    result = pair_logits.new_zeros(())
    for predicted, marginal, reached in (
        (predicted_start, start_logits, reached_start),
        (predicted_end, end_logits, reached_end),
    ):
        keep = reached & boundary_keep
        target = torch.sigmoid(torch.where(keep, marginal, torch.zeros_like(marginal)))
        squared = (predicted - target) ** 2
        result = result + (squared * keep).sum() / keep.sum().clamp_min(1)
    return result * 0.5


def abstention_loss(
    null_logits: torch.Tensor,
    mention_mask: torch.BoolTensor,
    query_mask: torch.BoolTensor,
) -> torch.Tensor:
    """Train a per-query gate whose target is one for an absent query."""
    target = (~mention_mask.any(-1)).to(null_logits.dtype)
    elementwise = F.binary_cross_entropy_with_logits(
        null_logits, target, reduction="none"
    )
    keep = query_mask.to(elementwise.dtype)
    return (elementwise * keep).sum() / keep.sum().clamp_min(1)


def count_log_rate_loss(
    count_log_rate: torch.Tensor,
    mention_mask: torch.BoolTensor,
    query_mask: torch.BoolTensor,
) -> torch.Tensor:
    """Poisson NLL for a count head whose output is the logarithm of its rate."""
    target = mention_mask.sum(-1).to(count_log_rate.dtype)
    elementwise = F.poisson_nll_loss(
        count_log_rate,
        target,
        log_input=True,
        full=False,
        reduction="none",
    )
    keep = query_mask.to(elementwise.dtype)
    return (elementwise * keep).sum() / keep.sum().clamp_min(1)
