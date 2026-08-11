"""Turn a cached frozen-guide opinion into the ``[B,Q,C]`` tensor the GIST veto needs.

The cache is sparse and text-keyed: per record, a handful of gold surfaces scored under the
record's own types and a few cross-record rivals. The veto is dense and candidate-keyed.
The gap is bridged in two steps because candidate columns do not exist until the head runs:

    build_sparse   outer forward, once per batch -- resolve cached surfaces to gold token
                   pairs, giving (spans [B,S,2], values [B,S,Q], reference [B,S], mask [B,S])
    densify        inside the loss -- match those token pairs against the live candidate
                   indices and scatter into the score tensor's own axis order

Only injected-rival cells are ever filled. Every other cell stays at exactly 0.0, which
with ``floor >= 0`` and a strict comparison cannot veto -- that is what keeps own-record
absent queries, where gold is authoritative, out of the veto's reach.

The span's own gold score travels SEPARATELY, as ``reference``, because the default
``candidate_pool="per_query"`` gives every query its own candidate list: column 3 of the
`person` query and column 3 of the `company` query are different spans, so the veto cannot
find a span's positive by looking down the query axis at a fixed column. Resolving the
reference by span identity here is what makes the veto work under either pool.

Counters are kept because every failure mode here is silent. A wrong hash, a mismatched
surface key or a swapped axis all present identically as "the veto changed nothing", and
this codebase has already shipped that failure twice. Read them with :func:`guide_counts`.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Optional, Tuple

import torch

from gliner2.training.guide_scores import surface_key

logger = logging.getLogger(__name__)

_COUNTS: "Counter[str]" = Counter()


def guide_counts() -> "Counter[str]":
    """Cache hit rates and veto activity accumulated so far.

    ``samples`` / ``samples_cached`` says whether the cache is being found at all;
    ``gold_spans`` / ``spans_matched`` whether surfaces line up; ``cells`` how much the
    guide actually has an opinion on; ``vetoed`` whether any of it changed the loss.
    """
    return _COUNTS


def reset_guide_counts() -> None:
    _COUNTS.clear()


def build_sparse_guide(
    batch: Any,
    targets: Any,
    guide_scores: Any,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Resolve cached spans to gold token pairs.

    Returns ``(spans [B,S,2], values [B,S,Q], reference [B,S], mask [B,S])``, or ``None``
    when no sample in the batch has a cache entry, so the caller can skip the veto entirely
    rather than multiply by an all-zero tensor.
    """
    pairs, mask = targets.mention_pairs, targets.mention_mask
    n_batch, n_query = int(pairs.shape[0]), int(pairs.shape[1])
    # ONE host transfer for the whole batch. Indexing gold slot by slot would cost a
    # device sync per element -- tens of thousands per step, and on an accelerator that
    # dwarfs everything the veto could save.
    gold_index = mask.nonzero().tolist()
    gold_pairs = pairs.tolist()
    _COUNTS["gold_spans"] += len(gold_index)

    per_sample = [{} for _ in range(n_batch)]
    entries: dict = {}
    for index in range(n_batch):
        _COUNTS["samples"] += 1
        entry = guide_scores.get(batch.original_texts[index])
        if entry is None:
            continue
        _COUNTS["samples_cached"] += 1
        layout = batch.query_layouts[index]
        names = {qid: layout.query(qid).role_name
                 for qid in range(layout.extractive_count())}
        rivals = [qid for qid, name in names.items() if name in entry.rival]
        _COUNTS["rival_queries"] += len(rivals)
        entries[index] = (entry, names, rivals)

    for index, query_id, gold in gold_index:
        found = entries.get(index)
        if found is None:
            continue
        entry, names, rivals = found
        start, end = gold_pairs[index][query_id][gold]
        row = entry.spans.get(surface_key(" ".join(batch.text_tokens[index][start:end])))
        if row is None:
            continue
        _COUNTS["spans_matched"] += 1
        spans = per_sample[index]
        cells, own = spans.get((start, end), ({}, 0.0))
        # The span's own gold score is the reference, kept out of `cells` so it can be
        # compared against by span rather than by candidate column.
        own = max(own, row.get(names.get(query_id, ""), 0.0))
        for rival_id in rivals:
            value = row.get(names[rival_id], 0.0)
            if value > 0.0:
                cells[rival_id] = value
        spans[(start, end)] = (cells, own)

    per_sample = [{k: v for k, v in spans.items() if v[0] and v[1] > 0.0}
                  for spans in per_sample]

    width = max((len(spans) for spans in per_sample), default=0)
    if width == 0:
        return None

    span_pairs = torch.zeros(n_batch, width, 2, dtype=pairs.dtype, device=device)
    values = torch.zeros(n_batch, width, n_query, dtype=torch.float32, device=device)
    reference = torch.zeros(n_batch, width, dtype=torch.float32, device=device)
    span_mask = torch.zeros(n_batch, width, dtype=torch.bool, device=device)
    for index, spans in enumerate(per_sample):
        for slot, ((start, end), (cells, own)) in enumerate(spans.items()):
            span_pairs[index, slot, 0] = start
            span_pairs[index, slot, 1] = end
            span_mask[index, slot] = True
            reference[index, slot] = own
            for query_id, value in cells.items():
                values[index, slot, query_id] = value
            _COUNTS["cells"] += len(cells)
    return span_pairs, values, reference, span_mask


def densify_guide(
    sparse: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    candidate_indices: torch.LongTensor,
    query_axis: int,
    candidate_axis: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter sparse guide values onto the candidate score tensor's own axis order.

    ``candidate_indices`` is ``[B,C,2]`` for the shared pool and ``[B,Q,C,2]`` per query;
    both results match whatever ``(query_axis, candidate_axis)`` the caller is using, so
    they can be handed straight to
    :func:`~gliner2.models.boundary.losses.apply_guide_veto` as ``guide_logits`` and
    ``reference``.
    """
    spans, values, reference, span_mask = sparse
    values = values.to(torch.float32)
    if candidate_indices.dim() == 3:
        match = (candidate_indices.unsqueeze(2) == spans.unsqueeze(1)).all(-1)
        match = (match & span_mask.unsqueeze(1)).to(values.dtype)    # [B,C,S]
        dense = torch.einsum("bcs,bsq->bcq", match, values)
        ref = torch.einsum("bcs,bs->bc", match, reference).unsqueeze(-1).expand_as(dense)
    else:
        match = (candidate_indices.unsqueeze(3) == spans[:, None, None, :, :]).all(-1)
        match = (match & span_mask[:, None, None, :]).to(values.dtype)  # [B,Q,C,S]
        dense = torch.einsum("bqcs,bsq->bqc", match, values)
        ref = torch.einsum("bqcs,bs->bqc", match, reference)
    first, second = (candidate_axis, query_axis) if candidate_indices.dim() == 3 \
        else (query_axis, candidate_axis)
    return (torch.movedim(dense, (1, 2), (first, second)),
            torch.movedim(ref, (1, 2), (first, second)))


def note_vetoes(before: torch.BoolTensor, after: torch.BoolTensor) -> None:
    """Record how many mined negatives the guide removed, and say so once."""
    removed = int(before.sum() - after.sum())
    if not removed:
        return
    first = _COUNTS["vetoed"] == 0
    _COUNTS["vetoed"] += removed
    if first:
        logger.info("GIST veto active: %d mined negatives removed on its first "
                    "firing (%s)", removed, dict(_COUNTS))


__all__ = ["build_sparse_guide", "densify_guide", "guide_counts",
           "note_vetoes", "reset_guide_counts"]
