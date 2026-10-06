"""Records and events for the boundary architecture: production instance head.

This module implements **Instance Formation and Record Disambiguation**. It
replaces count-first structure decoding with *instance identity* plus
*field-to-instance assignment* built on the same sparse boundary candidates
(no dense grids, no count head):

* **Anchor-driven (natural):** every detected anchor candidate seeds one record
  instance; each non-anchor field candidate is scored against every instance
  with an explicit ``ABSENT`` alternative.
* **Latent anchor:** no declared anchor - a learned selector scores each
  candidate as a potential instance seed, supervised only by record grouping.
* **Anchorless:** document-conditioned learned instance queries cross-attend the
  candidate states and predict object/``NO_OBJECT`` plus per-field pointers.

The low-level primitives ``FieldAssignmentScorer`` and ``RecordSetDecoder`` are
retained (and unit-tested) as building blocks; ``RecordHead`` is the integrated,
schema-aware module used by :class:`BoundaryExtractorModel` and the engine.

Record *count* is never predicted; it is derived from selected instances by the
global decoder implemented below.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gliner2.models.candidates import CandidateSet
from gliner2.models.boundary.constants import MASK_LOGIT
from gliner2.models.boundary.validation import (
    filter_match_indices,
    safe_query_ids,
)
from gliner2.processing.records import FieldCardinality, RecordFieldSpec, RecordSpec
from gliner2.processing.targets import RecordTarget, TargetCapacityError

logger = logging.getLogger(__name__)
from gliner2.training.matching import (
    build_dense_record_matching_cost,
    linear_sum_assignment,
)


# =============================================================================
# Backward-compatible low-level primitives
# =============================================================================

@dataclass(frozen=True)
class InstanceCandidate:
    """One record/event instance seeded by an anchor (trigger) span."""

    anchor_query_id: int
    anchor_start: int
    anchor_end: int
    score: float


@dataclass
class InstanceCandidateBatch:
    """Padded anchor instances for a batch: states ``[B, N, H]`` + mask ``[B, N]``."""

    states: torch.Tensor
    mask: torch.BoolTensor


def create_anchor_instances(
    anchor_candidates: CandidateSet,
    anchor_query_id: int,
) -> Tuple[InstanceCandidate, ...]:
    """One :class:`InstanceCandidate` per surviving anchor candidate."""
    instances: List[InstanceCandidate] = []
    for start, end, logit in anchor_candidates.for_query(anchor_query_id):
        instances.append(InstanceCandidate(anchor_query_id, start, end, float(logit)))
    return tuple(instances)


class FieldAssignmentScorer(nn.Module):
    """Score assigning each field candidate to each anchor instance.

    Returns ``[B, N, F, C]`` edge logits (anchor N x field F x candidate C).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.anchor_proj = nn.Linear(hidden_size, hidden_size)
        self.field_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        instance_candidates: InstanceCandidateBatch,
        field_candidate_states: torch.Tensor,   # [B, F, C, H]
        field_query_states: torch.Tensor,       # [B, F, H]
    ) -> torch.Tensor:
        anchor = self.anchor_proj(instance_candidates.states)          # [B, N, H]
        field_q = self.field_proj(field_query_states)                  # [B, F, H]
        query = anchor[:, :, None, :] + field_q[:, None, :, :]         # [B, N, F, H]
        logits = torch.einsum("bnfh,bfch->bnfc", query, field_candidate_states)
        return logits


@dataclass
class RecordSetOutput:
    """Set-decoder output.

    Shapes (``B`` samples, ``I`` instance queries, ``F`` fields, ``C`` candidates):
        object_logits:        [B, I]       object / no-object
        field_pointer_logits: [B, I, F, C] pointer over field candidates
    """

    object_logits: torch.Tensor
    field_pointer_logits: torch.Tensor


class RecordSetDecoder(nn.Module):
    """Fixed instance queries -> object + per-field candidate pointers.

    Object logits are conditioned on the document/schema by cross-attending the
    learned instance queries over the field-candidate states, so the predicted
    record count is input-dependent (well beyond the legacy 19-instance cap).
    """

    def __init__(self, hidden_size: int, instance_queries: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.instance_queries = instance_queries
        self.instance_embed = nn.Parameter(torch.randn(instance_queries, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.object_head = nn.Linear(hidden_size, 1)
        self.inst_proj = nn.Linear(hidden_size, hidden_size)
        self.field_proj = nn.Linear(hidden_size, hidden_size)

    def _condition(
        self,
        inst: torch.Tensor,                      # [B, I, H]
        field_candidate_states: torch.Tensor,    # [B, F, C, H]
        field_mask: Optional[torch.BoolTensor],   # [B, F, C]
    ) -> torch.Tensor:
        b, f, c, h = field_candidate_states.shape
        ctx = field_candidate_states.reshape(b, f * c, h)              # [B, FC, H]
        q = self.q_proj(inst)                                          # [B, I, H]
        k = self.k_proj(ctx)                                          # [B, FC, H]
        v = self.v_proj(ctx)
        attn = torch.einsum("bih,bjh->bij", q, k) / math.sqrt(h)       # [B, I, FC]
        if field_mask is not None:
            m = field_mask.reshape(b, 1, f * c)
            attn = attn.masked_fill(~m, float("-inf"))
            # A fully-masked instance row would produce NaNs; guard it.
            all_masked = ~m.any(dim=-1, keepdim=True)
            attn = attn.masked_fill(all_masked.expand_as(attn), 0.0)
        weights = torch.softmax(attn, dim=-1)
        pooled = torch.einsum("bij,bjh->bih", weights, v)             # [B, I, H]
        return inst + pooled

    def forward(
        self,
        field_query_states: torch.Tensor,        # [B, F, H]
        field_candidate_states: torch.Tensor,    # [B, F, C, H]
        field_mask: Optional[torch.BoolTensor] = None,  # [B, F, C]
    ) -> RecordSetOutput:
        b = field_candidate_states.shape[0]
        inst = self.instance_embed.unsqueeze(0).expand(b, -1, -1)     # [B, I, H]
        inst = self._condition(inst, field_candidate_states, field_mask)
        inst_h = self.inst_proj(inst)                                  # [B, I, H]

        object_logits = self.object_head(inst).squeeze(-1)            # [B, I]

        field_q = self.field_proj(field_query_states)                 # [B, F, H]
        query = inst_h[:, :, None, :] + field_q[:, None, :, :]        # [B, I, F, H]
        logits = torch.einsum("bifh,bfch->bifc", query, field_candidate_states)
        if field_mask is not None:
            logits = logits.masked_fill(~field_mask[:, None, :, :], float("-inf"))
        return RecordSetOutput(object_logits=object_logits, field_pointer_logits=logits)


# =============================================================================
# Integrated, schema-aware record head
# =============================================================================

@dataclass
class RecordGroupOutput:
    """Per-(sample, record group) decoder output consumed by loss + decode.

    ``assign_logits[f]`` has shape ``[Ni, 1 + Cf]``; column 0 is the explicit
    ``ABSENT`` alternative and columns ``1..Cf`` align with ``field_spans[f]``.
    """

    spec: RecordSpec
    object_logits: torch.Tensor                 # [Ni]
    assign_logits: List[torch.Tensor]           # per field: [Ni, 1 + Cf]
    field_query_ids: List[int]
    field_specs: List[RecordFieldSpec]
    field_spans: List[torch.LongTensor]         # per field: [Cf, 2] half-open
    field_cand_mask: List[torch.BoolTensor]     # per field: [Cf]
    field_cand_logits: List[torch.Tensor]       # per field: [Cf] pair logits
    # For natural/latent modes, the (field_index, candidate_index) seed of each
    # instance; None entries for anchorless learned queries.
    instance_seed: List[Optional[Tuple[int, int]]]
    instance_spans: List[Optional[Tuple[int, int]]]
    # [Ni, Ni] trigger x trigger coreference logits (record_coref_link, natural mode); None when off.
    coref_logits: Optional[torch.Tensor] = None

    @property
    def num_instances(self) -> int:
        return int(self.object_logits.shape[0])


@dataclass
class DenseRecordGroupOutput:
    """Shared-pool training representation with no ragged candidate axis."""

    spec: RecordSpec
    object_logits: torch.Tensor              # [I]
    assign_logits: torch.Tensor              # [I,F,1+C_doc]
    instance_mask: torch.BoolTensor           # [I]
    field_membership: torch.BoolTensor        # [F,C_doc]
    pool_spans: torch.LongTensor              # [C_doc,2]
    field_specs: Tuple[RecordFieldSpec, ...]
    field_query_ids: torch.LongTensor         # [F]
    instance_pool_index: torch.LongTensor     # [I], -1 for learned queries

    @property
    def num_instances(self) -> int:
        return self.object_logits.shape[0]


@dataclass
class DenseRecordBatchOutput:
    """Fully batched shared-pool record representation."""

    object_logits: torch.Tensor          # [B,R,I]
    assign_logits: torch.Tensor          # [B,R,I,F,1+C]
    instance_mask: torch.BoolTensor      # [B,R,I]
    field_membership: torch.BoolTensor   # [B,R,F,C]
    pool_spans: torch.LongTensor         # [B,C,2]
    field_mask: torch.BoolTensor         # [B,R,F]
    scalar_fields: torch.BoolTensor      # [B,R,F]
    modes: torch.LongTensor              # [B,R]
    anchor_fields: torch.LongTensor      # [B,R]
    group_mask: torch.BoolTensor         # [B,R]


class TriggerLink(nn.Module):
    """Trigger x trigger COREFERENCE link: are instances i and j mentions of ONE event? (COREFERENT_LINK_SPEC.md)

    s_ij = <W t_i, W t_j> / sqrt(d) + w * log1p(token distance) + b, symmetric. Reads instance states
    only -- the argument-score rows separate coreferent pairs at AUC 0.468 (below chance), because the
    role term is identical for every trigger; the inst_proj states already sit at 0.709 untrained, so W
    is initialised FROM inst_proj (``init_from``). Nothing else reads it: with it off or on, every other
    loss and score is unchanged. Text features (same sentence, lemma) are not available to the record
    head and are not used.
    """

    def __init__(self, hidden_size: int, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(hidden_size, dim)
        self.dist = nn.Linear(1, 1)
        nn.init.zeros_(self.dist.weight)
        nn.init.zeros_(self.dist.bias)

    def init_from(self, proj: nn.Linear) -> None:
        with torch.no_grad():
            self.w.weight.copy_(proj.weight)
            self.w.bias.copy_(proj.bias)

    def forward(self, inst_states: torch.Tensor, inst_spans: torch.Tensor) -> torch.Tensor:
        z = self.w(inst_states)                                                  # [Ni, D]
        mid = (inst_spans[:, 0] + inst_spans[:, 1]).float() / 2
        dist = torch.log1p((mid[:, None] - mid[None, :]).abs()).unsqueeze(-1)  # [Ni, Ni, 1]
        return z @ z.t() / math.sqrt(self.dim) + self.dist(dist).squeeze(-1)


class LinkJunction(nn.Module):
    """The trigger x argument JUNCTION: a learned link per (instance, candidate) pair, read per role.

    The additive assignment score is P (trigger x candidate) + R (role x candidate), and R is the
    same for every trigger. Measured: R does all the work (row AUC 0.948) while P barely separates
    the gold trigger from a false one (junction AUC 0.56-0.61) -- role fit with no join. This adds
    a gated bilinear link, role-gated so role and pair identity INTERACT, plus geometry (relative
    order, log distance). `v` and `geom_q` are zero-initialised: the term is exactly 0
    until it learns, so a warm start is bit-identical at step 0 (JUNCTION_LAYER_SPEC.md).
    """

    def __init__(self, hidden_size: int, dim: int, geom_dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.u = nn.Linear(hidden_size, dim)
        self.v = nn.Linear(hidden_size, dim)
        self.gate = nn.Linear(hidden_size, dim)
        # CONTINUOUS geometry (order, log distance), as SparseRelationScorer does -- no lookup table,
        # no bucket cap. The boundary architecture forbids embedding TABLES here (test_invariants.py).
        self.geom = nn.Linear(2, geom_dim)
        self.geom_q = nn.Linear(hidden_size, geom_dim)
        for layer in (self.v, self.geom_q):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _geometry(self, inst_spans: torch.Tensor, cand_spans: torch.Tensor) -> torch.Tensor:
        """[Ni, Cf, 2]: (candidate after the trigger, log1p distance between span centres)."""
        inst_mid = (inst_spans[:, 0] + inst_spans[:, 1]).float() / 2
        cand_mid = (cand_spans[:, 0] + cand_spans[:, 1]).float() / 2
        delta = cand_mid[None, :] - inst_mid[:, None]
        return torch.stack([(delta > 0).float(), torch.log1p(delta.abs())], dim=-1)

    def forward(self, inst_states, field_query_states, field_cand_states, inst_spans, field_spans) -> List[torch.Tensor]:
        u = self.u(inst_states)                                    # [Ni, D]
        out = []
        for f, cand in enumerate(field_cand_states):
            if cand.shape[0] == 0:
                out.append(None)
                continue
            gate = torch.sigmoid(self.gate(field_query_states[f]))   # [D]
            bilinear = (u * gate) @ self.v(cand).t() / math.sqrt(self.dim)          # [Ni, Cf]
            feats = self._geometry(inst_spans, field_spans[f]).to(cand.dtype)
            geo = self.geom(feats) @ self.geom_q(field_query_states[f])
            out.append(bilinear + geo)
        return out


class RecordHead(nn.Module):
    """Unified natural / latent / anchorless instance formation head.

    All three modes reduce to (instance states, object logits, null-aware field
    assignment). The head is invoked per sample with that sample's compiled
    :class:`RecordSpec` objects and the boundary candidate batch.
    """

    def __init__(self, hidden_size: int, record_dim: int, instance_queries: int, link: bool = False,
                 coref: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.record_dim = record_dim
        self.instance_queries = instance_queries
        # The trigger x argument junction (record_link_mode: junction). Built only when on, so
        # every existing checkpoint still loads strictly; `enable_link` adds it to a loaded model.
        self.link = LinkJunction(hidden_size, record_dim) if link else None

        self.inst_proj = nn.Linear(hidden_size, record_dim)
        self.field_proj = nn.Linear(hidden_size, record_dim)
        self.cand_proj = nn.Linear(hidden_size, record_dim)
        self.null_embed = nn.Parameter(torch.randn(record_dim) * 0.02)
        # The trigger x trigger coreference link (record_coref_link). Built only when on, like the junction.
        self.coref = None
        if coref:
            self.coref = TriggerLink(hidden_size, record_dim)
            self.coref.init_from(self.inst_proj)

        self.object_head = nn.Linear(hidden_size, 1)
        self.latent_seed_head = nn.Linear(hidden_size, 1)

        self.instance_embed = nn.Parameter(torch.randn(instance_queries, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, record_dim)
        self.k_proj = nn.Linear(hidden_size, record_dim)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

    def enable_link(self) -> None:
        """Add the junction to an already-built head (a warm start from a checkpoint without it).

        Zero-initialised, so the loaded model's scores are unchanged until it trains. Call before
        the optimizer is built, or the new parameters are never updated.
        """
        if self.link is None:
            ref = next(self.parameters())
            self.link = LinkJunction(self.hidden_size, self.record_dim).to(device=ref.device, dtype=ref.dtype)

    def enable_coref_link(self) -> None:
        """Add the coreference link to an already-built head (a warm start), initialised from inst_proj.
        Call before the optimizer is built, or its parameters are never updated."""
        if self.coref is None:
            ref = next(self.parameters())
            self.coref = TriggerLink(self.hidden_size, self.record_dim).to(device=ref.device, dtype=ref.dtype)
            self.coref.init_from(self.inst_proj)

    # ------------------------------------------------------------------ utils
    def _assign_logits(
        self,
        inst_states: torch.Tensor,        # [Ni, H]
        field_query_states: torch.Tensor,  # [F, H]
        field_cand_states: List[torch.Tensor],  # per field [Cf, H]
    ) -> List[torch.Tensor]:
        inst_q = self.inst_proj(inst_states)                       # [Ni, D]
        field_q = self.field_proj(field_query_states)              # [F, D]
        out: List[torch.Tensor] = []
        for f, cand in enumerate(field_cand_states):
            query = inst_q + field_q[f].unsqueeze(0)               # [Ni, D]
            null_col = query @ self.null_embed                     # [Ni]
            if cand.shape[0] == 0:
                out.append(null_col.unsqueeze(-1))                 # [Ni, 1]
                continue
            cand_p = self.cand_proj(cand)                          # [Cf, D]
            cand_scores = query @ cand_p.t()                       # [Ni, Cf]
            out.append(torch.cat([null_col.unsqueeze(-1), cand_scores], dim=-1))
        return out

    def _anchorless_states(
        self, field_cand_states: List[torch.Tensor]
    ) -> torch.Tensor:
        inst = self.instance_embed                                 # [I, H]
        ctx = [c for c in field_cand_states if c.shape[0] > 0]
        if not ctx:
            return inst
        ctx = torch.cat(ctx, dim=0)                                # [M, H]
        q = self.q_proj(inst)                                      # [I, D]
        k = self.k_proj(ctx)                                       # [M, D]
        v = self.v_proj(ctx)                                       # [M, H]
        attn = (q @ k.t()) / math.sqrt(self.record_dim)            # [I, M]
        weights = torch.softmax(attn, dim=-1)
        pooled = weights @ v                                       # [I, H]
        return inst + pooled

    # ---------------------------------------------------------------- forward
    def forward_groups_dense(
        self,
        query_states: torch.Tensor,
        candidates,
        routing: tuple[torch.Tensor, ...],
    ) -> DenseRecordBatchOutput:
        """Score every sample/record-group without Python tensor loops."""
        (
            field_query_ids,
            field_mask,
            scalar_fields,
            modes,
            anchor_fields,
            group_mask,
        ) = routing
        device = query_states.device
        field_query_ids = field_query_ids.to(device)
        field_mask = field_mask.to(device)
        scalar_fields = scalar_fields.to(device)
        modes = modes.to(device)
        anchor_fields = anchor_fields.to(device)
        group_mask = group_mask.to(device)
        b, groups, fields = field_query_ids.shape
        candidate_count = candidates.indices.shape[2]
        instance_count = max(candidate_count, self.instance_queries)
        hidden = query_states.shape[-1]
        if query_states.shape[1] == 0 or candidates.valid_mask.shape[1] == 0:
            raise ValueError("record routing requires at least one boundary query")
        batch_index = torch.arange(b, device=device)[:, None, None]
        safe_field_query_ids, valid_field_query_ids = safe_query_ids(
            field_query_ids,
            query_states.shape[1],
            candidates.valid_mask.shape[1],
            candidates.query_mask.shape[1],
            candidates.pair_logits.shape[1],
        )
        effective_field_mask = field_mask & valid_field_query_ids
        field_queries = query_states[batch_index, safe_field_query_ids]
        pool_states = candidates.candidate_states[:, 0]
        pool_spans = candidates.indices[:, 0]
        pool_mask = candidates.valid_mask[:, 0]
        membership = (
            candidates.valid_mask[batch_index, safe_field_query_ids]
            & candidates.query_mask[batch_index, safe_field_query_ids].unsqueeze(-1)
            & effective_field_mask.unsqueeze(-1)
            & pool_mask[:, None, None, :]
        )

        padded_pool_states = F.pad(
            pool_states, (0, 0, 0, instance_count - candidate_count)
        )
        pool_instances = padded_pool_states[:, None].expand(
            b, groups, instance_count, hidden
        )
        learned = F.pad(
            self.instance_embed,
            (0, 0, 0, instance_count - self.instance_queries),
        )
        learned = learned[None, None].expand(b, groups, -1, -1)
        q = self.q_proj(learned)
        k = self.k_proj(pool_states)[:, None].expand(b, groups, -1, -1)
        v = self.v_proj(pool_states)[:, None].expand(b, groups, -1, -1)
        attention = torch.einsum("brid,brcd->bric", q, k)
        attention = attention / math.sqrt(self.record_dim)
        attention = attention.masked_fill(
            ~pool_mask[:, None, None, :], MASK_LOGIT
        )
        anchorless_instances = learned + torch.einsum(
            "bric,brch->brih", torch.softmax(attention, -1), v
        )
        is_anchorless = modes == 2
        instance_states = torch.where(
            is_anchorless[..., None, None],
            anchorless_instances,
            pool_instances,
        )

        anchor_index = anchor_fields.clamp(min=0, max=membership.shape[2] - 1)
        anchor_membership = membership.gather(
            2,
            anchor_index[..., None, None].expand(
                b, groups, 1, candidate_count
            ),
        ).squeeze(2)
        pool_instance_mask = F.pad(
            pool_mask[:, None, :].expand(b, groups, -1),
            (0, instance_count - candidate_count),
        )
        natural_mask = F.pad(
            anchor_membership, (0, instance_count - candidate_count)
        )
        latent_mask = F.pad(
            membership.any(2), (0, instance_count - candidate_count)
        )
        learned_mask = (
            torch.arange(instance_count, device=device)
            < self.instance_queries
        )[None, None].expand(b, groups, -1)
        instance_mask = torch.where(
            (modes == 0)[..., None],
            natural_mask,
            torch.where((modes == 1)[..., None], latent_mask, learned_mask),
        )
        instance_mask &= group_mask[..., None]

        safe_anchor = anchor_index.clamp(max=field_query_ids.shape[2] - 1)
        anchor_qid = safe_field_query_ids.gather(
            2, safe_anchor.unsqueeze(-1)
        ).squeeze(-1).clamp(min=0)
        safe_qid = anchor_qid.clamp(0, candidates.pair_logits.shape[1] - 1)
        natural_object = candidates.pair_logits[
            torch.arange(b, device=device)[:, None],
            safe_qid,
        ]
        natural_object = F.pad(
            natural_object, (0, instance_count - candidate_count)
        )
        latent_object = F.pad(
            self.latent_seed_head(pool_states).squeeze(-1),
            (0, instance_count - candidate_count),
        )[:, None].expand(b, groups, -1)
        anchorless_object = self.object_head(anchorless_instances).squeeze(-1)
        object_logits = torch.where(
            (modes == 0)[..., None],
            natural_object,
            torch.where(
                (modes == 1)[..., None],
                latent_object,
                anchorless_object,
            ),
        )
        object_logits = object_logits.masked_fill(~instance_mask, MASK_LOGIT)

        instance_query = self.inst_proj(instance_states)
        field_query = self.field_proj(field_queries)
        assignment_query = (
            instance_query.unsqueeze(3) + field_query.unsqueeze(2)
        )
        null_logits = torch.einsum(
            "brifd,d->brif", assignment_query, self.null_embed
        )
        candidate_logits = torch.einsum(
            "brifd,bcd->brifc",
            assignment_query,
            self.cand_proj(pool_states),
        )
        candidate_logits = candidate_logits.masked_fill(
            ~membership.unsqueeze(2), MASK_LOGIT
        )
        assign_logits = torch.cat(
            (null_logits.unsqueeze(-1), candidate_logits), -1
        )
        return DenseRecordBatchOutput(
            object_logits,
            assign_logits,
            instance_mask,
            membership,
            pool_spans,
            field_mask,
            scalar_fields,
            modes,
            anchor_fields,
            group_mask,
        )

    def forward_group_dense(
        self,
        spec: RecordSpec,
        query_states: torch.Tensor,
        candidates,
        sample_index: int,
    ) -> DenseRecordGroupOutput:
        """Tensorized shared-document-pool forward used during training."""
        field_specs = tuple(spec.fields)
        field_query_ids = torch.as_tensor(
            [field.query_id for field in field_specs],
            dtype=torch.long,
            device=query_states.device,
        )
        if query_states.shape[0] == 0 or candidates.valid_mask.shape[1] == 0:
            raise ValueError("record routing requires at least one boundary query")
        safe_field_query_ids, valid_field_query_ids = safe_query_ids(
            field_query_ids,
            query_states.shape[0],
            candidates.valid_mask.shape[1],
            candidates.query_mask.shape[1],
            candidates.pair_logits.shape[1],
        )
        pool_states = candidates.candidate_states[sample_index, 0]  # [C,H]
        pool_spans = candidates.indices[sample_index, 0].to(torch.long)
        pool_mask = candidates.valid_mask[sample_index, 0]
        membership = (
            candidates.valid_mask[sample_index, safe_field_query_ids]
            & candidates.query_mask[sample_index, safe_field_query_ids].unsqueeze(-1)
            & valid_field_query_ids.unsqueeze(-1)
            & pool_mask.unsqueeze(0)
        )
        field_queries = query_states[safe_field_query_ids]

        if spec.mode == "natural":
            anchor_matches = field_query_ids == spec.anchor_query_id
            anchor_index = anchor_matches.to(torch.long).argmax()
            instance_states = pool_states
            anchor_valid = anchor_matches.any() & valid_field_query_ids[anchor_index]
            instance_mask = membership[anchor_index] & anchor_valid
            object_logits = candidates.pair_logits[
                sample_index, safe_field_query_ids[anchor_index]
            ]
            instance_pool_index = torch.arange(
                pool_states.shape[0], device=pool_states.device
            )
        elif spec.mode == "latent":
            instance_states = pool_states
            instance_mask = membership.any(0)
            object_logits = self.latent_seed_head(pool_states).squeeze(-1)
            instance_pool_index = torch.arange(
                pool_states.shape[0], device=pool_states.device
            )
        else:
            instance_states = self.instance_embed
            q = self.q_proj(instance_states)
            k = self.k_proj(pool_states)
            v = self.v_proj(pool_states)
            attention = torch.einsum("id,cd->ic", q, k) / math.sqrt(self.record_dim)
            attention = attention.masked_fill(~pool_mask.unsqueeze(0), MASK_LOGIT)
            pooled = torch.einsum("ic,ch->ih", torch.softmax(attention, -1), v)
            instance_states = instance_states + pooled
            instance_mask = torch.ones(
                self.instance_queries, dtype=torch.bool, device=pool_states.device
            )
            object_logits = self.object_head(instance_states).squeeze(-1)
            instance_pool_index = torch.full(
                (self.instance_queries,), -1, dtype=torch.long,
                device=pool_states.device,
            )

        instance_query = self.inst_proj(instance_states)
        field_query = self.field_proj(field_queries)
        query = instance_query[:, None, :] + field_query[None, :, :]
        null_logits = torch.einsum("ifd,d->if", query, self.null_embed)
        candidate_logits = torch.einsum(
            "ifd,cd->ifc", query, self.cand_proj(pool_states)
        )
        candidate_logits = candidate_logits.masked_fill(
            ~membership.unsqueeze(0), MASK_LOGIT
        )
        assign_logits = torch.cat(
            (null_logits.unsqueeze(-1), candidate_logits), dim=-1
        )
        return DenseRecordGroupOutput(
            spec=spec,
            object_logits=object_logits,
            assign_logits=assign_logits,
            instance_mask=instance_mask,
            field_membership=membership,
            pool_spans=pool_spans,
            field_specs=field_specs,
            field_query_ids=field_query_ids,
            instance_pool_index=instance_pool_index,
        )

    def forward_group(
        self,
        spec: RecordSpec,
        query_states: torch.Tensor,       # [Q, H] this sample's query states
        candidates,                        # CandidateTensorBatch (sample slice via index)
        sample_index: int,
    ) -> RecordGroupOutput:
        """Decode one record group for one sample."""
        device = query_states.device
        field_specs = list(spec.fields)
        field_query_ids = [f.query_id for f in field_specs]
        query_count = min(
            query_states.shape[0],
            candidates.valid_mask.shape[1],
            candidates.pair_logits.shape[1],
            candidates.candidate_states.shape[1],
        )
        if query_count <= 0:
            raise ValueError("record routing requires at least one boundary query")
        valid_query_ids = [
            0 <= query_id < query_count for query_id in field_query_ids
        ]
        safe_field_query_ids = [
            min(max(query_id, 0), query_count - 1)
            for query_id in field_query_ids
        ]

        # Gather per-field candidate tensors (state / span / mask / logit).
        field_cand_states: List[torch.Tensor] = []
        field_spans: List[torch.LongTensor] = []
        field_cand_mask: List[torch.BoolTensor] = []
        field_cand_logits: List[torch.Tensor] = []
        cand_states_all = candidates.candidate_states
        for qid, query_valid in zip(safe_field_query_ids, valid_query_ids):
            mask = candidates.valid_mask[sample_index, qid] & query_valid  # [C]
            keep = torch.nonzero(mask, as_tuple=False).flatten()
            spans = candidates.indices[sample_index, qid][keep]    # [Cf, 2]
            logits = candidates.pair_logits[sample_index, qid][keep]  # [Cf]
            states = cand_states_all[sample_index, qid][keep]      # [Cf, H]
            field_cand_states.append(states)
            field_spans.append(spans.to(torch.long))
            field_cand_mask.append(torch.ones(keep.shape[0], dtype=torch.bool, device=device))
            field_cand_logits.append(logits)

        fq = query_states[safe_field_query_ids]                    # [F, H]

        instance_seed: List[Optional[Tuple[int, int]]] = []
        instance_spans: List[Optional[Tuple[int, int]]] = []

        if spec.mode == "natural":
            anchor_field_idx = field_query_ids.index(spec.anchor_query_id)
            anchor_states = field_cand_states[anchor_field_idx]    # [Ca, H]
            anchor_spans = field_spans[anchor_field_idx]
            anchor_logits = field_cand_logits[anchor_field_idx]
            ni = anchor_states.shape[0]
            inst_states = anchor_states
            object_logits = anchor_logits
            for c, (start, end) in enumerate(anchor_spans.tolist()):
                instance_seed.append((anchor_field_idx, c))
                instance_spans.append((start, end))
        elif spec.mode == "latent":
            seed_states: List[torch.Tensor] = []
            seed_scores: List[torch.Tensor] = []
            for f_idx, states in enumerate(field_cand_states):
                if states.shape[0] == 0:
                    continue
                scores = self.latent_seed_head(states).squeeze(-1)  # [Cf]
                for c in range(states.shape[0]):
                    seed_states.append(states[c])
                    seed_scores.append(scores[c])
                    instance_seed.append((f_idx, c))
                    sp = field_spans[f_idx][c]
                    instance_spans.append((int(sp[0]), int(sp[1])))
            if seed_states:
                inst_states = torch.stack(seed_states, dim=0)
                object_logits = torch.stack(seed_scores, dim=0)
            else:
                inst_states = query_states.new_zeros((0, self.hidden_size))
                object_logits = query_states.new_zeros((0,))
        else:  # anchorless
            inst_states = self._anchorless_states(field_cand_states)  # [I, H]
            object_logits = self.object_head(inst_states).squeeze(-1)  # [I]
            for _ in range(inst_states.shape[0]):
                instance_seed.append(None)
                instance_spans.append(None)

        assign_logits = self._assign_logits(inst_states, fq, field_cand_states)
        if self.link is not None and spec.mode == "natural" and inst_states.shape[0] > 0:
            link = self.link(inst_states, fq, field_cand_states, anchor_spans, field_spans)
            assign_logits = [a if l is None else torch.cat([a[:, :1], a[:, 1:] + l], dim=-1)
                             for a, l in zip(assign_logits, link)]
        coref_logits = None
        if self.coref is not None and spec.mode == "natural" and inst_states.shape[0] > 0:
            coref_logits = self.coref(inst_states, anchor_spans)

        return RecordGroupOutput(
            spec=spec,
            object_logits=object_logits,
            assign_logits=assign_logits,
            field_query_ids=field_query_ids,
            field_specs=field_specs,
            field_spans=field_spans,
            field_cand_mask=field_cand_mask,
            field_cand_logits=field_cand_logits,
            instance_seed=instance_seed,
            instance_spans=instance_spans,
            coref_logits=coref_logits,
        )


__all__ = [
    "InstanceCandidate",
    "InstanceCandidateBatch",
    "create_anchor_instances",
    "FieldAssignmentScorer",
    "RecordSetOutput",
    "RecordSetDecoder",
    "RecordGroupOutput",
    "DenseRecordGroupOutput",
    "RecordHead",
    "DecodedRecord",
    "decode_group",
    "derive_count",
    "compute_group_loss",
    "compute_dense_group_loss",
    "build_dense_record_cost",
]


# =============================================================================
# Global record decoding
# =============================================================================

@dataclass
class DecodedRecord:
    """One decoded record: field query id -> selected half-open token spans."""

    fields: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    field_scores: Dict[int, List[float]] = field(default_factory=dict)
    anchor_span: Optional[Tuple[int, int]] = None
    score: float = 0.0


def _dedup_key(rec: DecodedRecord) -> Tuple:
    return tuple(
        (qid, tuple(sorted(spans)))
        for qid, spans in sorted(rec.fields.items())
    )


def decode_group(
    group: RecordGroupOutput,
    *,
    anchor_threshold: float = 0.5,
    field_threshold: float = 0.5,
    object_threshold: float = 0.5,
    temperature: float = 1.0,
    merge_coreferent: str = "off",
    coref_threshold: float = 0.5,
) -> List[DecodedRecord]:
    """Decode one record group into a list of :class:`DecodedRecord`.

    ``merge_coreferent`` (natural mode, COREFERENT_OWNERSHIP_SPEC 3b-i): selected instances that
    score a shared argument candidate at >= ``field_threshold`` BEFORE exclusive allocation are one
    event. The strongest keeps its row, the others contribute only their trigger spans. Read
    before allocation on purpose: exclusive allocation gives each candidate to ONE instance, so
    two mentions of one event never share a DECODED argument.
    """
    ni = group.num_instances
    if ni == 0:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    # Every tensor read below is ONE `.tolist()`: `float(t[i])` on a device tensor syncs per
    # element, which made decode ~half of eval time on a GPU device (measured 2026-10-01).
    obj_prob = torch.sigmoid(group.object_logits.detach() / temperature).tolist()
    select_thr = object_threshold if group.spec.mode == "anchorless" else anchor_threshold
    order = sorted(range(ni), key=lambda i: (-obj_prob[i], i))
    selected_instances = [
        inst for inst in order if obj_prob[inst] >= select_thr
    ]
    coref_members: Dict[int, List[int]] = {}
    merge_coreferent = {True: "args", False: "off"}.get(merge_coreferent, merge_coreferent)
    if merge_coreferent != "off" and group.spec.mode == "natural" and len(selected_instances) > 1:
        if merge_coreferent == "link":
            # A merge the setting promises and the model cannot do must fail LOUDLY: traced 2026-10-06,
            # eval overrides set `link` on a checkpoint without the module, coref_logits stayed None and
            # every threshold "merged" nothing -- a gate that could not fail.
            if group.coref_logits is None:
                raise ValueError("record_merge_coreferent: link, but this model has no coreference link "
                                 "(record_coref_link was off when it was built)")
            selected_instances, coref_members = _merge_by_link(group, selected_instances, coref_threshold)
        elif merge_coreferent == "args":
            selected_instances, coref_members = _merge_coreferent_instances(
                group, selected_instances, field_threshold, temperature)

    # Exclusive fields are a global assignment problem: greedily letting the
    # highest-object instance claim its favorite candidate can force later
    # instances onto unrelated spans. Solve scalar fields jointly and allocate
    # each list candidate to its strongest instance.
    scalar_choices: Dict[Tuple[int, int], Optional[Tuple[int, float]]] = {}
    list_owners: Dict[Tuple[int, int], Tuple[int, float]] = {}
    for f_idx, fspec in enumerate(group.field_specs):
        if not fspec.exclusive or not selected_instances:
            continue
        logits = torch.stack([
            group.assign_logits[f_idx][inst].detach() / temperature
            for inst in selected_instances
        ])
        candidate_count = max(int(logits.shape[-1]) - 1, 0)
        if fspec.cardinality.is_scalar:
            if candidate_count == 0:
                for inst in selected_instances:
                    scalar_choices[(inst, f_idx)] = None
                continue
            probs = torch.softmax(logits, dim=-1)
            candidate_probs = probs[:, 1:]
            eps = torch.finfo(candidate_probs.dtype).eps
            candidate_cost = -torch.log(candidate_probs.clamp_min(eps))
            row_count = len(selected_instances)
            diagonal = -torch.log(probs[:, 0].clamp_min(eps))
            if not fspec.allows_absent:
                # Keep a finite emergency ABSENT column for under-capacity
                # schemas, but never prefer it while a candidate is available.
                diagonal = candidate_cost.max().detach() + 50.0
                diagonal = diagonal.expand(row_count)
            invalid_cost = max(
                float(candidate_cost.max()),
                float(diagonal.max()),
            ) + 1_000.0
            absent_cost = candidate_cost.new_full(
                (row_count, row_count),
                invalid_cost,
            )
            absent_cost[torch.arange(row_count), torch.arange(row_count)] = diagonal
            cost = torch.cat((candidate_cost, absent_cost), dim=-1)
            rows, cols = linear_sum_assignment(cost)
            assignments = {int(row): int(col) for row, col in zip(rows, cols)}
            candidate_probs_list = candidate_probs.tolist()
            for row, inst in enumerate(selected_instances):
                col = assignments.get(row, candidate_count + row)
                if col >= candidate_count:
                    scalar_choices[(inst, f_idx)] = None
                    continue
                probability = candidate_probs_list[row][col]
                if probability < field_threshold and fspec.allows_absent:
                    scalar_choices[(inst, f_idx)] = None
                    continue
                scalar_choices[(inst, f_idx)] = (col, probability)
        else:
            if candidate_count == 0:
                continue
            best_prob, best_row = torch.sigmoid(logits[:, 1:]).max(dim=0)
            for cand_idx, (probability, row) in enumerate(zip(best_prob.tolist(), best_row.tolist())):
                if probability >= field_threshold:
                    list_owners[(f_idx, cand_idx)] = (
                        selected_instances[row],
                        probability,
                    )

    records: List[DecodedRecord] = []
    field_spans = [spans.tolist() for spans in group.field_spans]
    for inst in selected_instances:
        rec = DecodedRecord(score=obj_prob[inst])
        anchor_field_idx = None
        if group.spec.mode == "natural":
            anchor_field_idx = group.field_query_ids.index(group.spec.anchor_query_id)
            seed = group.instance_seed[inst]
            if seed is not None:
                rec.anchor_span = group.instance_spans[inst]

        for f_idx, fspec in enumerate(group.field_specs):
            qid = fspec.query_id
            spans = field_spans[f_idx]
            logits_row = group.assign_logits[f_idx][inst].detach() / temperature
            if anchor_field_idx is not None and f_idx == anchor_field_idx:
                if rec.anchor_span is not None:
                    rec.fields.setdefault(qid, []).append(rec.anchor_span)
                    rec.field_scores.setdefault(qid, []).append(rec.score)
                for member in coref_members.get(inst, []):
                    span = group.instance_spans[member]
                    if span is not None and span not in rec.fields.get(qid, []):
                        rec.fields.setdefault(qid, []).append(span)
                        rec.field_scores.setdefault(qid, []).append(obj_prob[member])
                continue

            if fspec.cardinality.is_scalar:
                if fspec.exclusive:
                    choice = scalar_choices.get((inst, f_idx))
                    if choice is None:
                        continue
                    cand_idx, probability = choice
                    span = (spans[cand_idx][0], spans[cand_idx][1])
                    rec.fields.setdefault(qid, []).append(span)
                    rec.field_scores.setdefault(qid, []).append(probability)
                    continue
                probs = torch.softmax(logits_row, dim=-1)
                probs_list = probs.tolist()
                chosen = None
                for col in torch.argsort(probs, descending=True).tolist():
                    if col == 0:
                        if fspec.allows_absent:
                            chosen = 0
                            break
                        continue
                    chosen = col
                    break
                if chosen is None or chosen == 0:
                    continue
                if probs_list[chosen] < field_threshold and fspec.allows_absent:
                    continue
                cand_idx = chosen - 1
                span = (spans[cand_idx][0], spans[cand_idx][1])
                rec.fields.setdefault(qid, []).append(span)
                rec.field_scores.setdefault(qid, []).append(probs_list[chosen])
            else:
                cand_logits = logits_row[1:]
                if cand_logits.numel() == 0:
                    continue
                probs = torch.sigmoid(cand_logits).tolist()
                selected: List[Tuple[int, int]] = []
                selected_scores: List[float] = []
                for cand_idx in range(cand_logits.shape[0]):
                    if fspec.exclusive:
                        owner = list_owners.get((f_idx, cand_idx))
                        if owner is None or owner[0] != inst:
                            continue
                        probability = owner[1]
                    else:
                        probability = probs[cand_idx]
                        if probability < field_threshold:
                            continue
                    span = (spans[cand_idx][0], spans[cand_idx][1])
                    selected.append(span)
                    selected_scores.append(probability)
                if selected:
                    rec.fields.setdefault(qid, []).extend(selected)
                    rec.field_scores.setdefault(qid, []).extend(selected_scores)
        if rec.fields:
            records.append(rec)

    if group.spec.mode in ("latent", "anchorless"):
        best: Dict[Tuple, DecodedRecord] = {}
        for rec in records:
            key = _dedup_key(rec)
            if key not in best or rec.score > best[key].score:
                best[key] = rec
        records = list(best.values())
    elif group.spec.mode == "natural":
        records.sort(
            key=lambda record: (
                record.anchor_span is None,
                record.anchor_span or (0, 0),
            )
        )
    return records


def _merge_by_link(group: RecordGroupOutput, selected: List[int], threshold: float):
    """(representatives, {representative: members}) -- single linkage on sigmoid(coref_logits) >= threshold."""
    prob = torch.sigmoid(group.coref_logits.detach()).tolist()
    parent = {i: i for i in selected}
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for a_pos, a in enumerate(selected):
        for b in selected[a_pos + 1:]:
            if prob[a][b] >= threshold and root(a) != root(b):
                parent[root(b)] = root(a)
    reps, members = [], {}
    for inst in selected:
        r = root(inst)
        if r == inst:
            reps.append(inst)
        else:
            members.setdefault(r, []).append(inst)
    return reps, members


def _merge_coreferent_instances(group: RecordGroupOutput, selected: List[int], field_threshold: float,
                                temperature: float):
    """(representatives in selection order, {representative: other members}) -- union of instances that
    score at least one shared list-field candidate at >= field_threshold."""
    high = {}
    for inst in selected:
        high[inst] = {(f, c) for f, fs in enumerate(group.field_specs) if not fs.cardinality.is_scalar
                      for c, p in enumerate(torch.sigmoid(group.assign_logits[f][inst, 1:].detach() / temperature).tolist())
                      if p >= field_threshold}
    parent = {i: i for i in selected}
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for a_pos, a in enumerate(selected):
        for b in selected[a_pos + 1:]:
            if high[a] & high[b] and root(a) != root(b):
                parent[root(b)] = root(a)
    reps, members = [], {}
    for inst in selected:                                   # selected is already strongest-first
        r = root(inst)
        if r == inst:
            reps.append(inst)
        else:
            members.setdefault(r, []).append(inst)
    return reps, members


def derive_count(records: List[DecodedRecord]) -> int:
    """Record count is the number of selected instances - never predicted."""
    return len(records)


# =============================================================================
# Record training losses
# =============================================================================

def _span_index(field_spans: torch.LongTensor) -> Dict[Tuple[int, int], int]:
    """Map each (start, end) span to its row. One `.tolist()` is one device sync; reading
    elements with `int()` synced per element and took 37.6% of an eb18 step at 768 spans."""
    return {(row[0], row[1]): i for i, row in enumerate(field_spans.tolist())}


def _resolve_value_cols(value_alternatives, span_to_idx):
    return [
        idx + 1 for span in value_alternatives
        if (idx := span_to_idx.get((int(span[0]), int(span[1])))) is not None
    ]


def _scalar_field_nll(logits_row: torch.Tensor, target_cols) -> torch.Tensor:
    logp = F.log_softmax(logits_row, dim=-1)
    cols = target_cols or [0]
    max_idx = logits_row.shape[-1] - 1
    cols = [min(c, max_idx) for c in cols if 0 <= c <= max_idx] or [0]
    idx = torch.tensor(cols, dtype=torch.long, device=logits_row.device)
    return -torch.logsumexp(logp[idx], dim=-1)


def _list_field_bce(logits_row: torch.Tensor, positive_cols, hard_k: int = 0) -> torch.Tensor:
    """BCE over a list field's candidates.

    ``hard_k`` = 0 (historical): the mean over EVERY candidate (up to the 768 budget), which
    divides a missed gold argument's surprisal by ~768 -- traced on eb18, gold arguments at
    P 0.000-0.038 scored a loss of 0.004-0.07. ``hard_k`` > 0: the mean over the gold candidates
    plus the ``hard_k`` highest-scoring wrong ones, so the loss reports the errors that matter.
    """
    cand_logits = logits_row[1:]
    if cand_logits.numel() == 0:
        return logits_row.new_zeros(())
    target = torch.zeros_like(cand_logits)
    n = cand_logits.shape[0]
    for col in positive_cols:
        idx = col - 1
        if 0 <= idx < n:
            target[idx] = 1.0
    if hard_k > 0:
        wrong = cand_logits.detach().masked_fill(target > 0, float("-inf"))
        k = min(hard_k, int((target == 0).sum()))
        keep = target > 0
        if k > 0:
            keep = keep.index_fill(0, torch.topk(wrong, k).indices, True)
        return F.binary_cross_entropy_with_logits(cand_logits[keep], target[keep], reduction="mean")
    return F.binary_cross_entropy_with_logits(cand_logits, target, reduction="mean")


def _field_target_cols(fspec, record: RecordTarget, span_to_idx):
    ft = record.field_for_query(fspec.query_id)
    if fspec.cardinality.is_scalar:
        return (
            _resolve_value_cols(ft.values[0], span_to_idx)
            if ft is not None and ft.values else [],
            True,
        )
    cols = []
    if ft is not None:
        for value in ft.values:
            cols.extend(_resolve_value_cols(value, span_to_idx))
    return cols, False


def _instance_field_loss(group, inst: int, record: RecordTarget, span_indices, hard_k: int = 0) -> torch.Tensor:
    total = group.object_logits.new_zeros(())
    n_fields = 0
    for f_idx, fspec in enumerate(group.field_specs):
        cols, is_scalar = _field_target_cols(fspec, record, span_indices[f_idx])
        row = group.assign_logits[f_idx][inst]
        total = total + (
            _scalar_field_nll(row, cols)
            if is_scalar else _list_field_bce(row, cols, hard_k)
        )
        n_fields += 1
    return total / max(n_fields, 1)


def _instance_field_logprob(group, inst: int, record: RecordTarget, span_indices) -> torch.Tensor:
    total = group.object_logits.new_zeros(())
    for f_idx, fspec in enumerate(group.field_specs):
        cols, is_scalar = _field_target_cols(fspec, record, span_indices[f_idx])
        row = group.assign_logits[f_idx][inst]
        if is_scalar:
            total = total - _scalar_field_nll(row, cols)
        else:
            cand_logits = row[1:]
            if cand_logits.numel() == 0:
                continue
            n = cand_logits.shape[0]
            target = torch.zeros_like(cand_logits)
            for col in cols:
                idx = col - 1
                if 0 <= idx < n:
                    target[idx] = 1.0
            total = total - F.binary_cross_entropy_with_logits(cand_logits, target, reduction="sum")
    return total


def _dense_gold_indicator(
    group: DenseRecordGroupOutput,
    records: Sequence[RecordTarget],
) -> torch.BoolTensor:
    """Map padded gold span alternatives to the shared pool on device."""
    n_gold = len(records)
    fields = len(group.field_specs)
    max_alternatives = max(
        (
            len(alternatives)
            for record in records
            for field_spec in group.field_specs
            for target in [record.field_for_query(field_spec.query_id)]
            if target is not None
            for alternatives in target.values
        ),
        default=1,
    )
    max_values = max(
        (
            len(target.values)
            for record in records
            for field_spec in group.field_specs
            for target in [record.field_for_query(field_spec.query_id)]
            if target is not None
        ),
        default=1,
    )
    gold_spans = torch.zeros(
        n_gold, fields, max_values, max_alternatives, 2,
        dtype=torch.long, device=group.pool_spans.device,
    )
    gold_mask = torch.zeros(
        n_gold, fields, max_values, max_alternatives,
        dtype=torch.bool, device=group.pool_spans.device,
    )
    # This is schema/annotation packing only; candidate matching below is one
    # broadcasted device comparison and never inspects candidate values in Python.
    for gold_index, record in enumerate(records):
        for field_index, field_spec in enumerate(group.field_specs):
            target = record.field_for_query(field_spec.query_id)
            if target is None:
                continue
            for value_index, alternatives in enumerate(target.values):
                if alternatives:
                    count = len(alternatives)
                    gold_spans[gold_index, field_index, value_index, :count] = (
                        torch.as_tensor(
                            alternatives, dtype=torch.long,
                            device=group.pool_spans.device,
                        )
                    )
                    gold_mask[gold_index, field_index, value_index, :count] = True
    matches = (
        group.pool_spans[None, None, None, None, :, :]
        == gold_spans[..., None, :]
    ).all(-1)
    matches &= gold_mask[..., None]
    return matches.any(2).any(2) & group.field_membership[None, :, :]


def build_dense_record_cost(
    group: DenseRecordGroupOutput,
    gold_indicator: torch.BoolTensor,
) -> torch.Tensor:
    """Vectorized ``[I,N_gold]`` matching cost for one dense group."""
    n_gold = gold_indicator.shape[0]
    if n_gold == 0:
        return group.object_logits.new_zeros((group.num_instances, 0))
    scalar = torch.as_tensor(
        [field.cardinality.is_scalar for field in group.field_specs],
        dtype=torch.bool,
        device=group.object_logits.device,
    )
    return build_dense_record_matching_cost(
        group.object_logits,
        group.assign_logits,
        gold_indicator,
        scalar,
        group.instance_mask,
    )


def compute_dense_group_loss(
    group: DenseRecordGroupOutput,
    records: Sequence[RecordTarget],
    gold_indicator: Optional[torch.BoolTensor] = None,
) -> Dict[str, torch.Tensor]:
    """Loss for the dense shared-pool representation."""
    device = group.object_logits.device
    zero = group.object_logits.new_zeros(())
    count = len(records)
    ni = group.num_instances
    # Record matching already has an explicit CPU boundary for Hungarian.
    # Capacity must use real hypotheses, not the padded tensor width.
    available_instances = int(group.instance_mask.detach().sum().cpu())
    if count == 0:
        target = torch.zeros_like(group.object_logits)
        losses = F.binary_cross_entropy_with_logits(
            group.object_logits, target, reduction="none"
        )
        object_loss = (
            losses * group.instance_mask.to(losses.dtype)
        ).sum() / group.instance_mask.sum().clamp_min(1)
        return {
            "object_loss": object_loss,
            "field_loss": zero,
            "object_count": available_instances,
            "field_count": 0,
        }
    if available_instances < count:
        raise TargetCapacityError(
            f"record group task={group.spec.task_index} has {count} gold instances "
            f"but only {available_instances} valid instance hypotheses "
            f"(padded width={ni}, mode={group.spec.mode}); "
            "increase boundary_head.record_instance_queries or candidate budget."
        )
    gold = (
        _dense_gold_indicator(group, records)
        if gold_indicator is None
        else gold_indicator.to(device=device, dtype=torch.bool)
    )
    n_cands_logits = group.assign_logits.shape[-1] - 1
    n_cands_gold = gold.shape[-1]
    if n_cands_gold > n_cands_logits:
        gold = gold[..., :n_cands_logits]
    elif n_cands_gold < n_cands_logits:
        pad = gold.new_zeros(*gold.shape[:-1], n_cands_logits - n_cands_gold)
        gold = torch.cat([gold, pad], dim=-1)
    scalar = torch.as_tensor(
        [field.cardinality.is_scalar for field in group.field_specs],
        dtype=torch.bool, device=device,
    )
    present = gold.any(-1)
    target = torch.cat(((~present).unsqueeze(-1), gold), -1)
    logp = F.log_softmax(group.assign_logits, -1)
    scalar_nll = -torch.logsumexp(
        logp[:, None].masked_fill(~target[None], MASK_LOGIT), -1
    )
    # Mean over each field's REAL candidates only. A plain `.mean(-1)` averaged over
    # the padded pool width, scaling the list-field loss by n_real / n_padded
    # (fastino-ai/GLiNER2#180); the sparse path means over real candidates.
    membership = group.field_membership.to(device=device, dtype=group.object_logits.dtype)
    if membership.shape[-1] > n_cands_logits:
        membership = membership[..., :n_cands_logits]
    elif membership.shape[-1] < n_cands_logits:
        membership = torch.cat(
            [membership, membership.new_zeros(*membership.shape[:-1],
                                              n_cands_logits - membership.shape[-1])], -1)
    list_bce = F.binary_cross_entropy_with_logits(
        group.assign_logits[:, None, :, 1:].expand(-1, count, -1, -1),
        gold[None].expand(ni, -1, -1, -1).to(group.object_logits.dtype),
        reduction="none",
    )
    list_nll = (list_bce * membership).sum(-1) / membership.sum(-1).clamp_min(1)
    field_nll = torch.where(
        scalar[None, None], scalar_nll, list_nll
    ).mean(-1)

    if group.spec.mode == "natural":
        anchor_field = group.field_query_ids == group.spec.anchor_query_id
        anchor_index = anchor_field.to(torch.long).argmax()
        anchor_gold = gold[:, anchor_index]
        anchor_present = anchor_gold.any(-1)
        matched_cols = torch.arange(count, device=device)[anchor_present]
        matched_rows = anchor_gold[anchor_present].to(torch.long).argmax(-1)
        object_loss = zero
    else:
        with torch.no_grad():
            cost = build_dense_record_cost(group, gold)
        rows, cols = linear_sum_assignment(cost)
        matched_rows = rows.to(device)
        matched_cols = cols.to(device)
        object_target = torch.zeros_like(group.object_logits)
        ni = group.object_logits.shape[0]
        valid_rows = matched_rows[matched_rows < ni]
        if valid_rows.numel() > 0:
            object_target.scatter_(0, valid_rows, 1.0)
        object_terms = F.binary_cross_entropy_with_logits(
            group.object_logits, object_target, reduction="none"
        )
        object_loss = (
            object_terms * group.instance_mask.to(object_terms.dtype)
        ).sum() / group.instance_mask.sum().clamp_min(1)
    valid_matches = (
        (matched_rows >= 0)
        & (matched_rows < field_nll.shape[0])
        & (matched_cols >= 0)
        & (matched_cols < field_nll.shape[1])
    )
    if valid_matches.any():
        safe_rows = matched_rows.clamp(min=0, max=field_nll.shape[0] - 1)
        valid_matches &= group.instance_mask[safe_rows]
    valid_rows = matched_rows[valid_matches]
    valid_cols = matched_cols[valid_matches]
    field_loss = (
        field_nll[valid_rows, valid_cols].mean()
        if valid_rows.numel()
        else zero
    )
    return {
        "object_loss": object_loss,
        "field_loss": field_loss,
        "object_count": available_instances,
        "field_count": matched_cols.numel() * len(group.field_specs),
    }


def compute_dense_batch_loss(
    output: DenseRecordBatchOutput,
    gold_indicator: torch.BoolTensor,
    record_mask: torch.BoolTensor,
) -> Dict[str, torch.Tensor]:
    """Vectorized record loss; only Hungarian assignment remains per group."""
    device = output.object_logits.device
    record_mask = record_mask.to(device)
    gold_indicator = gold_indicator.to(device)
    n_cands_logits = output.assign_logits.shape[-1] - 1
    n_cands_gold = gold_indicator.shape[-1]
    if n_cands_gold > n_cands_logits:
        gold_indicator = gold_indicator[..., :n_cands_logits]
    elif n_cands_gold < n_cands_logits:
        pad = gold_indicator.new_zeros(
            *gold_indicator.shape[:-1], n_cands_logits - n_cands_gold
        )
        gold_indicator = torch.cat([gold_indicator, pad], dim=-1)
    present = gold_indicator.any(-1)
    target = torch.cat(((~present).unsqueeze(-1), gold_indicator), -1)
    logp = F.log_softmax(output.assign_logits, -1)
    scalar_nll = -torch.logsumexp(
        logp.unsqueeze(3).masked_fill(
            ~target.unsqueeze(2), MASK_LOGIT
        ),
        -1,
    )
    candidates = output.assign_logits[..., 1:]
    list_nll = F.binary_cross_entropy_with_logits(
        candidates.unsqueeze(3).expand(
            *candidates.shape[:3],
            gold_indicator.shape[2],
            candidates.shape[3],
            candidates.shape[4],
        ),
        gold_indicator.unsqueeze(2).expand(
            *gold_indicator.shape[:2],
            output.object_logits.shape[2],
            gold_indicator.shape[2],
            gold_indicator.shape[3],
            gold_indicator.shape[4],
        ).to(candidates.dtype),
        reduction="none",
    ).mean(-1)
    field_nll = torch.where(
        output.scalar_fields[:, :, None, None, :],
        scalar_nll,
        list_nll,
    )
    field_nll = (
        field_nll * output.field_mask[:, :, None, None, :].to(field_nll.dtype)
    ).sum(-1) / output.field_mask.sum(-1)[:, :, None, None].clamp_min(1)

    with torch.no_grad():
        cost = build_dense_record_matching_cost(
            output.object_logits,
            output.assign_logits,
            gold_indicator,
            output.scalar_fields,
            output.instance_mask,
        )
        # One explicit host transfer covers capacity checks and the unavoidable
        # Hungarian matching boundary for every latent/anchorless group.
        cost_cpu = cost.detach().cpu()
        metadata = torch.stack(
            (
                output.instance_mask.sum(-1),
                record_mask.sum(-1),
                output.modes,
                output.group_mask.to(torch.long),
            ),
            -1,
        ).cpu()
        natural_gold = gold_indicator.gather(
            3,
            output.anchor_fields.clamp(min=0, max=gold_indicator.shape[3] - 1)[..., None, None, None].expand(
                *gold_indicator.shape[:3], 1, gold_indicator.shape[-1]
            ),
        ).squeeze(3).cpu()
        matched_batch: list[int] = []
        matched_group: list[int] = []
        matched_rows: list[int] = []
        matched_cols: list[int] = []
        bsz, groups = output.group_mask.shape
        for batch_index in range(bsz):
            for group_index in range(groups):
                available, count, mode, valid = metadata[
                    batch_index, group_index
                ].tolist()
                if not valid:
                    continue
                if available < count:
                    raise TargetCapacityError(
                        f"record group batch={batch_index} group={group_index} "
                        f"has {count} gold instances but only {available} valid "
                        "instance hypotheses"
                    )
                if count == 0:
                    continue
                if mode == 0:
                    anchors = natural_gold[
                        batch_index, group_index, :count
                    ]
                    columns = torch.nonzero(
                        anchors.any(-1), as_tuple=False
                    ).flatten()
                    rows = anchors[columns].to(torch.long).argmax(-1)
                else:
                    rows, columns = linear_sum_assignment(
                        cost_cpu[batch_index, group_index, :, :count]
                    )
                matched_batch.extend([batch_index] * len(rows))
                matched_group.extend([group_index] * len(rows))
                matched_rows.extend(rows.tolist())
                matched_cols.extend(columns.tolist())
    index = tuple(
        torch.as_tensor(values, dtype=torch.long, device=device)
        for values in (
            matched_batch,
            matched_group,
            matched_rows,
            matched_cols,
        )
    )
    index = filter_match_indices(
        index,
        field_nll.shape,
        instance_mask=output.instance_mask,
    )
    object_target = torch.zeros_like(output.object_logits)
    non_natural = (
        output.modes[index[0], index[1]] != 0
        if index[0].numel()
        else torch.zeros(0, dtype=torch.bool, device=device)
    )
    row_idx = index[2][non_natural]
    if row_idx.numel():
        object_target[
            index[0][non_natural],
            index[1][non_natural],
            row_idx,
        ] = 1.0
    object_keep = (
        output.instance_mask
        & output.group_mask[..., None]
        & (output.modes != 0)[..., None]
    )
    object_terms = F.binary_cross_entropy_with_logits(
        output.object_logits, object_target, reduction="none"
    )
    object_loss = (
        object_terms * object_keep.to(object_terms.dtype)
    ).sum() / object_keep.sum().clamp_min(1)
    field_loss = (
        field_nll[index].mean()
        if index[0].numel()
        else output.object_logits.new_zeros(())
    )
    return {"object_loss": object_loss, "field_loss": field_loss}


# --- the natural-mode anchor supervision gate -------------------------------------------
#
# WHY THIS IS COUNTED. `compute_group_loss` in natural mode resolves each gold record's
# anchor against the model's OWN candidate spans. A record whose anchor was not proposed
# is skipped ENTIRELY -- and silently, until this counter existed. For events the anchor is
# the trigger, so every missed trigger discards that instance's whole argument supervision,
# which is a self-reinforcing loop: propose few triggers -> train on few events -> keep
# proposing few triggers.
#
# It is the FOURTH candidate for the recall floor. EVENT_ARGUMENT_DIAGNOSIS 4f cleared three
# (no capacity caps, per-role cardinality at 7.5%, the mention-path skip_sample) and
# concluded "undertraining, nothing structural capping it". This is not a capacity cap; it
# is a supervision gate, and 4f's search did not cover it.
#
# NO GPU SYNC HERE, deliberately: `cols` and `seed_to_inst` are plain Python lists/dicts, so
# nothing is copied off the device. That is the mistake `_note_negative_queries` made -- it
# called int() on a tensor every step -- and it is not repeated.
#
# Reported as SHARES, because the absolute count is meaningless without its denominator:
# "12,000 skipped" says nothing until you know whether 13,000 or 1,300,000 were seen.
_ANCHOR_GATE = {"trained": 0, "anchor_not_proposed": 0, "anchor_not_seeded": 0,
                "no_gold_anchor": 0}
# BY TASK TYPE, because the aggregate cannot answer the only question that matters.
# `json_structures` and `events` both compile record groups, and a 100% pass driven by
# structures would say NOTHING about triggers while looking like an answer. Keyed
# "<task_type>/<outcome>".
_ANCHOR_GATE_BY_TASK: Dict[str, int] = {}
_ANCHOR_GATE_CALLS = 0


def _note_anchor_gate(outcome: str, task_type: str = "?") -> None:
    """Tally one gold record's fate at the anchor gate, and report with backoff."""
    global _ANCHOR_GATE_CALLS
    _ANCHOR_GATE[outcome] += 1
    _ANCHOR_GATE_BY_TASK[f"{task_type}/{outcome}"] = (
        _ANCHOR_GATE_BY_TASK.get(f"{task_type}/{outcome}", 0) + 1
    )
    _ANCHOR_GATE_CALLS += 1
    n = _ANCHOR_GATE_CALLS
    if n in (1000, 10000, 50000) or (n > 50000 and n % 250000 == 0):
        total = max(n, 1)
        logger.info(
            "anchor gate over %d gold records: trained %.1f%%, anchor NOT PROPOSED %.1f%%, "
            "proposed but not seeded %.1f%%, no gold anchor %.1f%% -- a record that does "
            "not train contributes NO argument supervision at all",
            n,
            100.0 * _ANCHOR_GATE["trained"] / total,
            100.0 * _ANCHOR_GATE["anchor_not_proposed"] / total,
            100.0 * _ANCHOR_GATE["anchor_not_seeded"] / total,
            100.0 * _ANCHOR_GATE["no_gold_anchor"] / total,
        )


def anchor_gate_stats() -> Dict[str, float]:
    """Snapshot of the gate, as counts plus shares. For probes and tests."""
    total = max(_ANCHOR_GATE_CALLS, 1)
    out = {f"{k}_n": v for k, v in _ANCHOR_GATE.items()}
    out.update({f"{k}_share": v / total for k, v in _ANCHOR_GATE.items()})
    out["seen"] = _ANCHOR_GATE_CALLS
    out["by_task"] = dict(_ANCHOR_GATE_BY_TASK)
    return out


def reset_anchor_gate() -> None:
    """Zero the gate. Probes call this so one run's numbers are its own."""
    global _ANCHOR_GATE_CALLS
    for k in _ANCHOR_GATE:
        _ANCHOR_GATE[k] = 0
    _ANCHOR_GATE_BY_TASK.clear()
    _ANCHOR_GATE_CALLS = 0


_NEGATIVE_INSTANCES = 0
_NEGATIVE_CALLS = 0


def _note_negative_instances(trained: int) -> None:
    """Cumulative count of negative instances actually trained, with the usual log backoff."""
    global _NEGATIVE_INSTANCES, _NEGATIVE_CALLS
    _NEGATIVE_INSTANCES += trained
    _NEGATIVE_CALLS += 1
    n = _NEGATIVE_CALLS
    if n in (1, 200, 1000, 5000) or (n > 5000 and n % 25000 == 0):
        logger.info("record negative instances: %d trained on role fields, cumulative over %d natural groups",
                    _NEGATIVE_INSTANCES, n)


def _negative_instances(group: RecordGroupOutput, records, anchor_qid: int, trained: set, k: int) -> List[int]:
    """The k highest-scoring instances that are NOT gold: not trained, seeded, no overlap with a gold trigger."""
    gold_spans = [(int(s[0]), int(s[1])) for r in records
                  if (a := r.field_for_query(anchor_qid)) is not None and a.values for s in a.values[0]]
    picked = []
    for i in torch.argsort(group.object_logits.detach(), descending=True).tolist():
        if len(picked) >= k:
            break
        span = group.instance_spans[i] if i < len(group.instance_spans) else None
        if i in trained or group.instance_seed[i] is None or span is None:
            continue
        if any(span[0] < ge and gs < span[1] for gs, ge in gold_spans):
            continue
        picked.append(i)
    return picked


def _negative_role_loss(group: RecordGroupOutput, inst: int, hard_k: int = 0) -> torch.Tensor:
    """ROLE fields only, empty target: "none of these candidates is your argument".

    The scalar trigger field is skipped on purpose. Trained only on gold, every instance points
    its trigger field at its own seed with certainty (P(ABSENT) = 0.0000 on real batches), so an
    ABSENT target there is a ~30-nat error that turns the field into an existence classifier --
    existence is a separate mechanism, built where the gate can see argument evidence.
    """
    rows = [group.assign_logits[f][inst] for f, fs in enumerate(group.field_specs) if not fs.cardinality.is_scalar]
    if not rows:
        return group.object_logits.new_zeros(())
    return sum(_list_field_bce(row, [], hard_k) for row in rows) / len(rows)


_COLUMN_TERMS = 0
_COLUMN_WINS = 0
_COLUMN_PAIRS = 0
_COLUMN_CALLS = 0


def _note_link_columns(terms: int, wins: int, pairs: int) -> None:
    """In-run junction gate: share of (gold trigger, false trigger) pairs where gold wins the column."""
    global _COLUMN_TERMS, _COLUMN_WINS, _COLUMN_PAIRS, _COLUMN_CALLS
    _COLUMN_TERMS += terms
    _COLUMN_WINS += wins
    _COLUMN_PAIRS += pairs
    _COLUMN_CALLS += 1
    n = _COLUMN_CALLS
    if n in (1, 200, 1000, 5000) or (n > 5000 and n % 25000 == 0):
        logger.info("junction column loss: %d gold-argument columns trained; gold trigger beats false trigger "
                    "in %.3f of %d pairs (junction AUC, cumulative over %d natural groups)",
                    _COLUMN_TERMS, _COLUMN_WINS / max(_COLUMN_PAIRS, 1), _COLUMN_PAIRS, n)


def _coref_link_loss(group: RecordGroupOutput, records, anchor_qid: int, span_indices, k: int):
    """(summed BCE, n pairs, wins, ranked pairs) over trigger-mention pairs: same gold record = 1,
    different gold records = 0 (the hard negatives), a gold mention vs the k hardest false triggers = 0.
    ``wins/ranked`` is the in-run AUC: positive pairs scoring above negative pairs that share a mention."""
    af = group.field_query_ids.index(anchor_qid)
    seed = {s[1]: i for i, s in enumerate(group.instance_seed) if s is not None and s[0] == af}
    owner = {}
    for ri, rec in enumerate(records):
        a = rec.field_for_query(anchor_qid)
        for c in _resolve_value_cols(a.values[0], span_indices[af]) if a is not None and a.values else []:
            if (c - 1) in seed:
                owner.setdefault(seed[c - 1], ri)
    gold = sorted(owner)
    false = _negative_instances(group, records, anchor_qid, set(gold), k)
    logits = group.coref_logits
    pos, neg = [], []
    for x, i in enumerate(gold):
        for j in gold[x + 1:]:
            (pos if owner[i] == owner[j] else neg).append((i, j))
        neg.extend((i, f) for f in false)
    if not pos and not neg:
        return logits.new_zeros(()), 0, 0, 0
    idx = pos + neg
    target = logits.new_tensor([1.0] * len(pos) + [0.0] * len(neg))
    scores = torch.stack([logits[i, j] for i, j in idx])
    loss = F.binary_cross_entropy_with_logits(scores, target, reduction="sum")
    wins = ranked = 0
    if pos and neg:
        p, n = scores[:len(pos)].detach(), scores[len(pos):].detach()
        wins = int((p[:, None] > n[None, :]).sum())
        ranked = p.numel() * n.numel()
    return loss, len(idx), wins, ranked


_CPAIR = {"pairs": 0, "wins": 0, "ranked": 0, "calls": 0}


def _note_coref_pairs(n_pairs: int, wins: int, ranked: int) -> None:
    """In-run proof the coreference link trains, and its pair AUC on the training batch, with backoff."""
    _CPAIR["pairs"] += n_pairs
    _CPAIR["wins"] += wins
    _CPAIR["ranked"] += ranked
    _CPAIR["calls"] += 1
    n = _CPAIR["calls"]
    if n in (1, 200, 1000, 5000) or (n > 5000 and n % 25000 == 0):
        logger.info("coref link: %d mention pairs trained; coreferent pairs outscore non-coreferent in %.3f of %d "
                    "(pair AUC, cumulative over %d natural groups)",
                    _CPAIR["pairs"], _CPAIR["wins"] / max(_CPAIR["ranked"], 1), _CPAIR["ranked"], n)


_COREF = {"records": 0, "owners": 0, "calls": 0}


def _note_coreferent_owners(owners: int) -> None:
    """In-run proof the treatment applied: records trained and mentions owning them, with backoff."""
    _COREF["records"] += 1
    _COREF["owners"] += owners
    _COREF["calls"] += 1
    n = _COREF["calls"]
    if n in (1, 200, 1000, 5000) or (n > 5000 and n % 25000 == 0):
        logger.info("coreferent ownership: %d records trained through %d seeded mentions (%.2f per record)",
                    _COREF["records"], _COREF["owners"], _COREF["owners"] / max(_COREF["records"], 1))


def _column_loss(group: RecordGroupOutput, records, anchor_qid: int, span_indices, k: int):
    """(sum of column terms, n terms, wins, pairs): which trigger owns each gold argument."""
    af = group.field_query_ids.index(anchor_qid)
    seed_to_inst = {s[1]: i for i, s in enumerate(group.instance_seed) if s is not None and s[0] == af}
    owners = []
    for rec in records:
        a = rec.field_for_query(anchor_qid)
        cols = _resolve_value_cols(a.values[0], span_indices[af]) if a is not None and a.values else []
        insts = sorted({seed_to_inst[c - 1] for c in cols if (c - 1) in seed_to_inst})
        if insts:
            owners.append((rec, insts))
    gold_all = {i for _, insts in owners for i in insts}
    false = _negative_instances(group, records, anchor_qid, gold_all, k)
    total, terms, wins, pairs = group.object_logits.new_zeros(()), 0, 0, 0
    if not false:
        return total, 0, 0, 0
    for rec, insts in owners:
        for f, fspec in enumerate(group.field_specs):
            if fspec.cardinality.is_scalar:
                continue
            cols, _ = _field_target_cols(fspec, rec, span_indices[f])
            for c in sorted(set(cols)):
                column = group.assign_logits[f][:, c]
                pos, neg = column[insts], column[false]
                total = total - (torch.logsumexp(pos, 0) - torch.logsumexp(torch.cat([pos, neg]), 0))
                terms += 1
                best = pos.detach().max()
                wins += int((neg.detach() < best).sum())
                pairs += len(false)
    return total, terms, wins, pairs


def compute_group_loss(group: RecordGroupOutput, records: Sequence[RecordTarget],
                       negative_instances: int = 0, role_hard_negatives: int = 0,
                       column_negatives: int = 0, coreferent_ownership: bool = False,
                       coref_negatives: int = 0) -> Dict[str, torch.Tensor]:
    """Compute object and field-assignment losses for one record group.

    ``negative_instances`` > 0 (natural mode) also trains that many of the highest-scoring
    FALSE instances on their ROLE fields (no candidate is theirs), returned as ``negative_loss``
    (mean over the group's negatives) and ``negative_count`` for a separate batch mean.
    """
    device = group.object_logits.device
    zero = torch.zeros((), device=device)
    span_indices = [_span_index(spans) for spans in group.field_spans]
    ni = group.num_instances
    if group.spec.mode == "natural":
        anchor_qid = group.spec.anchor_query_id
        anchor_f_idx = group.field_query_ids.index(anchor_qid)
        seed_to_inst = {
            seed[1]: i for i, seed in enumerate(group.instance_seed)
            if seed is not None and seed[0] == anchor_f_idx
        }
        field_loss, n, trained = zero, 0, set()
        for record in records:
            aft = record.field_for_query(anchor_qid)
            if aft is None or not aft.values:
                _note_anchor_gate("no_gold_anchor", group.spec.task_type)
                continue
            # THE SUPERVISION GATE. The gold anchor is resolved against the MODEL'S OWN
            # candidate spans, so a record whose anchor the model did not propose trains
            # NOTHING -- not the anchor, not any of its fields. For events the anchor is
            # the TRIGGER (processing/records.py:70), so a missed trigger silently discards
            # that instance's entire argument supervision. See _note_anchor_gate.
            cols = _resolve_value_cols(aft.values[0], span_indices[anchor_f_idx])
            if not cols:
                _note_anchor_gate("anchor_not_proposed", group.spec.task_type)
                continue
            # One owner (historical: the first seeded mention) or, with coreferent_ownership, EVERY
            # seeded mention, averaged so the record's weight does not grow with its mention count.
            owners = (sorted({seed_to_inst[c - 1] for c in cols if (c - 1) in seed_to_inst}) if coreferent_ownership
                      else [i for i in [seed_to_inst.get(cols[0] - 1)] if i is not None])
            if not owners:
                _note_anchor_gate("anchor_not_seeded", group.spec.task_type)
                continue
            _note_anchor_gate("trained", group.spec.task_type)
            field_loss = field_loss + sum(_instance_field_loss(group, i, record, span_indices, role_hard_negatives)
                                          for i in owners) / len(owners)
            n += 1
            trained.update(owners)
            if coreferent_ownership:
                _note_coreferent_owners(len(owners))
        out = {
            "object_loss": zero,
            "field_loss": field_loss / max(n, 1),
            "object_count": 0,
            "field_count": n * len(group.field_specs),
        }
        if negative_instances > 0:
            # Returned SEPARATELY and averaged on their own in the batch, so negatives never enter
            # the gold denominator -- folding an absent type's near-zero loss into it diluted the
            # gold field loss (traced: 0.0173 -> 0.0099, record gradient halved).
            negatives = _negative_instances(group, records, anchor_qid, trained, negative_instances)
            if negatives:
                out["negative_loss"] = sum(_negative_role_loss(group, i, role_hard_negatives)
                                           for i in negatives) / len(negatives)
                out["negative_count"] = len(negatives)
            _note_negative_instances(len(negatives))
        if column_negatives > 0:
            # The JUNCTION's training signal, returned separately with its own count (never in the
            # gold field denominator). R is identical down a column, so only the link can lower it.
            col, terms, wins, pairs = _column_loss(group, records, anchor_qid, span_indices, column_negatives)
            if terms:
                out["column_loss"] = col / terms
                out["column_count"] = terms
            _note_link_columns(terms, wins, pairs)
        if coref_negatives > 0 and group.coref_logits is not None:
            loss, n_pairs, wins, pairs = _coref_link_loss(group, records, anchor_qid, span_indices, coref_negatives)
            if n_pairs:
                out["coref_loss"] = loss / n_pairs
                out["coref_count"] = n_pairs
            _note_coref_pairs(n_pairs, wins, pairs)
        return out

    count = len(records)
    if count == 0:
        obj_target = torch.zeros(ni, device=device)
        object_loss = F.binary_cross_entropy_with_logits(group.object_logits, obj_target) if ni else zero
        return {
            "object_loss": object_loss,
            "field_loss": zero,
            "object_count": ni,
            "field_count": 0,
        }
    if ni < count:
        raise TargetCapacityError(
            f"record group task={group.spec.task_index} has {count} gold instances "
            f"but only {ni} instance hypotheses (mode={group.spec.mode}); "
            "increase boundary_head.record_instance_queries or candidate budget."
        )
    # The assignment cost only feeds the (non-differentiable) Hungarian solver,
    # so build it without autograd; matched object/field losses are rebuilt with
    # gradients below. This avoids constructing and discarding an Ni x count graph.
    with torch.no_grad():
        obj_logp = F.logsigmoid(group.object_logits)
        cost = torch.zeros(ni, count, device=device)
        for i in range(ni):
            for j, record in enumerate(records):
                cost[i, j] = -(obj_logp[i] + _instance_field_logprob(group, i, record, span_indices))
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_rows = {
        int(row) for row in row_ind.tolist() if 0 <= int(row) < ni
    }
    obj_target = torch.zeros(ni, device=device)
    for row in matched_rows:
        obj_target[row] = 1.0
    object_loss = F.binary_cross_entropy_with_logits(group.object_logits, obj_target)
    field_loss = zero
    valid_pairs = [
        (int(row), int(col))
        for row, col in zip(row_ind.tolist(), col_ind.tolist())
        if 0 <= int(row) < ni and 0 <= int(col) < len(records)
    ]
    for row, col in valid_pairs:
        field_loss = field_loss + _instance_field_loss(
            group, row, records[col], span_indices, role_hard_negatives
        )
    return {
        "object_loss": object_loss,
        "field_loss": field_loss / max(len(valid_pairs), 1),
        "object_count": ni,
        "field_count": len(valid_pairs) * len(group.field_specs),
    }
