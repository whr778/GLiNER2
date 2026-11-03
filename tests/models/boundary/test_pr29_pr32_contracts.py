"""PR-29--32 dense structured-head contracts."""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from gliner2.configuration import BoundaryHeadSettings, migrate_config_dict
from gliner2.models.base import QueryLayout
from gliner2.models.boundary.records import (
    DenseRecordBatchOutput,
    DenseRecordGroupOutput,
    build_dense_record_cost,
    compute_dense_batch_loss,
    compute_dense_group_loss,
)
from gliner2.models.boundary.relations import (
    RelationProposalSettings,
    RelationTypeSpec,
    TypedRelationPairGenerator,
)
from gliner2.models.outputs import CandidateTensorBatch
from gliner2.processing.records import (
    FieldCardinality,
    RecordFieldSpec,
    RecordSpec,
)
from gliner2.processing.targets import (
    RecordFieldTarget,
    RecordTarget,
    TargetCapacityError,
)


def _relation_candidates(batch: int, queries: int, candidates: int):
    starts = torch.arange(candidates).view(1, 1, candidates)
    indices = torch.stack((starts, starts + 1), -1).expand(batch, queries, -1, -1).clone()
    logits = torch.randn(batch, queries, candidates)
    valid = torch.rand(batch, queries, candidates) > 0.2
    return CandidateTensorBatch(
        indices=indices,
        proposal_logits=logits,
        pair_logits=logits,
        valid_mask=valid,
        query_mask=torch.ones(batch, queries, dtype=torch.bool),
    )


def _reference_pairs(candidates, specs, settings):
    result = set()
    for b, sample_specs in enumerate(specs):
        for r, spec in enumerate(sample_specs):
            heads, tails = [], []
            for qids, output in ((spec.head_query_ids, heads), (spec.tail_query_ids, tails)):
                for qid in qids:
                    for c in range(candidates.indices.shape[2]):
                        prob = float(torch.sigmoid(candidates.pair_logits[b, qid, c]))
                        if candidates.valid_mask[b, qid, c] and prob >= settings.argument_threshold:
                            span = candidates.indices[b, qid, c]
                            output.append((prob, int(span[0]), int(span[1]), qid))
                output.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
            heads = heads[: settings.heads_per_relation]
            tails = tails[: settings.tails_per_relation]
            pairs = [
                (hp * tp, hs, he, ts, te)
                for hp, hs, he, _ in heads
                for tp, ts, te, _ in tails
                if spec.allow_self or (hs, he) != (ts, te)
            ]
            pairs.sort(key=lambda x: -x[0])
            result.update((b, r, *pair[1:]) for pair in pairs[: settings.pair_cap])
    return result


def test_tensorized_relation_pair_set_matches_reference_randomized():
    random.seed(9)
    torch.manual_seed(9)
    for _ in range(25):
        candidates = _relation_candidates(3, 4, 7)
        specs = [
            [
                RelationTypeSpec("r0", (0, 1), (2, 3), allow_self=False),
                RelationTypeSpec("r1", (2,), (0, 3), allow_self=True),
            ]
            for _ in range(3)
        ]
        settings = RelationProposalSettings(
            heads_per_relation=4, tails_per_relation=5, pair_cap=11,
            argument_threshold=0.25,
        )
        pairs = TypedRelationPairGenerator(settings).generate_batched(
            candidates, [QueryLayout(queries=())] * 3, specs, compact=False
        )
        mask = pairs.pair_mask
        got = set(zip(
            pairs.batch_index[mask].tolist(), pairs.relation_index[mask].tolist(),
            pairs.head_start[mask].tolist(), pairs.head_end[mask].tolist(),
            pairs.tail_start[mask].tolist(), pairs.tail_end[mask].tolist(),
        ))
        assert got == _reference_pairs(candidates, specs, settings)


def test_dense_record_cost_matches_scalar_reference():
    torch.manual_seed(4)
    instances, fields, candidates, gold_count = 5, 2, 7, 3
    spec = RecordSpec(
        task_index=0, task_name="x", task_type="json_structures", mode="latent",
        fields=(
            RecordFieldSpec(0, "scalar", 0, FieldCardinality.OPTIONAL_ONE),
            RecordFieldSpec(1, "list", 1, FieldCardinality.ZERO_OR_MORE),
        ),
    )
    group = DenseRecordGroupOutput(
        spec=spec,
        object_logits=torch.randn(instances),
        assign_logits=torch.randn(instances, fields, 1 + candidates),
        instance_mask=torch.ones(instances, dtype=torch.bool),
        field_membership=torch.ones(fields, candidates, dtype=torch.bool),
        pool_spans=torch.stack((torch.arange(candidates), torch.arange(1, candidates + 1)), -1),
        field_specs=spec.fields,
        field_query_ids=torch.tensor([0, 1]),
        instance_pool_index=torch.arange(instances),
    )
    gold = torch.rand(gold_count, fields, candidates) > 0.75
    cost = build_dense_record_cost(group, gold)
    reference = torch.empty_like(cost)
    for i in range(instances):
        for n in range(gold_count):
            score = F.logsigmoid(group.object_logits[i])
            scalar_cols = gold[n, 0].nonzero().flatten() + 1
            if not len(scalar_cols):
                scalar_cols = torch.tensor([0])
            score += torch.logsumexp(F.log_softmax(group.assign_logits[i, 0], -1)[scalar_cols], 0)
            score -= F.binary_cross_entropy_with_logits(
                group.assign_logits[i, 1, 1:], gold[n, 1].float(), reduction="sum"
            )
            reference[i, n] = -score
    torch.testing.assert_close(cost, reference)


def test_dense_natural_loss_skips_missing_anchor_instead_of_using_row_zero():
    spec = RecordSpec(
        task_index=0,
        task_name="event",
        task_type="json_structures",
        mode="natural",
        fields=(
            RecordFieldSpec(
                0, "anchor", 0, FieldCardinality.REQUIRED_ONE, is_anchor=True
            ),
        ),
        anchor_query_id=0,
    )
    group = DenseRecordGroupOutput(
        spec=spec,
        object_logits=torch.zeros(2),
        assign_logits=torch.zeros(2, 1, 3),
        instance_mask=torch.ones(2, dtype=torch.bool),
        field_membership=torch.ones(1, 2, dtype=torch.bool),
        pool_spans=torch.tensor([[0, 1], [2, 3]]),
        field_specs=spec.fields,
        field_query_ids=torch.tensor([0]),
        instance_pool_index=torch.tensor([0, 1]),
    )
    missing = RecordTarget(
        "event:0",
        0,
        (RecordFieldTarget(0, (((5, 6),),)),),
        anchor_query_id=0,
    )
    losses = compute_dense_group_loss(group, [missing])
    assert losses["field_loss"].item() == 0.0
    assert losses["field_count"] == 0


def _missed_role_group(victim_values, victim_logits):
    """One natural event. Date is empty. Victim's candidate logits are the caller's."""
    spec = RecordSpec(
        task_index=0, task_name="event", task_type="events", mode="natural",
        fields=(
            RecordFieldSpec(0, "trigger", 0, FieldCardinality.REQUIRED_ONE, is_anchor=True),
            RecordFieldSpec(1, "date", 1, FieldCardinality.ZERO_OR_MORE),
            RecordFieldSpec(2, "victim", 2, FieldCardinality.ZERO_OR_MORE),
        ),
        anchor_query_id=0,
    )
    logits = torch.zeros(1, 3, 4)
    logits[0, 2, 1:] = victim_logits
    group = DenseRecordGroupOutput(
        spec=spec,
        object_logits=torch.zeros(1),
        assign_logits=logits,
        instance_mask=torch.ones(1, dtype=torch.bool),
        field_membership=torch.ones(3, 3, dtype=torch.bool),
        pool_spans=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        field_specs=spec.fields,
        field_query_ids=torch.tensor([0, 1, 2]),
        instance_pool_index=torch.tensor([0]),
    )
    records = [RecordTarget(
        "e0", 0, (
            RecordFieldTarget(0, (((0, 1),),)),
            RecordFieldTarget(1, ()),
            RecordFieldTarget(2, victim_values),
        ),
        anchor_query_id=0,
    )]
    return group, records


def _field_mean(anchor, date, victim):
    return (anchor + date + victim) / 3


def test_dense_unresolved_role_is_withheld_and_an_empty_role_is_not():
    """A gold span outside the pool adds nothing. An empty role stays all-negative.

    Traced before the change, both paths: field loss 0.924196, which is
    (log 4 + 2 * log 2) / 3. The missed victim was the second log 2.
    """
    victim_logits = torch.ones(3)
    group, records = _missed_role_group((((10, 11),),), victim_logits)
    anchor = -F.log_softmax(group.assign_logits[0, 0], -1)[1]
    date = F.binary_cross_entropy_with_logits(
        group.assign_logits[0, 1, 1:], torch.zeros(3), reduction="mean"
    )
    as_empty = F.binary_cross_entropy_with_logits(
        victim_logits, torch.zeros(3), reduction="mean"
    )
    withheld = _field_mean(anchor, date, group.assign_logits.new_zeros(()))
    trained_as_empty = _field_mean(anchor, date, as_empty)
    losses = compute_dense_group_loss(group, records)
    assert losses["field_count"] == 3
    assert float(losses["field_loss"]) == pytest.approx(float(withheld))
    assert float(withheld) < float(trained_as_empty)

    empty_group, empty_records = _missed_role_group((), victim_logits)
    empty = compute_dense_group_loss(empty_group, empty_records)
    assert float(empty["field_loss"]) == pytest.approx(float(trained_as_empty))

    partial_logits = torch.tensor([0.0, 1.0, 0.0])
    partial_group, partial_records = _missed_role_group(
        (((2, 3), (10, 11)),), partial_logits
    )
    matched = F.binary_cross_entropy_with_logits(
        partial_logits, torch.tensor([0.0, 1.0, 0.0]), reduction="mean"
    )
    partial_anchor = -F.log_softmax(partial_group.assign_logits[0, 0], -1)[1]
    partial_date = F.binary_cross_entropy_with_logits(
        partial_group.assign_logits[0, 1, 1:], torch.zeros(3), reduction="mean"
    )
    partial = compute_dense_group_loss(partial_group, partial_records)
    assert float(partial["field_loss"]) == pytest.approx(
        float(_field_mean(partial_anchor, partial_date, matched))
    )

    gold = torch.zeros(1, 1, 1, 3, 3, dtype=torch.bool)
    gold[0, 0, 0, 0, 0] = True
    batch = DenseRecordBatchOutput(
        object_logits=torch.zeros(1, 1, 1),
        assign_logits=group.assign_logits.view(1, 1, 1, 3, 4),
        instance_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        field_membership=torch.ones(1, 1, 3, 3, dtype=torch.bool),
        pool_spans=group.pool_spans.unsqueeze(0),
        field_mask=torch.ones(1, 1, 3, dtype=torch.bool),
        scalar_fields=torch.tensor([[[True, False, False]]]),
        modes=torch.zeros(1, 1, dtype=torch.long),
        anchor_fields=torch.zeros(1, 1, dtype=torch.long),
        group_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    batched = compute_dense_batch_loss(
        batch, gold, torch.ones(1, 1, 1, dtype=torch.bool),
        annotated=torch.tensor([[[[True, False, True]]]]),
    )
    assert float(batched["field_loss"]) == pytest.approx(float(withheld))


def test_dense_record_capacity_counts_only_real_instances():
    spec = RecordSpec(
        task_index=0,
        task_name="event",
        task_type="json_structures",
        mode="latent",
        fields=(RecordFieldSpec(0, "value", 0, FieldCardinality.OPTIONAL_ONE),),
    )
    group = DenseRecordGroupOutput(
        spec=spec,
        object_logits=torch.zeros(4),
        assign_logits=torch.zeros(4, 1, 5),
        instance_mask=torch.tensor([True, False, False, False]),
        field_membership=torch.ones(1, 4, dtype=torch.bool),
        pool_spans=torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]]),
        field_specs=spec.fields,
        field_query_ids=torch.tensor([0]),
        instance_pool_index=torch.arange(4),
    )
    records = [
        RecordTarget(str(index), 0, (), anchor_query_id=None)
        for index in range(2)
    ]
    with pytest.raises(TargetCapacityError, match="1 valid instance"):
        compute_dense_group_loss(group, records)


def test_new_and_legacy_structured_config_defaults():
    assert BoundaryHeadSettings().enable_records
    assert BoundaryHeadSettings().enable_relations
    migrated = migrate_config_dict({
        "architecture": "boundary", "config_version": 2, "boundary_head": {}
    })
    assert migrated["boundary_head"]["enable_records"] is False
    assert migrated["boundary_head"]["enable_relations"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_relation_training_pair_generation_has_no_host_transfer_cuda(monkeypatch):
    candidates = _relation_candidates(2, 3, 8)
    candidates = CandidateTensorBatch(
        **{
            name: getattr(candidates, name).cuda()
            for name in ("indices", "proposal_logits", "pair_logits", "valid_mask", "query_mask")
        }
    )
    original_cpu = torch.Tensor.cpu

    def forbidden_cpu(self, *args, **kwargs):
        raise AssertionError("relation proposal training path transferred to CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu)
    try:
        TypedRelationPairGenerator().generate_batched(
            candidates,
            [QueryLayout(queries=())] * 2,
            [[RelationTypeSpec("r", (0,), (1,))] for _ in range(2)],
            compact=False,
        )
    finally:
        monkeypatch.setattr(torch.Tensor, "cpu", original_cpu)
