"""The three code-review probes (Issues 2, 5a, 5b): each must agree with the code it measures.

Group: one event type, fields [trigger (anchor, scalar), Location (list)]. Instances 0..4 are the trigger candidates;
record A's trigger is candidate 0, B's is candidate 2. C's trigger (60,61) is not a candidate (not proposed); D's
trigger is candidate 4, which is a candidate but seeds no instance (not seeded).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

import gliner2.models.boundary.records as R
from gliner2.processing.records import FieldCardinality, RecordFieldSpec, RecordSpec
from gliner2.processing.targets import RecordFieldTarget, RecordTarget

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "train"))
import probe_absent_anchors as PA  # noqa: E402
import probe_anchor_field as PF  # noqa: E402
import probe_supervision_gate as PG  # noqa: E402

TRIG = RecordFieldSpec(query_id=0, name="trigger", role_index=0, cardinality=FieldCardinality.REQUIRED_ONE, is_anchor=True)
LOC = RecordFieldSpec(query_id=1, name="Location", role_index=1, cardinality=FieldCardinality.ZERO_OR_MORE)
SPEC = RecordSpec(task_index=0, task_name="Deploy", task_type="events", mode="natural", fields=(TRIG, LOC), anchor_query_id=0)


def record(rid, trigger, locs):
    return RecordTarget(rid, 0, fields=(RecordFieldTarget(0, values=((trigger,),)),
                                        RecordFieldTarget(1, values=tuple((s,) for s in locs))), anchor_query_id=0)


def group(seed_last=True):
    torch.manual_seed(0)
    anchor_spans = torch.tensor([[2, 3], [10, 11], [20, 21], [30, 31], [40, 41]])
    loc_spans = torch.tensor([[5, 7], [25, 27], [50, 52]])
    ni = anchor_spans.shape[0]
    seeds = [(0, c) for c in range(ni)]
    if not seed_last:
        seeds[4] = None
    return R.RecordGroupOutput(
        spec=SPEC, object_logits=torch.tensor([0.1, 0.2, 0.3, 2.0, 1.5]),
        assign_logits=[torch.randn(ni, 1 + ni), torch.randn(ni, 1 + loc_spans.shape[0])],
        field_query_ids=[0, 1], field_specs=[TRIG, LOC], field_spans=[anchor_spans, loc_spans],
        field_cand_mask=[torch.ones(ni, dtype=torch.bool), torch.ones(3, dtype=torch.bool)],
        field_cand_logits=[torch.zeros(ni), torch.zeros(3)],
        instance_seed=seeds, instance_spans=[tuple(s) for s in anchor_spans.tolist()],
    )


def split(g, inst, rec, mode, log=False):
    spans = [R._span_index(s) for s in g.field_spans]
    PF.MODE[0], PF.LOG[0] = mode, log
    try:
        return PF.field_loss_split(g, inst, rec, spans)
    finally:
        PF.MODE[0], PF.LOG[0] = "all", False


def test_anchor_split_reproduces_the_field_loss_and_adds_back_linearly():
    g, a = group(), record("A", (2, 3), [(5, 7)])
    spans = [R._span_index(s) for s in g.field_spans]
    real = R._instance_field_loss(g, 0, a, spans)
    assert torch.allclose(split(g, 0, a, "all"), real)
    anchor = R._scalar_field_nll(g.assign_logits[0][0], [1]) / 2
    role = R._list_field_bce(g.assign_logits[1][0], [1], 0) / 2
    assert torch.allclose(split(g, 0, a, "no_roles"), anchor)
    assert torch.allclose(split(g, 0, a, "no_anchor"), role)
    assert float(split(g, 0, a, "none")) == 0.0


def test_anchor_rows_are_logged_only_on_the_logging_forward():
    PF.ROWS.clear()
    g, a = group(), record("A", (2, 3), [(5, 7)])
    split(g, 0, a, "all")
    assert PF.ROWS == []
    split(g, 0, a, "all", log=True)
    (row,) = PF.ROWS
    assert row["self_in_target"] and row["targets"] == 1 and row["n_fields"] == 2 and row["candidates"] == 5
    assert abs(row["nll"] - float(R._scalar_field_nll(g.assign_logits[0][0], [1]))) < 1e-6
    PF.ROWS.clear()


def test_supervision_gate_replica_agrees_with_the_code_counter():
    g = group(seed_last=False)
    recs = [record("A", (2, 3), [(5, 7)]), record("B", (20, 21), [(5, 7), (25, 27)]),
            record("C", (60, 61), [(50, 52)]), record("D", (40, 41), [(25, 27), (50, 52)])]
    for k in PG.TALLY:
        PG.TALLY[k] = 0
    R.reset_anchor_gate()
    PG.spy(R.compute_group_loss)(g, recs, coreferent_ownership=True)
    st = R.anchor_gate_stats()["by_task"]
    assert st.get("events/trained") == 2 and st.get("events/anchor_not_proposed") == 1 and st.get("events/anchor_not_seeded") == 1
    assert PG.TALLY == {"records": 4, "dropped": 2, "args": 6, "args_dropped": 3}


def test_absent_anchor_count_classifies_anchor_role_and_entity_queries():
    queries = tuple(SimpleNamespace(query_id=q, task_type=t) for q, t in
                    ((0, "events"), (1, "events"), (2, "events"), (3, "events"), (4, "events"),
                     (5, "entities"), (6, "entities"), (7, "entities")))
    absent_type = RecordSpec(task_index=1, task_name="Attack", task_type="events", mode="natural",
                             fields=(RecordFieldSpec(2, "trigger", 0, is_anchor=True), RecordFieldSpec(3, "Target", 1),
                                     RecordFieldSpec(4, "Place", 2)), anchor_query_id=2)
    mask = torch.zeros(1, 8, 2, dtype=torch.bool)
    mask[0, 0, 0] = mask[0, 1, 0] = mask[0, 5, 1] = True          # Deploy present with a Location; entity 5 present
    batch = SimpleNamespace(query_layouts=[SimpleNamespace(queries=queries)], record_specs=[{0: SPEC, 1: absent_type}],
                            targets=SimpleNamespace(mention_mask=mask))
    assert PA.count_batch(batch) == {"anchor": [1, 1], "role": [1, 2], "entity": [1, 2]}


def test_anchor_probe_replaces_the_real_field_loss_inside_compute_group_loss():
    g = group()
    recs = [record("A", (2, 3), [(5, 7)]), record("B", (20, 21), [(5, 7), (25, 27)])]
    real = R._instance_field_loss
    expected = R.compute_group_loss(g, recs, coreferent_ownership=True)["field_loss"]
    R._instance_field_loss = PF.field_loss_split
    try:
        got = R.compute_group_loss(g, recs, coreferent_ownership=True)["field_loss"]
    finally:
        R._instance_field_loss = real
    assert torch.allclose(got, expected)
