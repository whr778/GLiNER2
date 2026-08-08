"""Records decoded through the joint beam (JOINT_IE_SCALING sec 3b).

Role edges carry `hypothesis` = the trigger node key, so grouping solution edges by
it reconstitutes event instances. Output must match the greedy record shape exactly,
so the eval harness can run both decode arms unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from gliner2.joint_ie.candidates import EdgeCandidate, NodeCandidate
from gliner2.joint_ie.optimizers.base import JointSolution
from gliner2.processing.records import FieldCardinality, RecordFieldSpec, RecordSpec

TEXT = "bombed the market with explosives"
#        0-6    7-10 11-17 18-22 23-33
START_MAP = [0, 7, 11, 18, 23]
END_MAP = [6, 10, 17, 22, 33]

QUERY_TYPES = ["trigger", "target", "instrument"]

TRIGGER = RecordFieldSpec(query_id=0, name="trigger", role_index=0,
                          cardinality=FieldCardinality.REQUIRED_ONE, is_anchor=True)
TARGET = RecordFieldSpec(query_id=1, name="target", role_index=1,
                         cardinality=FieldCardinality.OPTIONAL_ONE)
INSTRUMENT = RecordFieldSpec(query_id=2, name="instrument", role_index=2,
                             cardinality=FieldCardinality.ZERO_OR_MORE)

SPEC = RecordSpec(
    task_index=0, task_name="attack", task_type="events", mode="natural",
    fields=(TRIGGER, TARGET, INSTRUMENT), anchor_query_id=0,
)

# _record_groups yields RecordGroupOutputs, each carrying its own spec.
_GROUP = SimpleNamespace(spec=SPEC)


def _model():
    from gliner2 import ExtractorConfig
    from gliner2.inference.engine import BoundaryExtractor
    from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
    from tests.fixtures.tiny_encoder import build_tiny_encoder_config
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    tokenizer = build_tiny_tokenizer()
    head = dict(TINY_BOUNDARY_HEAD)
    head.update(enable_records=True, record_dim=24, record_instance_queries=8,
                decode_mode="joint")
    model = BoundaryExtractor(
        ExtractorConfig(model_name="tiny-bert-fixture", architecture="boundary",
                        boundary_head=head, token_pooling="first"),
        encoder_config=build_tiny_encoder_config(vocab_size=len(tokenizer)),
        tokenizer=tokenizer,
    )
    model.eval()
    return model


def _solution():
    """One event: trigger [0,1), target [2,3), two instruments [3,4) and [4,5)."""
    trigger = ("trigger", 0, 1)
    nodes = tuple(
        NodeCandidate(t, s, e, 3.0, candidate_id=(t, s, e))
        for t, s, e in [("trigger", 0, 1), ("target", 2, 3),
                        ("instrument", 3, 4), ("instrument", 4, 5)]
    )
    edges = (
        EdgeCandidate("attack::target", trigger, ("target", 2, 3), 2.0,
                      slot="target", hypothesis=trigger),
        EdgeCandidate("attack::instrument", trigger, ("instrument", 3, 4), 1.0,
                      hypothesis=trigger),
        EdgeCandidate("attack::instrument", trigger, ("instrument", 4, 5), 1.0,
                      hypothesis=trigger),
    )
    return JointSolution(nodes, edges, 6.0)


def _format(model, include_spans=False):
    return model._format_joint_records(
        [_GROUP], _solution(), QUERY_TYPES, offset=0,
        start_map=START_MAP, end_map=END_MAP, text=TEXT, text_len=5,
        include_confidence=False, include_spans=include_spans,
    )


def test_role_edges_rebuild_an_event_instance():
    """Grouping by hypothesis reconstitutes the instance, in greedy's shape."""
    out = _format(_model())

    assert list(out) == ["attack"]
    assert len(out["attack"]) == 1
    inst = out["attack"][0]

    # Field order follows the schema, scalars are strings, lists are lists.
    assert list(inst) == ["trigger", "target", "instrument"]
    assert inst["trigger"] == "bombed"          # anchor comes from the trigger key
    assert inst["target"] == "market"           # scalar -> str
    assert inst["instrument"] == ["with", "explosives"]  # list -> list, both kept


def test_multi_valued_role_keeps_every_filler():
    """A ZERO_OR_MORE role must not be capped at one; decision B exists for this."""
    inst = _format(_model())["attack"][0]
    assert isinstance(inst["instrument"], list)
    assert len(inst["instrument"]) == 2


def test_two_triggers_produce_two_instances():
    """`hypothesis` is the trigger key, so distinct triggers never merge."""
    model = _model()
    a, b = ("trigger", 0, 1), ("trigger", 2, 3)
    nodes = tuple(
        NodeCandidate(t, s, e, 3.0, candidate_id=(t, s, e))
        for t, s, e in [("trigger", 0, 1), ("trigger", 2, 3), ("target", 4, 5)]
    )
    edges = (
        EdgeCandidate("attack::target", a, ("target", 4, 5), 2.0,
                      slot="target", hypothesis=a),
        EdgeCandidate("attack::target", b, ("target", 4, 5), 2.0,
                      slot="target", hypothesis=b),
    )
    out = model._format_joint_records(
        [_GROUP], JointSolution(nodes, edges, 4.0), QUERY_TYPES, offset=0,
        start_map=START_MAP, end_map=END_MAP, text=TEXT, text_len=5,
        include_confidence=False, include_spans=False,
    )
    assert len(out["attack"]) == 2
    assert {i["trigger"] for i in out["attack"]} == {"bombed", "market"}


def test_plain_relations_are_not_treated_as_roles():
    """An edge without '::' is a relation and must not become a record."""
    model = _model()
    node_ids = [("person", 0, 1), ("org", 2, 3)]
    nodes = tuple(NodeCandidate(t, s, e, 3.0, candidate_id=(t, s, e))
                  for t, s, e in node_ids)
    solution = JointSolution(
        nodes, (EdgeCandidate("works_for", node_ids[0], node_ids[1], 2.0),), 5.0
    )
    out = model._format_joint_records(
        [_GROUP], solution, QUERY_TYPES, offset=0, start_map=START_MAP,
        end_map=END_MAP, text=TEXT, text_len=5,
        include_confidence=False, include_spans=False,
    )
    assert out == {}


def test_joint_mode_does_not_double_emit_records():
    """Joint mode must skip the greedy record pass; both running would duplicate."""
    from gliner2.processor import SamplingConfig, SchemaTransformer
    from gliner2.training import ExtractorCollator
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    model = _model()
    assert model.boundary_settings.decode_mode == "joint"
    processor = SchemaTransformer(
        tokenizer=build_tiny_tokenizer(), sampling_config=SamplingConfig(),
    )
    collator = ExtractorCollator(processor, is_training=False, architecture="boundary")
    batch = collator([(TEXT, {
        "json_structures": [{"attack": {"trigger": "bombed", "target": "market"}}],
        "record_metadata": {"attack": {"mode": "natural", "anchor": "trigger"}},
    })])
    with torch.no_grad():
        results = model._extract_from_batch(
            batch, 0.5, [{}], include_confidence=False, include_spans=False,
        )
    # Untrained: content is unpredictable, but any emitted instance list must not
    # contain duplicates, which is exactly what double-emission would produce.
    for value in results[0].values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            assert len(value) == len({str(sorted(v.items())) for v in value})
