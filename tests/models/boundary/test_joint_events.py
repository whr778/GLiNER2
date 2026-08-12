"""Events emitted from the joint beam (JOINT_IE_SCALING Phase A, Tier 1).

Events are supervised as MENTIONS -- one extractive query per trigger and per role -- so
the beam already selects their candidates: `query_types` covers every spec, and
`joint_decode` receives them. They were simply never collected out of the solution, and
`_decode_events` sits after the `if joint: ... continue` in `decode`, so **joint mode
emitted no events at all**. That is the same failure `_decode_events` was written to fix
on the greedy side, where every event metric read 0.0 at every threshold.

Why the existing suite could not catch it: `test_joint_records.py` and
`test_record_role_edges.py` construct `RecordSpec(task_type="events", ...)` BY HAND and
never call `compile_record_specs`, which only compiles `RECORD_TASK_TYPES ==
("json_structures",)`. They prove the role-edge machinery works for events while being
structurally unable to notice that no such spec is ever built from a real schema.

Shape parity with the greedy path is the load-bearing property: the eval harness reads
`{event_type: [{"triggers": [...], "arguments": [{"role", "entity"}]}]}` and must not be
able to tell the arms apart by structure.
"""

from __future__ import annotations

TEXT = "bombed the market with explosives"
#       0-6    7-10 11-17 18-22 23-33
START_MAP = [0, 7, 11, 18, 23]
END_MAP = [6, 10, 17, 22, 33]

# Qualified per query, as `_decode_joint` builds them.
QUERY_TYPES = ["attack::trigger", "attack::target", "attack::instrument"]

SPECS = [
    {"task_type": "events", "task_name": "attack", "field_name": "trigger",
     "field_index": 0},
    {"task_type": "events", "task_name": "attack", "field_name": "target",
     "field_index": 1},
    {"task_type": "events", "task_name": "attack", "field_name": "instrument",
     "field_index": 2},
]


def _model():
    from gliner2 import ExtractorConfig
    from gliner2.inference.engine import BoundaryExtractor
    from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
    from tests.fixtures.tiny_encoder import build_tiny_encoder_config
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    tokenizer = build_tiny_tokenizer()
    head = dict(TINY_BOUNDARY_HEAD)
    head.update(decode_mode="joint")
    model = BoundaryExtractor(
        ExtractorConfig(model_name="tiny-bert-fixture", architecture="boundary",
                        boundary_head=head, token_pooling="first"),
        encoder_config=build_tiny_encoder_config(vocab_size=len(tokenizer)),
        tokenizer=tokenizer,
    )
    model.eval()
    return model


def _spans(selected):
    """`event_spans` as `_decode_joint` accumulates it: qualified type -> span tuples."""
    out = {qt: [] for qt in QUERY_TYPES}
    for qtype, text, start, end in selected:
        out[qtype].append((text, 0.9, start, end))
    return out


def _format(model, selected, include_spans=False):
    return model._format_joint_events(
        SPECS, QUERY_TYPES, _spans(selected),
        include_confidence=False, include_spans=include_spans,
    )


def test_beam_selected_mentions_become_an_event():
    model = _model()
    out = _format(model, [
        ("attack::trigger", "bombed", 0, 6),
        ("attack::target", "the market", 7, 17),
        ("attack::instrument", "explosives", 23, 33),
    ])

    assert out == {"attack": [{
        "triggers": ["bombed"],
        "arguments": [{"role": "target", "entity": "the market"},
                      {"role": "instrument", "entity": "explosives"}],
    }]}


def test_shape_matches_the_greedy_path_exactly():
    """The eval harness must not be able to tell the arms apart by structure."""
    model = _model()
    out = _format(model, [
        ("attack::trigger", "bombed", 0, 6),
        ("attack::target", "the market", 7, 17),
    ])

    assert set(out) == {"attack"}
    instance = out["attack"][0]
    assert set(instance) == {"triggers", "arguments"}
    assert isinstance(instance["triggers"], list)
    assert all(set(a) == {"role", "entity"} for a in instance["arguments"])


def test_a_group_with_no_trigger_is_dropped():
    """Matches the greedy path: gold arguments are keyed by their trigger."""
    model = _model()
    out = _format(model, [("attack::target", "the market", 7, 17)])

    assert out == {}


def test_no_selected_mentions_emits_nothing():
    model = _model()

    assert _format(model, []) == {}


def test_multi_valued_role_keeps_every_filler():
    model = _model()
    out = _format(model, [
        ("attack::trigger", "bombed", 0, 6),
        ("attack::instrument", "with", 18, 22),
        ("attack::instrument", "explosives", 23, 33),
    ])

    assert out["attack"][0]["arguments"] == [
        {"role": "instrument", "entity": "with"},
        {"role": "instrument", "entity": "explosives"},
    ]


def test_joint_decode_wires_events_in():
    """The gap was structural: nothing called the assembler from the joint path.

    Asserting the call site exists is crude, but it is the property that was missing --
    the assembler itself was never the problem, its absence from `_decode_joint` was.
    """
    import inspect

    source = inspect.getsource(_model()._decode_joint)
    assert "_format_joint_events" in source
