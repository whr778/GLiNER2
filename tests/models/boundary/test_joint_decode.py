"""Joint (joint_ie beam) decode path on the boundary engine.

The greedy decode selects per query; the joint path routes the same boundary
candidate scores through the joint_ie typed-constraint beam so entities and
relations are chosen together. The load-bearing property is *key consistency*:
mention nodes and relation endpoints must be typed from one `QueryLayout`, or
every edge references a node that does not exist and the beam prunes them all.
"""

from __future__ import annotations

import torch

from gliner2.models.boundary.engine import _layout_from_ext_specs
from gliner2.models.boundary.relations import RelationTypeSpec
from gliner2.models.outputs import CandidateTensorBatch

TEXT = "Alice works for Acme"
START_MAP = [0, 6, 12, 16]
END_MAP = [5, 11, 15, 20]

SPECS = [
    {"group_index": 0, "field_index": 0, "task_type": "entities",
     "task_name": "entities", "field_name": "person"},
    {"group_index": 0, "field_index": 1, "task_type": "entities",
     "task_name": "entities", "field_name": "org"},
]


def _relation_model():
    """A relation-enabled tiny boundary model with a confident relation scorer."""
    from gliner2 import ExtractorConfig
    from gliner2.inference.engine import BoundaryExtractor
    from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
    from tests.fixtures.tiny_encoder import build_tiny_encoder_config
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    tokenizer = build_tiny_tokenizer()
    head = dict(TINY_BOUNDARY_HEAD)
    head.update(enable_relations=True, decode_mode="joint")
    model = BoundaryExtractor(
        ExtractorConfig(
            model_name="tiny-bert-fixture",
            architecture="boundary",
            boundary_head=head,
            token_pooling="first",
        ),
        encoder_config=build_tiny_encoder_config(vocab_size=len(tokenizer)),
        tokenizer=tokenizer,
    )
    with torch.no_grad():
        model.relation_scorer.mlp[-1].bias.fill_(10.0)
    return model


def _candidates():
    """person -> token [0,1) ("Alice"), org -> token [3,4) ("Acme"); both confident."""
    return CandidateTensorBatch(
        indices=torch.tensor([[[[0, 1]], [[3, 4]]]]),
        proposal_logits=torch.zeros(1, 2, 1),
        pair_logits=torch.full((1, 2, 1), 5.0),
        valid_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        query_mask=torch.ones(1, 2, dtype=torch.bool),
    )


def _core(model):
    return {
        "rel_specs": [[{
            "spec": RelationTypeSpec("works_for", (0,), (1,)),
            "query_state": torch.zeros(model.hidden_size),
        }]],
        "text_states": torch.zeros(1, 4, model.hidden_size),
    }


def test_layout_from_ext_specs_types_queries_by_role_name():
    """ext_specs order is candidate-slot order, so query_id indexes the layout."""
    layout = _layout_from_ext_specs(SPECS)
    assert [q.role_name for q in layout.queries] == ["person", "org"]
    assert layout.query(0).role_name == "person"
    assert layout.query(1).role_name == "org"


def test_joint_decode_links_relation_to_its_mention_nodes():
    """The crux: edges must reference mention nodes, so the beam keeps them.

    With the empty ``QueryLayout`` the greedy path used to pass, endpoint keys
    would be ``("0", 0, 1)`` while mention keys are ``("person", 0, 1)`` -- no
    match, and every edge would be pruned. Asserting the relation survives is
    the regression guard for that.
    """
    model = _relation_model()
    decoded = model._decode_joint(
        0,
        _core(model),
        _candidates(),
        threshold=0.5,
        offset=0,
        start_map=START_MAP,
        end_map=END_MAP,
        text=TEXT,
        text_len=4,
        include_confidence=False,
        include_spans=True,
        layout=_layout_from_ext_specs(SPECS),
        specs=SPECS,
    )

    assert decoded["works_for"] == [{
        "head": {"text": "Alice", "start": 0, "end": 5},
        "tail": {"text": "Acme", "start": 16, "end": 20},
    }]
    entities = decoded["entities"][0]
    assert [e["text"] for e in entities["person"]] == ["Alice"]
    assert [e["text"] for e in entities["org"]] == ["Acme"]


def test_joint_decode_matches_greedy_on_the_same_candidates():
    """Both arms must agree on an unambiguous case; they differ only under
    competition, which is what the scaling experiment measures."""
    model = _relation_model()
    common = dict(
        offset=0, start_map=START_MAP, end_map=END_MAP, text=TEXT, text_len=4,
        include_confidence=False, include_spans=True,
        layout=_layout_from_ext_specs(SPECS),
    )
    joint = model._decode_joint(
        0, _core(model), _candidates(), threshold=0.5, specs=SPECS, **common
    )
    greedy = model._decode_relations(
        0, _core(model), _candidates(), {"relation_metadata": {}},
        threshold=0.5, **common
    )
    assert joint["works_for"] == greedy["works_for"]


def test_decode_mode_defaults_to_greedy():
    """The shipped path must be untouched unless the flag is set."""
    from gliner2.configuration import BoundaryHeadSettings

    assert BoundaryHeadSettings().decode_mode == "greedy"


def test_joint_decode_mode_runs_through_the_public_extract_path():
    """The flag is wired: a relation-enabled model extracts with the beam on."""
    model = _relation_model()
    assert model.boundary_settings.decode_mode == "joint"
    result = model.extract_relations(TEXT, ["works_for"], threshold=0.5)
    assert isinstance(result, dict)
