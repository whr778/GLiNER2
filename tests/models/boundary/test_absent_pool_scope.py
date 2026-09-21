"""`absent_negatives_scope`: which event queries the absent pool may touch.

absneg2 measured the two halves of the `events` task bucket moving in OPPOSITE directions --
argument recall +0.0329 (754 gold arguments left "never found") against event_type recall
-0.0818 and trigger recall -0.0468. Pooling absent ANCHORS teaches the model not to propose
event instances; pooling absent ROLES teaches it to rank fillers. This scopes the pool to the
second.
"""

import dataclasses
import types

import pytest
import torch

from gliner2.configuration import validate_boundary_head


def _spec(task_type, fields):
    return types.SimpleNamespace(
        task_type=task_type,
        fields=tuple(types.SimpleNamespace(query_id=q, name=n, is_anchor=a)
                     for q, n, a in fields),
    )


class _Model:
    """Just enough of BoundaryExtractorModel to exercise the method under test."""
    from gliner2.models.boundary.model import BoundaryExtractorModel
    _absent_pool_task_ids = BoundaryExtractorModel._absent_pool_task_ids

    def __init__(self, scope):
        self.boundary_settings = types.SimpleNamespace(absent_negatives_scope=scope)


def _batch():
    # query 0-1 entities, 2 event trigger (anchor), 3-4 event roles
    return types.SimpleNamespace(record_specs=(
        {0: _spec("events", [(2, "trigger", True), (3, "Victim", False), (4, "Place", False)])},
    ))


IDS = torch.tensor([[0, 0, 2, 2, 2]])
MASK = torch.ones(1, 5, dtype=torch.bool)


def test_default_scope_is_identity():
    """The measured path must stay bit-identical, so return the SAME object."""
    out = _Model("all")._absent_pool_task_ids(_batch(), MASK, IDS)
    assert out is IDS


def test_roles_scope_excludes_only_the_anchor():
    out = _Model("roles")._absent_pool_task_ids(_batch(), MASK, IDS)
    assert out.tolist() == [[0, 0, -1, 2, 2]]
    assert IDS.tolist() == [[0, 0, 2, 2, 2]], "input tensor must not be mutated"


def test_none_task_ids_stays_none():
    assert _Model("roles")._absent_pool_task_ids(_batch(), MASK, None) is None


def test_without_record_specs_nothing_changes():
    empty = types.SimpleNamespace(record_specs=None)
    assert _Model("roles")._absent_pool_task_ids(empty, MASK, IDS) is IDS


def test_non_event_specs_are_left_alone():
    """A structure record also has an anchor; only EVENT anchors are in scope."""
    batch = types.SimpleNamespace(record_specs=(
        {0: _spec("json_structures", [(2, "anchor", True), (3, "field", False)])},
    ))
    out = _Model("roles")._absent_pool_task_ids(batch, MASK, IDS)
    assert out.tolist() == [[0, 0, 2, 2, 2]]


def test_query_id_out_of_range_is_ignored():
    batch = types.SimpleNamespace(record_specs=(
        {0: _spec("events", [(99, "trigger", True)])},
    ))
    out = _Model("roles")._absent_pool_task_ids(batch, MASK, IDS)
    assert out.tolist() == [[0, 0, 2, 2, 2]]


@pytest.mark.parametrize("scope", ["all", "roles"])
def test_config_accepts_both_scopes(scope):
    assert validate_boundary_head({"absent_negatives_scope": scope})[
        "absent_negatives_scope"] == scope


def test_config_rejects_an_unknown_scope():
    with pytest.raises(ValueError, match="absent_negatives_scope"):
        validate_boundary_head({"absent_negatives_scope": "anchors_only"})


def test_default_config_is_all():
    assert validate_boundary_head({})["absent_negatives_scope"] == "all"
