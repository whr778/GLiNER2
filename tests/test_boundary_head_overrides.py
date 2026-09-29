"""``model.boundary_head`` overrides must reach the model on the ``pretrained`` path.

``_build_model`` applies leftover config keys with ``setattr(model.config, key,
value)``. That works for flat keys, but ``boundary_settings`` is built from
``config.boundary_head`` inside the model's ``__init__``, which ``from_pretrained``
has already run -- so a ``boundary_head`` override landed on the config and was
never read. Every such override in every ``pretrained:`` config was decorative.

It surfaced as a treatment arm that was a silent duplicate of its control: a
config setting ``boundary_head.task_loss_weights`` built a model reporting
``task_loss_weights=None``. The arm trained for ten minutes before the check
caught it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))

from train import _apply_boundary_head_overrides  # noqa: E402

from gliner2.configuration import BoundaryHeadSettings  # noqa: E402


class FakeConfig:
    def __init__(self, boundary_head):
        self.boundary_head = boundary_head


class FakeModel:
    """Stands in for a loaded checkpoint: a config carrying the CHECKPOINT's
    boundary_head, plus the settings object built from it at construction."""

    class FakePairScorer:
        """The real SparseBoundaryPairScorer takes ``use_inside_evidence`` as a
        constructor ARGUMENT and stores its own copy, which its forward reads."""

        def __init__(self, settings):
            self.use_inside_evidence = settings.use_inside_evidence

    class FakeHead:
        """The head keeps its OWN settings reference, and copies three values out
        of it at construction -- all true of the real BoundaryHead. It also owns
        the pair scorer, which holds a SECOND copy of ``use_inside_evidence``."""

        def __init__(self, settings):
            self.settings = settings
            self.hard_negatives_per_positive = settings.hard_negatives_per_positive
            self.minimum_hard_negatives = settings.minimum_hard_negatives
            self.use_inside_evidence = settings.use_inside_evidence
            self.pair_scorer = FakeModel.FakePairScorer(settings)

    def __init__(self, boundary_head):
        self.config = FakeConfig(dict(boundary_head))
        self.boundary_settings = BoundaryHeadSettings(**boundary_head)
        self.boundary_head = FakeModel.FakeHead(self.boundary_settings)


CHECKPOINT = {"enable_relations": True, "enable_records": True}


def test_loss_override_rebuilds_the_settings_object():
    model = FakeModel(CHECKPOINT)
    assert model.boundary_settings.task_loss_weights is None  # precondition

    _apply_boundary_head_overrides(
        model, {**CHECKPOINT, "task_loss_weights": {"events": 2.0}}
    )

    assert model.boundary_settings.task_loss_weights == {"events": 2.0}


def test_override_is_merged_not_replaced():
    """The checkpoint's own boundary_head keys must survive an override that
    does not mention them."""
    model = FakeModel({**CHECKPOINT, "record_loss_weight": 3.0})

    _apply_boundary_head_overrides(model, {"task_loss_weights": {"events": 2.0}})

    assert model.boundary_settings.record_loss_weight == 3.0
    assert model.boundary_settings.enable_records is True
    assert model.boundary_settings.task_loss_weights == {"events": 2.0}


def test_structural_override_raises_rather_than_half_applying():
    """enable_records sizes modules built during __init__. Applying it after the
    fact would leave the settings and the modules disagreeing."""
    model = FakeModel(CHECKPOINT)

    with pytest.raises(SystemExit, match="structural"):
        _apply_boundary_head_overrides(model, {"enable_records": False})


def test_structural_key_matching_the_checkpoint_is_not_a_conflict():
    """Configs restate enable_relations/enable_records to document intent. That
    must stay legal when it agrees with the checkpoint."""
    model = FakeModel(CHECKPOINT)

    _apply_boundary_head_overrides(
        model, {**CHECKPOINT, "task_loss_weights": {"events": 0.5}}
    )

    assert model.boundary_settings.task_loss_weights == {"events": 0.5}


def test_overrides_reach_the_HEAD_settings_not_just_the_model():
    """dfaaa2a rebuilt `model.boundary_settings`, but the head holds its OWN
    reference built in __init__. Every knob the head reads through `self.settings`
    -- the soft_iou/rerank/proposal/count weights, boundary_negative_weight,
    negative_query_ratio, task_loss_weight_scope -- stayed at the checkpoint value,
    so a config setting them produced a treatment arm inert in exactly the way that
    commit was meant to end. Measured before the fix: scope="all" on the model and
    "span" on the head."""
    model = FakeModel(CHECKPOINT)
    assert model.boundary_head.settings.task_loss_weight_scope == "span"

    _apply_boundary_head_overrides(
        model,
        {"task_loss_weight_scope": "all", "rerank_listwise_weight": 0.77,
         "minimum_hard_negatives": 9},
    )

    assert model.boundary_settings.task_loss_weight_scope == "all"
    assert model.boundary_head.settings.task_loss_weight_scope == "all"
    assert model.boundary_head.settings is model.boundary_settings
    assert model.boundary_head.settings.rerank_listwise_weight == 0.77
    # Copied at construction rather than read live, so assigning settings alone
    # would not move it.
    assert model.boundary_head.minimum_hard_negatives == 9


def test_use_inside_evidence_reaches_BOTH_forward_readers():
    """The third copied-at-construction value, and the only one with two readers.

    It adds no parameters, so it is a legitimate warm-start override -- but the
    forward pass reads ``head.use_inside_evidence`` and
    ``head.pair_scorer.use_inside_evidence``, never ``settings``. Measured on
    fastino/gliner2.5-multi-v1 before the fix: config and settings went to False
    while both readers stayed True, i.e. an arm identical to its control.
    """
    model = FakeModel({**CHECKPOINT, "use_inside_evidence": True})

    _apply_boundary_head_overrides(model, {"use_inside_evidence": False})

    assert model.boundary_settings.use_inside_evidence is False
    assert model.boundary_head.use_inside_evidence is False
    assert model.boundary_head.pair_scorer.use_inside_evidence is False


@pytest.mark.parametrize("key", [
    "enable_abstention", "enable_count_head", "enable_span_content",
    "boundary_attention_layers", "candidate_attention_layers",
    "query_attention_layers", "endpoint_difference_features",
    "query_conditioned_inside_weight",
    # Shape-only: a state_dict KEY diff cannot see these, a shape diff can.
    "enable_rotary_endpoints", "record_dim", "record_instance_queries",
    # Named by from_pretrained's own load error.
    "candidate_attention_heads", "pool_boundary_top_k", "pool_size",
])
def test_parameter_changing_flags_are_refused_on_a_warm_start(key):
    """Each of these adds or removes parameter tensors (measured by diffing
    state_dict keys), so none can be applied to an already-built model.

    Unguarded they were worse than an error: the override still landed on
    ``model.config`` and was SAVED, so the run trained the checkpoint's
    architecture while writing a config describing a different one -- and the
    next ``from_pretrained`` died on a state-dict mismatch.
    """
    model = FakeModel(CHECKPOINT)
    current = model.config.boundary_head.get(key)
    flip = (not current) if isinstance(current, bool) else 7

    with pytest.raises(SystemExit, match=key):
        _apply_boundary_head_overrides(model, {key: flip})


# ---------------------------------------------------------------------------
# The record gate's operating point must survive the eval-time filter
# ---------------------------------------------------------------------------

def test_the_record_operating_point_survives_the_eval_time_filter():
    """`evaluate_config` DROPS any boundary_head override outside its allowlist.

    It does it silently -- `bh = {k: v for k, v in bh.items() if k in
    _EVAL_TIME_BOUNDARY_KEYS}` -- so a missing key turns `--record-anchor-threshold`
    into a decorative flag that prints its override and changes nothing, which is the
    same failure this module was written for.

    All THREE keys are load-bearing: the value; `_wins`, without which the record gate
    is the span gate and the value does nothing; and the proposal threshold, because
    `validate_boundary_head` enforces proposal <= anchor and refuses any anchor below
    the 0.2 default.
    """
    from train import _EVAL_TIME_BOUNDARY_KEYS, _STRUCTURAL_BOUNDARY_KEYS

    required = (
        "record_anchor_threshold",
        "record_anchor_threshold_wins",
        "record_anchor_proposal_threshold",
    )
    missing = [k for k in required if k not in _EVAL_TIME_BOUNDARY_KEYS]
    assert not missing, (
        f"{missing} would be filtered out of the boundary overrides in silence, so "
        "--record-anchor-threshold would change nothing"
    )
    # An operating point is not structural: it adds and removes no tensors and is read
    # at decode. If one of these were classed structural, eval would REFUSE instead.
    assert not [k for k in required if k in _STRUCTURAL_BOUNDARY_KEYS]


def test_the_override_reaches_boundary_settings_on_a_real_config():
    """End to end through the real validator, not a hand-built dict."""
    from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head

    shipped = BoundaryHeadSettings(**validate_boundary_head({}))
    assert shipped.record_anchor_threshold_wins is False, "default must not move"

    merged = validate_boundary_head({
        "record_anchor_threshold": 0.1,
        "record_anchor_proposal_threshold": 0.1,
        "record_anchor_threshold_wins": True,
    })
    settings = BoundaryHeadSettings(**merged)
    assert settings.record_anchor_threshold_wins is True
    assert settings.record_anchor_threshold == 0.1


# ---------------------------------------------------------------------------
# Proposal width: derived settings do not follow an assignment to the head
# ---------------------------------------------------------------------------

def test_proposal_width_keys_survive_the_eval_time_filter():
    """`candidate_budget` and the top-k knobs are operating points, not structure.

    They are the binding constraint on long documents: gold coverage at a 4096-token
    window is 8.1% at the shipped 16/16/128 and 18.7% at 128/128/384 with
    `boundary_top_k_alpha` on, measured with ProposalStats.gold_hit_without_injection.
    Filtered out, an override prints and changes nothing.
    """
    from train import _EVAL_TIME_BOUNDARY_KEYS, _STRUCTURAL_BOUNDARY_KEYS

    required = ("candidate_budget", "start_top_k", "end_top_k",
                "boundary_top_k_alpha", "boundary_top_k_max")
    missing = [k for k in required if k not in _EVAL_TIME_BOUNDARY_KEYS]
    assert not missing, f"{missing} would be dropped from boundary overrides in silence"
    assert not [k for k in required if k in _STRUCTURAL_BOUNDARY_KEYS]


def test_resync_reaches_the_proposer_and_a_bare_assignment_does_not():
    """THE POINT OF THE HELPER, and it fails if the helper stops working.

    `ProposalSettings` is built in `__init__` and the proposer re-reads its own frozen
    copy every forward, so assigning `boundary_settings` leaves the proposer where it
    was constructed. That is how a validated, printed override changes nothing.
    """
    import dataclasses

    from gliner2.configuration import BoundaryHeadSettings
    from gliner2.models.boundary.model import (
        ProposalSettings, proposal_settings_from_head, resync_derived_settings,
    )

    class FakeProposer:
        def __init__(self, settings):
            self.settings = settings

    class FakeModel:
        def __init__(self, proposer, head_settings):
            self._p = proposer
            self.boundary_settings = head_settings

        def modules(self):
            return [self, self._p]

    shipped = BoundaryHeadSettings()
    proposer = FakeProposer(proposal_settings_from_head(shipped))
    model = FakeModel(proposer, shipped)
    assert proposer.settings.start_top_k == shipped.start_top_k

    wider = dataclasses.replace(shipped, start_top_k=128, end_top_k=128,
                                candidate_budget=384, boundary_top_k_alpha=0.05)

    # A bare assignment to the head is the trap: the proposer does not move.
    model.boundary_settings = wider
    assert proposer.settings.start_top_k == shipped.start_top_k, (
        "the proposer moved without a resync, so this test no longer guards anything"
    )

    n = resync_derived_settings(model, wider)
    assert n == 1
    assert proposer.settings.start_top_k == 128
    assert proposer.settings.candidate_budget == 384
    assert proposer.settings.boundary_top_k_alpha == 0.05


def test_resync_leaves_the_head_settings_object_alone():
    """BoundaryHeadSettings also has `candidate_budget`, so a hasattr filter would match
    it and raise rebuilding it with ProposalSettings fields. Type check, not duck-typing.
    """
    from gliner2.configuration import BoundaryHeadSettings
    from gliner2.models.boundary.model import resync_derived_settings

    class HoldsHeadSettings:
        def __init__(self):
            self.settings = BoundaryHeadSettings()

    holder = HoldsHeadSettings()

    class M:
        def modules(self):
            return [holder]

    assert resync_derived_settings(M(), BoundaryHeadSettings(start_top_k=64)) == 0
    assert isinstance(holder.settings, BoundaryHeadSettings)


def test_decoding_wider_than_training_is_refused():
    """The invariant the config relies on: train wider than you decode."""
    import pytest as _pytest

    from gliner2.configuration import validate_boundary_head

    ok = validate_boundary_head({"candidate_budget": 384,
                                 "training_candidate_budget": 384})
    assert ok["candidate_budget"] == 384
    with _pytest.raises(ValueError, match="training_candidate_budget"):
        validate_boundary_head({"candidate_budget": 2048,
                                "training_candidate_budget": 384})
