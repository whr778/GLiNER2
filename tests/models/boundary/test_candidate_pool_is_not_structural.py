"""`candidate_pool` selects a forward path; it does not size anything.

It spent its life in `_STRUCTURAL_BOUNDARY_KEYS`, which refuses a boundary_head
override on the `pretrained` path because "the checkpoint's modules are already
built". For `candidate_pool` that premise is false, and the cost was a real
experiment: on 2026-09-17 a three-arm A/B had BOTH treatment arms refused at
startup, leaving a control that re-measured the baseline.

These tests pin the narrowing in both directions -- the key is free, and the
guard still refuses a key that genuinely changes tensors.
"""

from dataclasses import replace

import pytest
import torch

from gliner2.configuration import BoundaryHeadSettings
from gliner2.models.boundary.model import BoundaryHead


def _head(pool):
    torch.manual_seed(0)
    return BoundaryHead(
        hidden_size=32,
        settings=BoundaryHeadSettings(
            boundary_dim=16, pair_dim=16, candidate_pool=pool,
            enable_records=True, enable_relations=True,
        ),
    )


def test_candidate_pool_changes_no_tensor():
    """The measurement the guard should have been built on.

    Not a hard-coded key list: it builds the head both ways and diffs. If the flag
    ever DOES start sizing a module, this fails and the guard entry must come back.
    """
    per_query = {k: tuple(v.shape) for k, v in _head("per_query").state_dict().items()}
    shared = {k: tuple(v.shape) for k, v in _head("shared").state_dict().items()}

    assert set(per_query) == set(shared), (
        "candidate_pool added or removed tensors; it IS structural after all -- "
        f"only per_query: {sorted(set(per_query) - set(shared))[:5]}, "
        f"only shared: {sorted(set(shared) - set(per_query))[:5]}"
    )
    reshaped = {k: (per_query[k], shared[k]) for k in per_query if per_query[k] != shared[k]}
    assert not reshaped, f"candidate_pool resized tensors: {reshaped}"


def test_shared_pool_modules_exist_under_per_query():
    """The reason the flag is free: these are built unconditionally (model.py:246).

    They receive no gradient under per_query, which is a HANDICAP for a warm-started
    shared arm -- it starts from an untrained pool -- but it is not a load failure.
    """
    names = [n for n, _ in _head("per_query").named_parameters() if "shared_pool" in n]
    assert names, "shared_pool_* parameters must exist even under per_query"


def test_the_guard_no_longer_refuses_candidate_pool():
    from tools.train.train import _STRUCTURAL_BOUNDARY_KEYS

    assert "candidate_pool" not in _STRUCTURAL_BOUNDARY_KEYS


def test_the_guard_still_refuses_a_genuinely_structural_key():
    """The narrowing must be a narrowing, not a hole.

    `pool_size` and `pool_boundary_top_k` size this SAME module and were left
    guarded because they were never measured; `enable_span_content` is in the
    2026-09-05 measured group.
    """
    from tools.train.train import _STRUCTURAL_BOUNDARY_KEYS

    for key in ("enable_span_content", "enable_records", "pool_size",
                "pool_boundary_top_k", "record_dim"):
        assert key in _STRUCTURAL_BOUNDARY_KEYS, f"{key} must stay guarded"


def test_override_reaches_the_head_not_just_the_config():
    """A structural-key removal is worthless if the override lands nowhere.

    `from_pretrained` builds `boundary_settings` from the CHECKPOINT's config, so a
    plain setattr on `model.config` is dropped. The HEAD holds its own reference,
    and rebuilding only the model's left every knob at its checkpoint value once
    before. Both must move. Uses a REAL head, because a stub passes by having no
    attributes to leave stale.
    """
    from tools.train.train import _apply_boundary_head_overrides

    head = _head("per_query")

    class FakeConfig:
        boundary_head = {"candidate_pool": "per_query",
                         "enable_records": True, "enable_relations": True}

    class FakeModel:
        config = FakeConfig()

    model = FakeModel()
    model.boundary_head = head
    _apply_boundary_head_overrides(model, {"candidate_pool": "shared"})

    assert model.boundary_settings.candidate_pool == "shared"
    assert model.boundary_head.settings.candidate_pool == "shared", \
        "the head kept its checkpoint value -- the arm would be inert"
    assert model.config.boundary_head["candidate_pool"] == "shared"


def _shared_pool_grad_norm(head, golden_batch):
    """Backward one real loss and return the gradient mass on the shared pool."""
    head.train()
    out = head(
        golden_batch["token_states"], golden_batch["text_mask"],
        golden_batch["query_states"], golden_batch["query_mask"],
        targets=golden_batch["targets"],
    )
    head.zero_grad(set_to_none=True)
    out.total_loss.backward()
    total = sum(
        float(p.grad.detach().float().norm() ** 2)
        for n, p in head.named_parameters()
        if "shared_pool" in n and p.grad is not None
    )
    return total ** 0.5


def test_shared_pool_carries_gradient_ONLY_under_shared(golden_batch):
    """The premise the A/B gate rests on -- if this is false the gate is decoration.

    Presence of the tensors separates nothing (they are in every checkpoint). What
    separates the arms is whether the forward reaches them, and gradient is the only
    read of that which a config typo cannot fake.
    """
    settings = golden_batch["head"].settings

    per_query = BoundaryHead(48, replace(settings, candidate_pool="per_query"), query_dim=48)
    shared = BoundaryHead(48, replace(settings, candidate_pool="shared"), query_dim=48)

    control = _shared_pool_grad_norm(per_query, golden_batch)
    treatment = _shared_pool_grad_norm(shared, golden_batch)

    assert control == 0.0, (
        f"per_query put {control:.3e} of gradient on the shared pool -- the control "
        "arm is not a control"
    )
    assert treatment > 0.0, (
        "shared put NO gradient on the shared pool -- the treatment arm is inert "
        "and the A/B would report a null that means nothing"
    )
