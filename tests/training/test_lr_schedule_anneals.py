"""A decaying schedule must actually reach ~0 by the final step.

`cosine_restarts` computes ``cos(pi * ((num_cycles * progress) % 1.0))``. With
``num_cycles=0.5`` the argument only ever reaches 0.5, so it NEVER WRAPS: the schedule
performs no restarts and ends at 50% of base LR, with the final epoch still taking
full-size steps. That was the default until 2026-09-22, and all 158 configs in this repo
name `cosine_restarts` while none sets `num_cycles` -- so every model this project has
trained finished without annealing, and nothing reported it because the curve was never
printed.

These tests fail on the old default.
"""

from __future__ import annotations

import math

import pytest
import torch

from gliner2.training.trainer import TrainingConfig, get_scheduler

TOTAL, WARMUP = 63_275, 3_163
DECAYING = ("linear", "cosine", "cosine_restarts")


def _lambda(scheduler_type, num_cycles):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    return get_scheduler(opt, scheduler_type, TOTAL, WARMUP, num_cycles).lr_lambdas[0]


def test_default_num_cycles_anneals_to_zero():
    """The shipped default must anneal. At 0.5 the tail sits at 0.500 and this fails."""
    fn = _lambda("cosine_restarts", TrainingConfig.__dataclass_fields__["num_cycles"].default)
    assert fn(TOTAL - 1) < 0.01


@pytest.mark.parametrize("scheduler_type", DECAYING)
def test_every_decaying_schedule_reaches_zero_at_the_last_step(scheduler_type):
    fn = _lambda(scheduler_type, TrainingConfig.__dataclass_fields__["num_cycles"].default)
    assert fn(TOTAL - 1) < 0.01, f"{scheduler_type} ends at {fn(TOTAL - 1):.3f} of base LR"


def test_default_cosine_restarts_matches_plain_cosine_on_every_real_step():
    """With num_cycles=1.0 the two are the same schedule wherever training actually runs.
    They differ only at step == TOTAL, which never executes (steps are 0..TOTAL-1)."""
    a = _lambda("cosine_restarts", 1.0)
    b = _lambda("cosine", 1.0)
    for step in (WARMUP, TOTAL // 2, int(TOTAL * 0.9), TOTAL - 2, TOTAL - 1):
        assert math.isclose(a(step), b(step), abs_tol=1e-9), f"diverge at {step}"


def test_warmup_is_a_ramp_that_reaches_full_lr():
    fn = _lambda("cosine_restarts", 1.0)
    assert fn(0) == pytest.approx(0.0, abs=1e-9)
    assert fn(WARMUP // 2) == pytest.approx(0.5, abs=0.01)
    assert fn(WARMUP) == pytest.approx(1.0, abs=1e-9)


def test_half_cycle_is_still_reachable_and_still_degenerate():
    """Not a recommendation -- a record. Anyone who sets 0.5 deliberately gets a
    non-restarting half-cosine ending at half LR, and should see it stated here."""
    fn = _lambda("cosine_restarts", 0.5)
    assert fn(TOTAL - 1) == pytest.approx(0.5, abs=0.01)
