"""Shared pytest fixtures for the GLiNER2 test suite."""

from __future__ import annotations

import pytest
import torch

from tests.fixtures.tiny_encoder import build_tiny_encoder_config
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Seed RNGs before every test for reproducibility, and reset the global
    torch determinism state afterwards. A trainer run with deterministic=True
    flips use_deterministic_algorithms process-wide; without this reset it would
    leak into later tests (e.g. forcing a less memory-optimized gather backward)."""
    torch.manual_seed(0)
    yield
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def tiny_tokenizer():
    return build_tiny_tokenizer()


@pytest.fixture
def tiny_encoder_config(tiny_tokenizer):
    return build_tiny_encoder_config(vocab_size=len(tiny_tokenizer))


@pytest.fixture
def tiny_span_model():
    return build_tiny_span_model()
