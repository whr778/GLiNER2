"""The eval accumulation window must be a pure latency optimisation.

Decoding blocks on a device sync -- `keep.nonzero(...)` inside `_group_scored_candidates`
returns a data-dependent shape, so the CPU waits for the GPU to drain the queued forward.
Profiled at 32% of eval wall time in exactly one call per batch. Queueing N forwards
before consuming any means N batches share one stall.

This is HF `Trainer`'s DEFAULT eval behaviour; their `eval_accumulation_steps` exists to
make it SLOWER under memory pressure. We had no way to do it at all.

Unlike raising batch_size it adds no padding, which matters here: documents run
466-6,000 characters, every batch pads to its longest member, and batching alone was
measured slower (5.1 docs/s at batch 1 vs 2.0 at batch 32).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_forward_batch_exists_and_is_separate_from_the_decode():
    """The split is the whole mechanism; without it there is nothing to queue."""
    from gliner2.models.boundary.engine import BoundaryExtractor as E
    assert hasattr(E, "_forward_batch")
    assert hasattr(E, "_extract_from_batch")
    import inspect
    assert "forward" in inspect.signature(E._extract_from_batch).parameters, (
        "_extract_from_batch must accept a precomputed forward, or the window cannot "
        "hand it one")


def test_forward_batch_does_not_synchronise():
    """A .cpu()/.tolist()/.item()/nonzero in the forward half makes the window useless.

    Checked on the source rather than at runtime: a sync here would not fail a test, it
    would silently give back the stall the window exists to remove.
    """
    import inspect
    from gliner2.models.boundary.engine import BoundaryExtractor as E
    src = inspect.getsource(E._forward_batch)
    body = src.split('"""', 2)[-1]          # ignore the docstring, which names them
    for bad in (".cpu()", ".tolist()", ".item()", "nonzero"):
        assert bad not in body, f"_forward_batch synchronises on {bad}"


def test_default_window_is_one_so_behaviour_is_unchanged():
    """Opt-in until measured. A silent default change to a hot path is not a speedup."""
    src = (ROOT / "gliner2" / "inference" / "runtime.py").read_text(encoding="utf-8")
    assert '_eval_accumulation_steps", 1)' in src, (
        "the accumulation window must default to 1 (previous behaviour)")


def test_window_reaches_the_decode_with_a_precomputed_forward():
    """With a window > 1 the decode must be HANDED the queued forward. If it recomputed
    instead, the window would double the work while appearing to succeed."""
    src = (ROOT / "gliner2" / "inference" / "runtime.py").read_text(encoding="utf-8")
    assert "forward=fwd" in src
