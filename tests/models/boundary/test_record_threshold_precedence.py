"""The record gate WAS the span gate, and `record_anchor_threshold` was unreachable.

`_decode_records` took `float(threshold) if threshold is not None`, and
`_extract_from_batch(threshold: float)` is a required positional that both call sites
fill with the SPAN threshold -- so the span value always won and the record knob did
nothing on any eval path. Traced 2026-09-29 on a real checkpoint: the setting moved
0.5 -> 0.1 -> 0.01 while the decode used 0.3 each time and the output never changed.
A sweep over that axis reads perfectly flat, which is how it hid.

`record_anchor_threshold_wins` makes it reachable, OFF by default because every
checkpoint on the Hub carries an explicit `record_anchor_threshold: 0.5` and honouring
it unconditionally would move every existing model off the span threshold -- on an axis
measured to swing event_argument strict F1 2.8x.
"""

from __future__ import annotations

import dataclasses

import torch

from gliner2.configuration import BoundaryHeadSettings
from gliner2.models.boundary.engine import resolve_record_threshold

from tests.models.boundary.test_record_head_pipeline import _build_tiny_records_model


def _settings(**kw):
    return dataclasses.replace(BoundaryHeadSettings(), **kw)


def test_the_span_threshold_governs_by_default():
    """Flag off: the decode is bit-identical to what every checkpoint was measured with."""
    off = _settings(record_anchor_threshold=0.1)
    assert off.record_anchor_threshold_wins is False
    for span in (0.5, 0.3, 0.05):
        assert resolve_record_threshold(off, span) == span


def test_the_setting_governs_once_the_flag_is_set():
    on = _settings(record_anchor_threshold=0.1, record_anchor_threshold_wins=True)
    for span in (0.5, 0.3, 0.05):
        assert resolve_record_threshold(on, span) == 0.1


def test_a_missing_span_threshold_falls_back_to_the_setting_either_way():
    """`threshold=None` is the one case the setting always governed."""
    for wins in (False, True):
        s = _settings(record_anchor_threshold=0.2, record_anchor_threshold_wins=wins)
        assert resolve_record_threshold(s, None) == 0.2


def test_an_older_settings_object_without_the_flag_keeps_the_span_threshold():
    """A checkpoint predating the flag must not change behaviour on load."""

    class Old:
        record_anchor_threshold = 0.5

    assert resolve_record_threshold(Old(), 0.3) == 0.3


def test_the_decode_actually_consults_the_resolver(monkeypatch):
    """THE ANTI-ORPHAN GATE.

    Every test above passes even if `_decode_records` stops calling the resolver and
    goes back to reading `threshold` directly -- which is exactly the defect being
    fixed. This one runs a real extraction on a real model and fails if the production
    decode does not route through it.
    """
    model = _build_tiny_records_model()
    calls = []
    real = resolve_record_threshold

    def spy(settings, threshold):
        calls.append(threshold)
        return real(settings, threshold)

    monkeypatch.setattr("gliner2.models.boundary.engine.resolve_record_threshold", spy)
    with torch.no_grad():
        model.extract_json(
            "the earthquake struck the city",
            {"quake": {"place": "string", "size": "string"}},
            threshold=0.3,
        )
    assert calls, (
        "the record decode never called resolve_record_threshold -- the knob is "
        "unreachable again and a sweep over it will read flat"
    )
    assert calls[0] == 0.3, f"the span threshold did not reach the resolver: {calls[0]}"
