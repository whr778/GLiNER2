"""TODO 22: record_field_threshold is a real ARGUMENT gate behind record_field_threshold_wins."""
import sys
from pathlib import Path

from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head
from gliner2.models.boundary.engine import resolve_record_field_threshold

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "train"))
import train as T  # noqa: E402


def _settings(**kw):
    return BoundaryHeadSettings(**validate_boundary_head(kw))


def test_off_by_default_fields_follow_the_record_gate():
    s = _settings(record_field_threshold=0.05)
    assert s.record_field_threshold_wins is False
    assert resolve_record_field_threshold(s, 0.3) == 0.3     # measured checkpoints decode unchanged


def test_on_makes_the_field_threshold_real():
    s = _settings(record_field_threshold=0.05, record_field_threshold_wins=True)
    assert resolve_record_field_threshold(s, 0.3) == 0.05


def test_field_gate_is_an_eval_time_override_not_structural():
    for key in ("record_field_threshold", "record_field_threshold_wins"):
        assert key in T._EVAL_TIME_BOUNDARY_KEYS
        assert key not in T._STRUCTURAL_BOUNDARY_KEYS
