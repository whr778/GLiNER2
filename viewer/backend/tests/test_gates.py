"""Per-request record/argument gates (TODO 19/22): mapping, restore, span models untouched."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402
import gliner2.training.eval_metrics as E  # noqa: E402


def test_unset_options_produce_no_overrides():
    assert app._gate_overrides(app.Options()) == {}


def test_gates_map_to_boundary_keys_with_their_wins_flags():
    ov = app._gate_overrides(app.Options(record_threshold=0.1, argument_threshold=0.05))
    assert ov == {"record_anchor_threshold": 0.1, "record_anchor_proposal_threshold": 0.1,
                  "record_anchor_threshold_wins": True,
                  "record_field_threshold": 0.05, "record_field_threshold_wins": True}


def test_apply_returns_the_prior_values_so_the_shared_model_is_restored(monkeypatch):
    model = SimpleNamespace(config=SimpleNamespace(boundary_head={
        "record_field_threshold": 0.5, "record_field_threshold_wins": False}))
    monkeypatch.setattr(E, "apply_boundary_overrides", lambda m, ov: m.config.boundary_head.update(ov))
    restore = app._apply_gates(model, {"record_field_threshold": 0.05, "record_field_threshold_wins": True})
    assert model.config.boundary_head["record_field_threshold"] == 0.05
    app._apply_gates(model, restore)
    assert model.config.boundary_head == {"record_field_threshold": 0.5, "record_field_threshold_wins": False}


def test_span_model_is_left_alone():
    model = SimpleNamespace(config=SimpleNamespace(boundary_head=None))
    assert app._apply_gates(model, {"record_field_threshold": 0.05}) == {}
