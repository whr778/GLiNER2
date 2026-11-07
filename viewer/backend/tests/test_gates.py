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


def test_an_absent_merge_key_is_removed_after_the_request():
    class Model:
        def __init__(self):
            self.config = SimpleNamespace(boundary_head={"record_coref_link": True, "record_coref_link_threshold": 0.5})
            self.boundary_head = None

        def modules(self):
            return []

    model = Model()
    restore = app._apply_gates(model, {
        "record_merge_coreferent": "link",
        "record_coref_link_threshold": 0.7,
    })
    assert model.config.boundary_head["record_merge_coreferent"] == "link"
    assert model.config.boundary_head["record_coref_link_threshold"] == 0.7
    app._apply_gates(model, restore)
    assert "record_merge_coreferent" not in model.config.boundary_head
    assert model.config.boundary_head["record_coref_link_threshold"] == 0.5
    assert model.config.boundary_head["record_coref_link"] is True


def test_merge_overrides_stay_off_until_set():
    assert "record_merge_coreferent" not in app._gate_overrides(app.Options())
    assert "record_coref_link_threshold" not in app._gate_overrides(app.Options())
    ov = app._gate_overrides(app.Options(merge_coreferent="link", coref_threshold=0.7))
    assert ov["record_merge_coreferent"] == "link"
    assert ov["record_coref_link_threshold"] == 0.7


def test_candidate_mentions_keeps_the_entity_and_the_pronoun():
    from gliner2.inference.entity_coref import candidate_mentions
    mentions = candidate_mentions(
        "Ann said she left.", {"Person": [{"text": "Ann", "start": 0, "end": 3}]})
    assert [m["text"] for m in mentions] == ["Ann", "she"]


def test_entity_coref_refuses_a_model_without_the_link():
    from fastapi import HTTPException
    try:
        app._attach_entity_coref(SimpleNamespace(), "Ann said she left.", {"entities": {}}, 0.5)
    except HTTPException as exc:
        assert exc.status_code == 422
    else:
        raise AssertionError("expected 422")


def test_span_model_is_left_alone():
    model = SimpleNamespace(config=SimpleNamespace(boundary_head=None))
    assert app._apply_gates(model, {"record_field_threshold": 0.05}) == {}


def test_extract_forwards_per_head_thresholds_and_head_spans(monkeypatch):
    import torch

    captured = {}

    class DummyModel:
        def __init__(self):
            self.config = SimpleNamespace(boundary_head={}, label_map=None, label_style=None)
            self._param = torch.zeros(1)

        def parameters(self):
            return iter([self._param])

        def batch_extract_long(self, texts, schema, **kwargs):
            captured.update(kwargs)
            return [{"entities": {"Person": [{"text": "hackers", "start": 4, "end": 11}]}}]

    monkeypatch.setattr(app, "get_model", lambda model_id, device=None: DummyModel())
    req = app.ExtractRequest(
        text="The hackers attacked.",
        schema={"entities": ["Person"]},
        options=app.Options(
            threshold=0.9,
            per_head_thresholds={"entity": 0.9, "event": 0.7, "relation": 0.3},
            head_spans=True,
        ),
    )
    res = app.extract(req)
    assert res["result"]["entities"]["Person"][0]["text"] == "hackers"
    assert captured["threshold"] == 0.9
    assert captured["per_head_thresholds"] == {"entity": 0.9, "event": 0.7, "relation": 0.3}
    assert captured["head_spans"] is True

