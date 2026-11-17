"""The per-language blind test must accept and apply eval-time boundary_head overrides.

`eval.py` on a config with both `eval_by_language` and boundary-head eval keys (eb18) died
with `TypeError: _blind_test_by_language() got an unexpected keyword argument
'boundary_overrides'` (2026-10-02), losing both eb18 passes of a paid re-score.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
import train as T  # noqa: E402
import gliner2.training.eval_metrics as E  # noqa: E402
import gliner2.training.metrics as M  # noqa: E402


def test_by_language_path_receives_boundary_overrides(monkeypatch):
    seen = {}
    monkeypatch.setattr(E, "load_with_overrides", lambda best, ov=None, map_location=None: seen.setdefault("ov", ov))
    monkeypatch.setattr(M, "compute_metrics", lambda *a, **k: {})
    monkeypatch.setattr(T, "_annotate_languages", lambda recs: [r.setdefault("_lang", "eng") for r in recs])
    recs = [{"input": f"doc {i}", "output": {}} for i in range(30)]
    T._run_blind_test("ckpt", recs, 2, 0.5, True,
                      {"global_decode": True, "boundary_overrides": {"start_top_k": 16}})
    assert seen["ov"] == {"start_top_k": 16}


def test_by_language_path_runs_the_full_menu_pass(monkeypatch):
    """`eval.py --full-menu` on an `eval_by_language` config died the same way with
    'full_menu' (2026-10-15); every language and the combined set must get the second pass."""
    calls = []

    def fake(model, ds, **k):
        calls.append(bool(k.get("full_menu")))
        return {"eval_x_strict_micro_f1": 0.5}

    class Cfg:
        default_schema = {"events": {"Attack": []}}

    monkeypatch.setattr(E, "load_with_overrides", lambda best, ov=None, map_location=None: type("Mdl", (), {"config": Cfg})())
    monkeypatch.setattr(M, "compute_metrics", fake)
    monkeypatch.setattr(E, "compute_metrics", fake)
    monkeypatch.setattr(T, "_annotate_languages", lambda recs: [r.setdefault("_lang", "eng") for r in recs])
    recs = [{"input": f"doc {i}", "output": {}} for i in range(30)]
    m = T._run_blind_test("ckpt", recs, 2, 0.5, True, {"full_menu": True, "menu_negatives": 20})
    assert calls == [False, True, False, True]  # eng, then all: gold pass then full-menu pass
    assert m["eval_fullmenu_x_strict_micro_f1"] == 0.5
    assert m["by_language"]["eng"]["eval_fullmenu_x_strict_micro_f1"] == 0.5
    assert m["fullmenu_sizes"]["events"] == 1


def test_default_schema_mode_dataset_train_and_union_base():
    import pytest
    from gliner2.training.config import TrainingConfig

    with pytest.raises(ValueError, match="default_schema_mode"):
        TrainingConfig(default_schema_mode="bogus")

    base_schema = {
        "events": {"Attack": ["Attacker"], "Hire": ["Employee"]},
        "relations": ["works_at"],
        "open_vocab": ["entities"],
    }
    train_recs = [
        {
            "output": {
                "entities": {"Weapon": ["missile"]},
                "events": [{"event_type": "Deploy", "arguments": [{"role": "Subject"}]}],
            }
        }
    ]
    eval_recs = [
        {
            "output": {
                "events": [{"event_type": "Deploy", "arguments": [{"role": "Subject"}, {"role": "Location"}]}],
            }
        }
    ]
    partial = {"cmnee_ner_off": ["entities"]}

    ds_only = T.resolve_default_schema(
        "dataset", train_recs=train_recs, eval_recs=eval_recs,
        partial_annotation=partial, base_schema=base_schema,
    )
    assert ds_only == {"events": {"Deploy": ["Location", "Subject"]}}

    tr_only = T.resolve_default_schema(
        "train", train_recs=train_recs, eval_recs=eval_recs,
        partial_annotation=partial, base_schema=base_schema,
    )
    assert tr_only == {"entities": ["Weapon"], "events": {"Deploy": ["Subject"]}}

    unioned = T.resolve_default_schema(
        "union_base", train_recs=train_recs, eval_recs=eval_recs,
        partial_annotation=partial, base_schema=base_schema,
    )
    assert unioned["open_vocab"] == ["entities"]
    assert "entities" not in unioned
    assert set(unioned["events"]) == {"Attack", "Hire", "Deploy"}
    assert unioned["relations"] == ["works_at"]


def test_score_model_schema_includes_empty_gold_records_and_auto_gates_open_vocab():
    from gliner2.training.trainer import ExtractorDataset

    class ClosedCfg:
        default_schema = {"events": {"Attack": ["Attacker"], "Deploy": ["Subject"]}}
        label_map = None
        label_style = None

    class OpenVocabCfg:
        default_schema = {"events": {"Attack": ["Attacker"]}, "open_vocab": ["entities"]}
        label_map = None
        label_style = None

    assert E.should_score_model_schema("auto", ClosedCfg) is True
    assert E.should_score_model_schema("auto", OpenVocabCfg) is False
    assert E.should_score_model_schema(True, OpenVocabCfg) is True
    assert E.should_score_model_schema(False, ClosedCfg) is False

    seen_schemas = []

    class FakeModel:
        config = ClosedCfg

        def batch_extract(self, texts, schemas, **kw):
            seen_schemas.extend(schemas)
            return [
                {"event_extraction": {"Attack": [{"triggers": ["hit"], "Attacker": ["Alice"]}]}},
                {"event_extraction": {"Deploy": [{"triggers": ["moved"], "Subject": ["Army"]}]}},
            ]

    records = [
        {"input": "Alice hit Bob.", "output": {"events": [{"event_type": "Attack", "triggers": ["hit"], "arguments": [{"role": "Attacker", "entity": "Alice"}]}]}},
        {"input": "Sunny weather.", "output": {"events": []}},
    ]
    ms = E.score_model_schema(
        FakeModel(), ExtractorDataset(records, shuffle=False, validate=False),
        threshold=0.5, per_head_thresholds={"event": 0.5},
    )
    assert len(seen_schemas) == 2  # empty-gold doc was NOT skipped
    assert ms["schema_counts"] == {"events": 2}
    assert ms["threshold"] == 0.5
    assert ms["per_head_thresholds"] == {"event": 0.5}
    # 1 TP on doc 0 + 1 FP on empty doc 1 -> event_type precision = 0.5, recall = 1.0
    assert ms["eval_event_type_strict_micro_precision"] == 0.5
    assert ms["eval_event_type_strict_micro_recall"] == 1.0
