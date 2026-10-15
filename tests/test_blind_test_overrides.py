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
