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
