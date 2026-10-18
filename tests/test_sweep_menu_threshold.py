"""``sweep_menu_threshold.py --val-only`` picks on val and never reads a test split; ``--device`` moves the model."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))
import sweep_menu_threshold as S  # noqa: E402


class _Model:
    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self

    def parameters(self):
        import torch
        yield torch.zeros(1)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    import gliner2
    import gliner2.training.eval_metrics as em
    reads, model = [], _Model()
    cfg = tmp_path / "c.yaml"
    cfg.write_text("eval: {}\n", encoding="utf-8")
    monkeypatch.setattr(gliner2.AutoExtractor, "from_pretrained", staticmethod(lambda _: model))
    monkeypatch.setattr(S.T, "_parse_eval_settings", lambda *a, **k: dict(
        chunk_size=None, chunk_overlap=0, global_decode=False, global_decode_config=None, batch_size=1))
    monkeypatch.setattr(S.T, "menu_split", lambda cfg, path, mode, split: reads.append((mode, split)) or ([], [], {}))
    monkeypatch.setattr(em, "sweep_thresholds", lambda *a, **k: (0.4, {}, {0.4: {}}))
    monkeypatch.setattr(em, "compute_metrics", lambda *a, **k: {})
    return reads, model, cfg, tmp_path / "out.json"


def test_val_only_reads_no_test_split(stubbed):
    reads, model, cfg, out = stubbed
    S.main(["--config", str(cfg), "--checkpoint", "x", "--menu", "app:news55", "--val-only",
            "--device", "cpu", "--out", str(out)])
    assert reads == [("app:news55", "val")]
    assert json.loads(out.read_text(encoding="utf-8"))["test"] == {}
    assert json.loads(out.read_text(encoding="utf-8"))["chosen_threshold"] == 0.4
    assert model.moved_to == "cpu"


def test_without_val_only_the_test_is_scored(stubbed):
    reads, model, cfg, out = stubbed
    S.main(["--config", str(cfg), "--checkpoint", "x", "--menu", "app:news55", "--out", str(out)])
    assert reads == [("app:news55", "val"), ("app:news55", "test"), ("gold", "test")]
    assert model.moved_to is None
