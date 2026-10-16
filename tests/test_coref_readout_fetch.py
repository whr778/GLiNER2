"""The coref readout pulls its val files itself: a box holds only what its config names.

corefB trained these corpora `train_only`, so no val file ever reached the box and the readout's
first open() raised -- silently, because the runner only published the log beside the json.
"""
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))

import coref_readout  # noqa: E402


def test_readout_fetches_a_missing_val_file_before_opening_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    asked = []

    def fake_fetch(path):
        asked.append(path)
        Path(path).parent.mkdir(exist_ok=True)
        Path(path).write_text("", encoding="utf-8")

    monkeypatch.setattr(coref_readout, "_fetch_if_missing", fake_fetch)
    res = coref_readout.readout(model=SimpleNamespace(mention_link=None), corpora=["gap_coref"], limit=5, max_words=None)
    assert asked == ["data/gap_coref.val.jsonl"] and res["gap_coref"]["docs"] == 0
