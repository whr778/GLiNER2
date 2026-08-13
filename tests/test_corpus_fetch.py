"""``_split_files`` pulls a missing corpus split from the Hub when the registry says where.

The registry field is ``hf_jsonl`` (GLiNER2 JSONL splits), deliberately NOT
``hf_id``, which is upstream attribution and for most corpora points at raw
parquet or BIO tags.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))
from model_card import load_registry  # noqa: E402
from train import _split_files  # noqa: E402


@pytest.fixture
def spy(monkeypatch):
    """Record hf_hub_download calls instead of hitting the network."""
    calls = []

    def fake(repo_id, filename, repo_type, local_dir):
        calls.append((repo_id, filename, repo_type, local_dir))
        Path(local_dir, filename).write_text("", encoding="utf-8")
        return str(Path(local_dir, filename))

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    return calls


def test_present_file_is_not_fetched(spy, tmp_path):
    (tmp_path / "synthetic_haiku45_5k.val.jsonl").write_text("{}\n", encoding="utf-8")

    _split_files([str(tmp_path / "synthetic_haiku45_5k")], "val")

    assert spy == []


def test_missing_file_with_hf_jsonl_is_fetched(spy, tmp_path):
    out = _split_files([str(tmp_path / "synthetic_haiku45_5k")], "val")

    assert len(spy) == 1
    repo_id, filename, repo_type, local_dir = spy[0]
    assert repo_id == "whr778/synthetic_haiku45_5k"
    assert filename == "synthetic_haiku45_5k.val.jsonl"
    assert repo_type == "dataset"
    assert Path(local_dir) == tmp_path
    assert out == [str(tmp_path / "synthetic_haiku45_5k.val.jsonl")]


def test_missing_file_without_hf_jsonl_is_left_alone(spy, tmp_path):
    """Corpora with no hf_jsonl keep the old behaviour: the path is returned as-is
    and the downstream reader is what complains.

    biomed_ner is registered but has no hf_jsonl, which is the case for every
    corpus except the synthetic ones -- guard picked for that, so adding hf_jsonl
    to a corpus elsewhere cannot silently make this test vacuous.
    """
    assert "hf_jsonl" not in (load_registry()["datasets"]["biomed_ner"] or {})

    out = _split_files([str(tmp_path / "biomed_ner")], "val")

    assert spy == []
    assert out == [str(tmp_path / "biomed_ner.val.jsonl")]


def test_unknown_corpus_is_left_alone(spy, tmp_path):
    _split_files([str(tmp_path / "not_a_registered_corpus")], "train")

    assert spy == []
