"""``_split_files`` pulls a missing corpus split from the Hub when the registry says where.

The registry field is ``hf_jsonl`` (GLiNER2 JSONL splits), deliberately NOT
``hf_id``, which is upstream attribution and for most corpora points at raw
parquet or BIO tags.
"""

import sys
from pathlib import Path

import httpx

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))
from model_card import load_registry  # noqa: E402
from train import _split_files  # noqa: E402


@pytest.fixture
def data_dir(tmp_path):
    """A corpus directory literally named ``data``.

    ``_fetch_if_missing`` resolves by DIRECTORY since `f3b6909`: anything outside
    ``data/`` is treated as a slice and is deliberately NOT fetched by basename. A bare
    ``tmp_path`` therefore exercises the refusal branch, not the fetch branch, which left
    these tests asserting against a path they never reached.
    """
    d = tmp_path / "data"
    d.mkdir()
    return d


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


def test_present_file_is_not_fetched(spy, data_dir):
    (data_dir / "synthetic_haiku45_5k.val.jsonl").write_text("{}\n", encoding="utf-8")

    _split_files([str(data_dir / "synthetic_haiku45_5k")], "val")

    assert spy == []


def test_missing_file_with_hf_jsonl_is_fetched(spy, data_dir):
    out = _split_files([str(data_dir / "synthetic_haiku45_5k")], "val")

    assert len(spy) == 1
    repo_id, filename, repo_type, local_dir = spy[0]
    assert repo_id == "whr778/synthetic_haiku45_5k"
    assert filename == "synthetic_haiku45_5k.val.jsonl"
    assert repo_type == "dataset"
    assert Path(local_dir) == data_dir
    assert out == [str(data_dir / "synthetic_haiku45_5k.val.jsonl")]


def test_missing_file_without_hf_jsonl_is_left_alone(spy, data_dir):
    """Corpora with no hf_jsonl keep the old behaviour: the path is returned as-is
    and the downstream reader is what complains.

    The corpus is chosen from the registry at run time rather than hardcoded. This test
    named `biomed_ner` on the premise that only synthetic corpora carry `hf_jsonl`; that
    stopped being true when 60 corpora were hosted on the Hub (`51e0a28`) and the test
    broke on its own guard. Selecting dynamically keeps the guard's intent -- the test
    still fails loudly rather than going vacuous -- without pinning it to a corpus whose
    hosting status is expected to change.
    """
    registry = load_registry()["datasets"]
    name = next((k for k, v in sorted(registry.items()) if v and "hf_jsonl" not in v), None)
    assert name, "every registered corpus now has hf_jsonl; this path is unreachable"

    out = _split_files([str(data_dir / name)], "val")

    assert spy == []
    assert out == [str(data_dir / f"{name}.val.jsonl")]


def test_unknown_corpus_is_left_alone(spy, data_dir):
    _split_files([str(data_dir / "not_a_registered_corpus")], "train")

    assert spy == []


def test_corpus_with_no_such_split_is_dropped_and_named(monkeypatch, capsys, data_dir):
    """DuEE's mirror ships train + validation only, so ``data/duee.test.jsonl`` is a
    404 from a repo that exists. That made ``--split test`` fatal for every config
    listing duee -- the whole 137k lineage -- and killed a Lambda box mid-run.

    The corpus is dropped and NAMED. Silence is what the `_event_split` docstring
    already forbids: a blind test scoring fewer corpora than it claims.
    """
    from huggingface_hub.errors import RemoteEntryNotFoundError

    def refuse(repo_id, filename, repo_type, local_dir):
        # RemoteEntryNotFoundError is an HfHubHTTPError and needs the real response.
        raise RemoteEntryNotFoundError(f"no {filename} in {repo_id}",
                                       response=httpx.Response(
                                           404, request=httpx.Request("GET", "https://hf.co")))

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)

    out = _split_files([str(data_dir / "duee")], "test")

    assert out == []
    assert "has no test split" in capsys.readouterr().out


def test_a_merely_absent_file_still_reaches_the_reader(spy, data_dir):
    """Only a POSITIVE 'no such entry' from the Hub drops a corpus. A file that is
    simply not there yet keeps the old contract, so a broken mount or a failed
    download surfaces downstream instead of silently shrinking the split."""
    out = _split_files([str(data_dir / "not_a_registered_corpus")], "test")

    assert out == [str(data_dir / "not_a_registered_corpus.test.jsonl")]
