"""dedupe_splits: precedence test > val > train, exact keys and --near-dup containment."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "data"))
import dedupe_splits as D  # noqa: E402

STORY = " ".join(f"word{i}" for i in range(40))


def _write(path, texts):
    path.write_text("".join(json.dumps({"input": t, "output": {}}) + "\n" for t in texts), encoding="utf-8")


def _splits(tmp_path, train, val, test):
    paths = {s: tmp_path / f"c.{s}.jsonl" for s in ("train", "val", "test")}
    for s, texts in zip(paths, (train, val, test)):
        _write(paths[s], texts)
    return paths


def _read(path):
    return [json.loads(line)["input"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_near_dup_holders_flags_a_doc_containing_a_later_doc():
    held = D.near_dup_holders([STORY], ["Header. " + STORY + " Footer.", "unrelated text " * 10], 0.2)
    assert held == {0}


def test_near_dup_drops_the_train_copy_and_keeps_the_test_doc(tmp_path):
    paths = _splits(tmp_path, ["Trial evidence: " + STORY + " more", "other train doc " * 10],
                    ["a val doc " * 10], [STORY])
    assert D.dedupe(paths, near_dup=0.2) == 1
    assert _read(paths["train"]) == ["other train doc " * 10]
    assert _read(paths["test"]) == [STORY]


def test_without_near_dup_only_exact_keys_drop(tmp_path):
    paths = _splits(tmp_path, ["Trial evidence: " + STORY + " more"], ["a val doc " * 10], [STORY])
    assert D.dedupe(paths) == 0
