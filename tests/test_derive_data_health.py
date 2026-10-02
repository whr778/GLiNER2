"""tools/derive/data_health.py: contamination blocks, shift warns, siblings are found."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "derive"))
from data_health import _flat, contamination, cross_corpus, label_shift, siblings  # noqa: E402


def _write(base: Path, splits: dict) -> dict:
    paths = {}
    for s, texts in splits.items():
        p = Path(f"{base}.{s}.jsonl")
        p.write_text("\n".join(json.dumps({"input": t, "output": {}}) for t in texts) + "\n", encoding="utf-8")
        paths[s] = p
    return paths


def test_test_duplicate_and_leak_block_train_duplicate_warns(tmp_path):
    paths = _write(tmp_path / "c", {"train": ["a", "a", "Leak"], "val": ["v"], "test": ["leak ", "t", "t"]})
    found = {(f["severity"], f["check"], (f["evidence"] or {}).get("split") or (f["evidence"] or {}).get("pair"))
             for f in contamination(str(tmp_path / "c"), paths)}
    assert ("WARN", "duplicates", "train") in found
    assert ("BLOCK", "duplicates", "test") in found
    assert ("BLOCK", "cross_split", "train&test") in found       # case/whitespace-normalized key


def test_siblings_and_cross_corpus_overlap(tmp_path):
    a = _write(tmp_path / "ace_a", {"train": ["x"], "val": ["v"], "test": ["shared"]})
    _write(tmp_path / "ace_b", {"train": ["shared"], "val": ["w"], "test": ["y"]})
    sibs = siblings(str(tmp_path / "ace_a"), [])
    assert sibs == [str(tmp_path / "ace_b")]
    f = cross_corpus(str(tmp_path / "ace_a"), a, sibs)
    assert f and f[0]["evidence"]["our_eval_in_their_train"] == 1


def test_label_shift_unseen_and_thin_support():
    by_split = {"train": {"event_types": {"Attack": 90, "Meet": 10}},
                "val": {"event_types": {"Attack": 45, "Meet": 5}},
                "test": {"event_types": {"Meet": 40, "Travel": 3}}}
    measured = {}
    checks = {f["check"] for f in label_shift(by_split, measured)}
    assert {"label_shift", "unseen_labels", "thin_support"} <= checks
    assert measured["tvd"]["event_types:train/val"] < 0.15 < measured["tvd"]["event_types:train/test"]


def test_classifications_flatten_to_task_label():
    assert _flat({"sentiment": {"pos": 3, "neg": 1}}) == {"sentiment:pos": 3, "sentiment:neg": 1}
