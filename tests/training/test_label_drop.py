"""A label mapped to null is DROPPED by the training transform and by inference's apply_label_map (TODO #40)."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "train"))
import train as T  # noqa: E402
from gliner2.inference.label_map import apply_label_map  # noqa: E402

LABELS = {"relations": {"map": {"no relation": None, "No_Relation": None, "capital": "Capital"}}}


def test_training_transform_drops_null_mapped_relations():
    rec = {"input": "x", "output": {"relations": [{"no relation": {"head": "A", "tail": "B"}},
                                                  {"capital": {"head": "France", "tail": "Paris"}}]}}
    out = T.transform_record(rec, T._category_fns(LABELS))
    assert out["output"]["relations"] == [{"Capital": {"head": "France", "tail": "Paris"}}]


def test_inference_drops_raw_and_styled_spellings():
    menu = {"relations": [{"no relation": {"head": "", "tail": ""}}, "No_Relation", {"capital": {"head": "", "tail": ""}}]}
    out, applied = apply_label_map(menu, LABELS)
    assert out["relations"] == [{"Capital": {"head": "", "tail": ""}}]
    assert applied["relations"]["no relation"] is None


def test_unmapped_labels_pass_through():
    out, _ = apply_label_map({"relations": ["spouse"]}, LABELS)
    assert out["relations"] == ["spouse"]


def test_v4_is_v3_plus_the_pinned_drops_only():
    root = Path(__file__).resolve().parents[2] / "tools/train/config/labels"
    v3 = yaml.safe_load((root / "unified-full-v3.yaml").read_text(encoding="utf-8"))["labels"]
    v4 = yaml.safe_load((root / "unified-full-v4.yaml").read_text(encoding="utf-8"))["labels"]
    changed = {(c, k) for c in v4 if isinstance(v4[c], dict) and "map" in v4[c]
               for k in set(v3[c]["map"]) | set(v4[c]["map"]) if v3[c]["map"].get(k, "<absent>") != v4[c]["map"].get(k, "<absent>")}
    assert changed == {("relations", "no relation"), ("relations", "No_Relation")}
    assert v4["relations"]["map"]["no relation"] is None
