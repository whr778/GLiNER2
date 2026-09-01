"""Tests for the data/ restore tool.

The rule under test cost a real bug: shadowing parent-repo copies by BASENAME suppressed
three base splits the 137k config needs, because data/x.val.jsonl and
data/scaling_joint/x.val.jsonl share a name while being different files. Only scaling
SLICES are misplaced in parent repos.
"""

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "data" / "restore_from_hf.py"


def _load():
    spec = importlib.util.spec_from_file_location("restore_from_hf", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


restore = _load()


def test_slice_pattern_matches_scaling_slices():
    for name in ("docfee.j100k.test.jsonl", "duee.j40k.train.jsonl",
                 "casie.s10k.train.jsonl", "sentence_rex.j10k.val.jsonl"):
        assert restore.SLICE.search(name), f"{name} should be treated as a slice"


def test_slice_pattern_leaves_base_splits_alone():
    """These share a basename with a scaling_joint file but are NOT the same file."""
    for name in ("sentence_rex.val.jsonl", "biored.val.jsonl",
                 "bio_ner_relations.val.jsonl", "chfinann.val.jsonl",
                 "docfee.train.jsonl", "casie.test.jsonl"):
        assert not restore.SLICE.search(name), f"{name} must stay restorable"


def test_wanted_from_config_reads_corpora_and_event_files():
    cfg = (Path(__file__).resolve().parents[1] / "tools" / "train" / "config"
           / "joint-boundary-mmbert-137k.yaml")
    if not cfg.exists():
        return
    wanted = restore.wanted_from_config(cfg)
    assert "data/sentence_rex.val.jsonl" in wanted          # a corpora base split
    assert "data/scaling_joint/chfinann.val.jsonl" in wanted  # an event_files path
    assert len(wanted) == 36
