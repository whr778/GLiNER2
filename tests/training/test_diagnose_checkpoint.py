"""diagnose_checkpoint.val_sources: event_files val first, then EVALUATED corpora only (never train_only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "train"))
from diagnose_checkpoint import val_sources  # noqa: E402


def test_val_sources_skip_train_only_and_keep_event_files():
    cfg = {"data": {"corpora": ["data/casie", "data/wikievents_off", "data/biored"], "train_only": ["wikievents_off"],
                    "event_files": {"cmnee": {"train": "x", "val": "data/scaling_joint/cmnee.val.jsonl"},
                                    "maven": {"train": "y"},
                                    "casie": {"val": "data/scaling_joint/casie.val.jsonl"}}}}
    assert val_sources(cfg) == {"cmnee": "data/scaling_joint/cmnee.val.jsonl",
                                "casie": "data/scaling_joint/casie.val.jsonl",
                                "biored": "data/biored.val.jsonl"}
