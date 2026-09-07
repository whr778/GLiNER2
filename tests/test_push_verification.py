"""A push that writes nothing must not report success.

2026-09-07: `whr778/gliner2-eb16-composed` contains exactly one file, `.gitattributes`.
upload_folder returned WITHOUT raising, the runner printed PUSH OK, the trap terminated
the box, and ~15 hours of A100 went with it. The retry added that morning could not help,
because nothing threw. Verification is the only thing that catches this class.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "train"))


def test_verify_reports_missing_weights_as_missing():
    from save_or_die import verify
    api = MagicMock()
    api.list_repo_files.return_value = [".gitattributes"]      # the real failure
    assert verify(api, "ns/name", ["config.json", "model.safetensors"]) == [
        "config.json", "model.safetensors"]


def test_verify_is_satisfied_only_when_every_file_is_there():
    from save_or_die import verify
    api = MagicMock()
    api.list_repo_files.return_value = [".gitattributes", "config.json"]
    assert verify(api, "ns/name", ["config.json", "model.safetensors"]) == [
        "model.safetensors"], "a partial push is not a push"
    api.list_repo_files.return_value = [".gitattributes", "config.json", "model.safetensors"]
    assert verify(api, "ns/name", ["config.json", "model.safetensors"]) == []


def test_an_unlistable_repo_counts_as_empty_not_as_success():
    """If the repo cannot be read, the safe assumption is that nothing is in it."""
    from save_or_die import verify
    api = MagicMock()
    api.list_repo_files.side_effect = RuntimeError("403")
    assert verify(api, "ns/name", ["config.json"]) == ["config.json"]


def test_push_to_hub_verifies_rather_than_trusting_a_clean_return():
    src = (ROOT / "tools" / "train" / "push_to_hub.py").read_text(encoding="utf-8")
    assert "_missing()" in src, "push_to_hub must check the Hub, not just catch exceptions"
    assert "MODEL NOT SAVED" in src, "a failed push must say so unmistakably"
    # the verify must run even when upload_folder did NOT raise
    i, j = src.index("except Exception as e"), src.index("gone = _missing()")
    assert j > i, "verification must sit outside the except branch"
