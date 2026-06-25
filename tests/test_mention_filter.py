"""Unit tests for the shared mention-type filter (tools/data/_mention_filter.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "data"))
from _mention_filter import MentionFilter, load_mention_filter  # noqa: E402


def test_keep_all_when_inactive():
    mf = MentionFilter(None)
    assert not mf.active
    assert mf.allows("PRO") and mf.allows("NOM") and mf.allows(None)


def test_allow_list_is_case_insensitive_and_keeps_untyped():
    mf = MentionFilter({"NAM", "NOM"})
    assert mf.active
    assert mf.allows("NAM") and mf.allows("nom") and mf.allows("  Nam ")
    assert not mf.allows("PRO")
    assert mf.allows(None) and mf.allows("")  # untyped is never filtered


def test_empty_allow_set_drops_all_typed():
    mf = MentionFilter(set())
    assert mf.active
    assert not mf.allows("NAM") and not mf.allows("PRO")
    assert mf.allows(None)


def test_load_no_path_keeps_all():
    assert load_mention_filter(None, "ace2005").allowed is None


def test_load_per_converter_override_and_global_fallback(tmp_path):
    cfg = tmp_path / "mf.yaml"
    cfg.write_text(
        "allow: [NAM, NOM, PRO]\n"
        "converters:\n"
        "  ace2005:\n"
        "    allow: [NAM, NOM]\n",
        encoding="utf-8",
    )
    assert load_mention_filter(cfg, "ace2005").allowed == {"NAM", "NOM"}   # override
    assert load_mention_filter(cfg, "other").allowed == {"NAM", "NOM", "PRO"}  # global


def test_load_global_only(tmp_path):
    cfg = tmp_path / "mf.yaml"
    cfg.write_text("allow: [NAM]\n", encoding="utf-8")
    assert load_mention_filter(cfg, "ace2005").allowed == {"NAM"}


def test_load_no_allow_key_keeps_all(tmp_path):
    cfg = tmp_path / "mf.yaml"
    cfg.write_text("converters:\n  ace2005:\n    note: nothing\n", encoding="utf-8")
    assert load_mention_filter(cfg, "ace2005").allowed is None


def test_load_missing_file_errors():
    import pytest
    with pytest.raises(SystemExit):
        load_mention_filter(Path("/no/such/mention_filter.yaml"), "ace2005")
