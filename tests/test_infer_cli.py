"""Tests for the tools/infer.py CLI helpers (no model load)."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import infer  # noqa: E402


class TestReadTexts:
    def test_literal_string(self):
        assert infer._read_texts("hello world") == ["hello world"]

    def test_txt_file(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text("a document", encoding="utf-8")
        assert infer._read_texts(str(p)) == ["a document"]

    def test_jsonl_file(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text('{"input": "doc one"}\n{"input": "doc two"}\n', encoding="utf-8")
        assert infer._read_texts(str(p)) == ["doc one", "doc two"]


class TestBuildSchema:
    def _args(self, **kw):
        ns = argparse.Namespace(entities=None, events=None, schema_json=None)
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def test_entities_comma_split_and_trim(self):
        assert infer._build_schema(self._args(entities="person, org ,loc")) == {
            "entities": ["person", "org", "loc"]
        }

    def test_events_json(self):
        assert infer._build_schema(self._args(events='{"Attack":["Target"]}')) == {
            "events": {"Attack": ["Target"]}
        }

    def test_schema_json_overrides_others(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text('{"entities":["a"]}', encoding="utf-8")
        assert infer._build_schema(self._args(entities="x", schema_json=str(p))) == {
            "entities": ["a"]
        }

    def test_empty_errors(self):
        with pytest.raises(SystemExit):
            infer._build_schema(self._args())


class TestParseArgs:
    def test_flags_and_defaults(self):
        ns = infer._parse_args(
            ["--model", "m", "--input", "hi", "--entities", "a", "--global-decode"]
        )
        assert ns.global_decode is True
        assert ns.chunk_size == 384 and ns.chunk_overlap == 128 and ns.beam_width == 8
