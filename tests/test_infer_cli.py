"""Tests for the tools/infer.py CLI helpers (no model load)."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

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
        ns = argparse.Namespace(entities=None, events=None, schema_json=None,
                                model_schema=False, tasks=None)
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
        assert ns.beam_width == 8
        assert ns.threshold is None and ns.chunk_size is None and ns.chunk_overlap is None

    def test_unset_global_decode_is_none(self):
        ns = infer._parse_args(["--model", "m", "--input", "hi", "--entities", "a"])
        assert ns.global_decode is None


SHIPPED = {"open_vocab": ["entities"], "events": {"Attack": ["Target"]},
           "relations": ["works_for"], "classifications": [{"task": "t", "labels": ["a"]}]}


def _model(inference_defaults=None, default_schema=None, max_len=None):
    return SimpleNamespace(config=SimpleNamespace(
        inference_defaults=inference_defaults, default_schema=default_schema, max_len=max_len))


class TestModelSchema:
    def test_drops_open_vocab_marker(self):
        assert "open_vocab" not in infer._model_schema(_model(default_schema=SHIPPED).config)

    def test_tasks_narrow_it(self):
        assert infer._model_schema(_model(default_schema=SHIPPED).config, "events") == {
            "events": {"Attack": ["Target"]}}

    def test_cli_entities_fill_the_open_vocab(self):
        args = argparse.Namespace(entities="Person", events=None, schema_json=None,
                                  model_schema=True, tasks="events")
        assert infer._build_schema(args, _model(default_schema=SHIPPED).config) == {
            "events": {"Attack": ["Target"]}, "entities": ["Person"]}

    def test_no_shipped_schema_errors(self):
        with pytest.raises(SystemExit):
            infer._model_schema(_model().config)


class TestDecodeSettings:
    STORED = {"threshold": 0.3, "chunk_size": 4096, "chunk_overlap": 0, "global_decode": True}

    def _args(self, *argv):
        return infer._parse_args(["--model", "m", "--input", "hi", "--entities", "a", *argv])

    def test_checkpoint_defaults_fill_unset_flags(self):
        assert infer._decode_settings(self._args(), _model(self.STORED)) == self.STORED

    def test_explicit_flags_win(self):
        got = infer._decode_settings(
            self._args("--threshold", "0.5", "--chunk-size", "200", "--no-global-decode"),
            _model(self.STORED))
        assert got == {"threshold": 0.5, "chunk_size": 200, "chunk_overlap": 0, "global_decode": False}

    def test_no_stored_defaults_falls_back(self):
        got = infer._decode_settings(self._args(), _model(max_len=600))
        assert got == {"threshold": 0.5, "chunk_size": 400, "chunk_overlap": 0, "global_decode": False}
