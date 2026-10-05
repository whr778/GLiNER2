"""Tests for the tools/infer.py CLI helpers (no model load)."""

import argparse
import json
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


class _StubModel:
    """batch_extract_long echoes each text's schema keys, so ordering and chunking are checkable."""
    def __init__(self):
        self.config = SimpleNamespace(inference_defaults={"threshold": 0.3, "chunk_size": 64,
                                                          "chunk_overlap": 0, "global_decode": False},
                                      label_map=None, default_schema=None, max_len=None)
        self.calls = []

    def batch_extract_long(self, texts, schemas, **kw):
        self.calls.append(len(texts))
        return [{"echo": t, "schema": sorted(s)} for t, s in zip(texts, schemas)]


class TestOutputJsonl:
    RECS = [{"input": "a", "output": {"entities": {"Person": ["a"]}}},
            {"input": "b", "output": {}},
            {"input": "c", "output": {"events": [{"event_type": "Attack", "triggers": ["c"], "arguments": []}]}}]

    def _run(self, tmp_path, monkeypatch, *argv):
        src = tmp_path / "in.jsonl"
        src.write_text("".join(json.dumps(r) + "\n" for r in self.RECS), encoding="utf-8")
        stub = _StubModel()
        import gliner2
        monkeypatch.setattr(gliner2.AutoExtractor, "from_pretrained", staticmethod(lambda *a, **k: stub))
        out = tmp_path / "preds.jsonl"
        infer.main(["--model", "m", "--input", str(src), "--output", str(out), *argv])
        return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()], stub

    def test_one_line_per_record_in_order_with_gold(self, tmp_path, monkeypatch):
        rows, _ = self._run(tmp_path, monkeypatch, "--gold-schema")
        assert [r["input"] for r in rows] == ["a", "b", "c"]
        assert [r["gold"] for r in rows] == [r["output"] for r in self.RECS]

    def test_gold_schema_is_per_record_and_empty_gold_is_not_decoded(self, tmp_path, monkeypatch):
        rows, stub = self._run(tmp_path, monkeypatch, "--gold-schema")
        assert rows[0]["output"]["schema"] == ["entities"]
        assert rows[1]["output"] == {}
        assert rows[2]["output"]["schema"] == ["events"]
        assert sum(stub.calls) == 2

    def test_written_in_chunks(self, tmp_path, monkeypatch):
        _, stub = self._run(tmp_path, monkeypatch, "--entities", "Person", "--docs-per-write", "2")
        assert stub.calls == [2, 1]

    def test_gold_schema_refuses_other_schema_options(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit):
            self._run(tmp_path, monkeypatch, "--gold-schema", "--entities", "Person")
