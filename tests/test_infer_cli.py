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
        return [{"echo": t, "schema": sorted(s), "schema_full": s} for t, s in zip(texts, schemas)]


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


class TestLabelsFile:
    LABELS = "labels:\n  entities:\n    rollup: false\n    separator: .\n    map:\n      Company Name: CompanyName\n"
    RECS = [{"input": "a", "output": {"entities": {"Company Name": ["Acme"], "Person": ["Bo"]}}}]

    def _run(self, tmp_path, monkeypatch, *argv):
        (tmp_path / "labels.yaml").write_text(self.LABELS, encoding="utf-8")
        return TestOutputJsonl._run(self, tmp_path, monkeypatch,
                                    "--labels-file", str(tmp_path / "labels.yaml"), *argv)

    def test_gold_mapped_carries_the_transformed_labels_and_gold_stays_raw(self, tmp_path, monkeypatch):
        rows, _ = self._run(tmp_path, monkeypatch, "--gold-schema")
        assert rows[0]["gold"] == self.RECS[0]["output"]
        assert rows[0]["gold_mapped"]["entities"] == {"CompanyName": ["Acme"], "Person": ["Bo"]}

    def test_gold_schema_is_built_from_the_mapped_gold(self, tmp_path, monkeypatch):
        rows, _ = self._run(tmp_path, monkeypatch, "--gold-schema")
        assert set(rows[0]["output"]["schema_full"]["entities"]) == {"CompanyName", "Person"}

    def test_mismatch_counts_differing_entries(self):
        block = {"entities": {"map": {"a": "A", "b": "B"}}}
        assert infer._map_mismatch(block, {"entities": {"map": {"a": "A", "b": "X"}}}) == {"entities": 1}
        assert infer._map_mismatch(block, {"entities": {"map": {"a": "A", "b": "B"}}}) == {}


class _BoundaryStub(_StubModel):
    """A boundary-shaped stub: records the coreference settings IN FORCE when decode runs."""
    def __init__(self, coref_link=True):
        super().__init__()
        from gliner2.configuration import BoundaryHeadSettings
        self.config.boundary_head = {"record_coref_link": coref_link}
        self.boundary_settings = BoundaryHeadSettings(record_coref_link=coref_link)
        self.seen = None

    def modules(self):
        return []

    def batch_extract_long(self, texts, schemas, **kw):
        s = self.boundary_settings
        self.seen = (s.record_merge_coreferent, s.record_coref_link_threshold)
        return super().batch_extract_long(texts, schemas, **kw)


class TestCoreferentMerge:
    def _run(self, tmp_path, monkeypatch, stub, *argv):
        src = tmp_path / "in.jsonl"
        src.write_text(json.dumps({"input": "a", "output": {}}) + "\n", encoding="utf-8")
        import gliner2
        monkeypatch.setattr(gliner2.AutoExtractor, "from_pretrained", staticmethod(lambda *a, **k: stub))
        infer.main(["--model", "m", "--input", str(src), "--entities", "Person",
                    "--output", str(tmp_path / "p.jsonl"), *argv])
        return stub

    def test_flags_are_the_settings_decode_runs_with(self, tmp_path, monkeypatch):
        stub = self._run(tmp_path, monkeypatch, _BoundaryStub(),
                         "--merge-coreferent", "link", "--coref-threshold", "0.7")
        assert stub.seen == ("link", 0.7)
        assert stub.config.boundary_head["record_merge_coreferent"] == "link"

    def test_no_flags_keep_the_checkpoint_settings(self, tmp_path, monkeypatch):
        stub = self._run(tmp_path, monkeypatch, _BoundaryStub())
        assert stub.seen == ("off", 0.5)

    def test_link_on_a_checkpoint_without_the_link_refuses(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="record_coref_link"):
            self._run(tmp_path, monkeypatch, _BoundaryStub(coref_link=False), "--merge-coreferent", "link")

    def test_overrides_only_name_what_was_passed(self):
        args = infer._parse_args(["--model", "m", "--input", "x", "--coref-threshold", "0.6"])
        assert infer._coref_overrides(args) == {"record_coref_link_threshold": 0.6}
        assert infer._coref_overrides(infer._parse_args(["--model", "m", "--input", "x"])) == {}

    def test_unknown_mode_is_refused(self):
        with pytest.raises(SystemExit):
            infer._parse_args(["--model", "m", "--input", "x", "--merge-coreferent", "union"])


class TestDevice:
    def test_default_is_auto(self):
        assert infer._parse_args(["--model", "m", "--input", "x"]).device == "auto"

    def test_explicit_cpu(self):
        assert infer._parse_args(["--model", "m", "--input", "x", "--device", "cpu"]).device == "cpu"

    def test_unknown_device_is_rejected(self):
        with pytest.raises(SystemExit):
            infer._parse_args(["--model", "m", "--input", "x", "--device", "gpu"])


class TestSubfolderAndStructures:
    def test_split_model_subfolder_at_and_colon(self, tmp_path):
        assert infer._split_model_subfolder("whr778/gliner2-eb22g@checkpoint-epoch-1") == (
            "whr778/gliner2-eb22g", {"subfolder": "checkpoint-epoch-1"}
        )
        assert infer._split_model_subfolder("whr778/gliner2-eb22g:checkpoint-epoch-2") == (
            "whr778/gliner2-eb22g", {"subfolder": "checkpoint-epoch-2"}
        )
        assert infer._split_model_subfolder("whr778/gliner2-eb22g") == ("whr778/gliner2-eb22g", {})
        local = tmp_path / "a:b"
        local.mkdir()
        assert infer._split_model_subfolder(str(local)) == (str(local), {})

    def test_declare_records_defaults_mode_and_anchor(self):
        sc = {"structures": {"r": {"fields": [{"name": "a"}, {"name": "b"}]}}}
        out = infer._declare_records(sc)
        assert out["structures"]["r"]["mode"] == "natural"
        assert out["structures"]["r"]["anchor"] == "a"

    def test_declare_records_accepts_list_shorthand_and_preserves_explicit_mode(self):
        sc = {
            "structures": {
                "s1": ["dead", "injured"],
                "s2": {"mode": "anchorless", "fields": [{"name": "x"}]},
            }
        }
        out = infer._declare_records(sc)
        assert out["structures"]["s1"] == {
            "fields": [{"name": "dead"}, {"name": "injured"}],
            "mode": "natural",
            "anchor": "dead",
        }
        assert out["structures"]["s2"]["mode"] == "anchorless"
        assert "anchor" not in out["structures"]["s2"]

    def test_cli_relations_and_structures_flags(self):
        ns = infer._parse_args([
            "--model", "m", "--input", "x",
            "--relations", "located_in, part_of",
            "--structures", '{"casualty": ["dead", "injured"]}',
        ])
        sc = infer._build_schema(ns)
        assert sc["relations"] == ["located_in", "part_of"]
        assert sc["structures"]["casualty"]["mode"] == "natural"
        assert sc["structures"]["casualty"]["anchor"] == "dead"

    def test_decode_settings_per_head_thresholds_and_head_spans(self):
        from types import SimpleNamespace
        model = SimpleNamespace(config=SimpleNamespace(
            max_len=4096,
            inference_defaults={
                "threshold": 0.9,
                "chunk_size": 2730,
                "chunk_overlap": 0,
                "global_decode": False,
                "per_head_thresholds": {"entity": 0.9, "relation": 0.3, "event": 0.7},
            },
        ))
        # Default: loads per_head_thresholds from inference_defaults
        ns = infer._parse_args(["--model", "m", "--input", "x", "--entities", "Person"])
        d = infer._decode_settings(ns, model)
        assert d["threshold"] == 0.9
        assert d["per_head_thresholds"] == {"entity": 0.9, "relation": 0.3, "event": 0.7}
        assert "head_spans" not in d

        # Explicit --threshold disables stored per_head_thresholds unless --per-head-thresholds is given
        ns_thr = infer._parse_args(["--model", "m", "--input", "x", "--entities", "Person", "--threshold", "0.5"])
        d_thr = infer._decode_settings(ns_thr, model)
        assert d_thr["threshold"] == 0.5
        assert "per_head_thresholds" not in d_thr

        # --no-per-head-thresholds disables stored per_head_thresholds
        ns_no = infer._parse_args(["--model", "m", "--input", "x", "--entities", "Person", "--no-per-head-thresholds"])
        d_no = infer._decode_settings(ns_no, model)
        assert "per_head_thresholds" not in d_no

        # Explicit --per-head-thresholds and --head-spans
        ns_custom = infer._parse_args([
            "--model", "m", "--input", "x", "--entities", "Person",
            "--per-head-thresholds", '{"entity": 0.8, "event": 0.4}',
            "--head-spans",
        ])
        d_custom = infer._decode_settings(ns_custom, model)
        assert d_custom["per_head_thresholds"] == {"entity": 0.8, "event": 0.4}
        assert d_custom["head_spans"] is True


class TestHeadSpansPostProcessing:
    def test_strip_leading_stopwords_preserves_offsets_and_quantifiers(self):
        from gliner2.inference.head_spans import (
            strip_leading_stopwords_from_results,
            strip_leading_stopwords_text,
        )
        doc = "In the northern province, the hackers attacked the hospital and killed at least 246 people in New York on May 15."
        s1, d1 = strip_leading_stopwords_text("the hackers")
        assert (s1, d1) == ("hackers", 4)
        s2, d2 = strip_leading_stopwords_text("In the northern province")
        assert (s2, d2) == ("northern province", 7)
        # Quantifier and proper-noun guards
        assert strip_leading_stopwords_text("at least 246 people") == ("at least 246 people", 0)
        assert strip_leading_stopwords_text("a dozen people") == ("a dozen people", 0)
        assert strip_leading_stopwords_text("New York") == ("New York", 0)
        assert strip_leading_stopwords_text("May 15") == ("May 15", 0)

        h_idx = doc.index("the hackers")
        hosp_idx = doc.index("the hospital")
        toll_idx = doc.index("at least 246 people")
        raw = {
            "entities": {
                "Person": [{"text": "the hackers", "start": h_idx, "end": h_idx + 11, "confidence": 0.9}],
                "Organization": [
                    {"text": "the hospital", "start": hosp_idx, "end": hosp_idx + 12, "confidence": 0.8},
                    {"text": "hospital", "start": hosp_idx + 4, "end": hosp_idx + 12, "confidence": 0.7},
                ],
            },
            "event_extraction": {
                "Attack": [{
                    "triggers": [{"text": "attacked", "start": 38, "end": 46}],
                    "arguments": [
                        {"role": "Attacker", "entity": {"text": "the hackers", "start": h_idx, "end": h_idx + 11}},
                        {"role": "Victim", "entity": {"text": "the hospital", "start": hosp_idx, "end": hosp_idx + 12}},
                        {"role": "Casualties", "entity": {"text": "at least 246 people", "start": toll_idx, "end": toll_idx + 19}},
                    ],
                }],
            },
            "relation_extraction": {
                "targeted": [(
                    "the hackers",
                    "the hospital",
                )],
            },
        }
        cleaned = strip_leading_stopwords_from_results(raw)
        ent_p = cleaned["entities"]["Person"][0]
        assert ent_p["text"] == "hackers"
        assert doc[ent_p["start"]:ent_p["end"]] == "hackers"
        # Deduplicated after stripping "the hospital" -> "hospital" at identical offsets
        assert len(cleaned["entities"]["Organization"]) == 1
        assert cleaned["entities"]["Organization"][0]["text"] == "hospital"

        args = {a["role"]: a["entity"] for a in cleaned["event_extraction"]["Attack"][0]["arguments"]}
        assert args["Attacker"]["text"] == "hackers"
        assert doc[args["Attacker"]["start"]:args["Attacker"]["end"]] == "hackers"
        assert args["Victim"]["text"] == "hospital"
        assert doc[args["Victim"]["start"]:args["Victim"]["end"]] == "hospital"
        assert args["Casualties"]["text"] == "at least 246 people"
        assert cleaned["relation_extraction"]["targeted"] == [("hackers", "hospital")]

    def test_entity_wildcard_expansion_in_schema_and_label_map(self):
        from gliner2.inference.label_map import apply_label_map
        from gliner2.inference.schema import DEFAULT_OPEN_VOCAB_ENTITIES, Schema

        assert len(DEFAULT_OPEN_VOCAB_ENTITIES) == 125
        for token in ("*", "any", ["*"], ["any", "custom_type"]):
            s = Schema.from_dict({"entities": token})
            built = s.build()
            assert "person" in built["entities"]
            assert "organization" in built["entities"]
            if isinstance(token, list) and "custom_type" in token:
                assert len(built["entities"]) == 126
                assert "custom_type" in built["entities"]
            else:
                assert len(built["entities"]) == 125

        mapped, rewrites = apply_label_map(
            {"entities": ["*"]},
            {"entities": {"map": {"person": "Person"}}},
            {"name": "title_snake"},
        )
        assert "Person" in mapped["entities"]
        assert "Geopolitical_Entity" in mapped["entities"]
        assert len(mapped["entities"]) == 125


