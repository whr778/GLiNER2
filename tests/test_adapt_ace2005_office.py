"""Office ACE records keep their offsets once adapted to the GLiNER2 shape."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "train"))
import train as T  # noqa: E402

from gliner2.processing.coref import attach
from gliner2.processing.surface import SurfaceSpan
from gliner2.processor import _parse_gold
from gliner2.training.data import InputExample

_ADAPT = Path(__file__).resolve().parents[1] / "tools" / "data" / "adapt_ace2005_office.py"


def _load():
    spec = importlib.util.spec_from_file_location("adapt_ace2005_office", _ADAPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


adapt = _load()

TEXT = "Dallas hired Smith. Dallas lost."


def _office():
    return {
        "input": TEXT,
        "output": {
            "entities": {"GPE": ["Dallas"], "PER": ["Smith"]},
            "relations": [{"ORG-AFF.Employment": {"head": "Dallas", "tail": "Smith"}}],
            "events": [{
                "event_type": "Personnel.End-Position",
                "triggers": ["lost"],
                "arguments": [{"role": "Entity", "entity": "Dallas"}],
            }],
            "entity_facts": [
                {"start": 0, "end": 6, "text": "Dallas", "mention_id": "m1", "mention_type": "NAM", "entity_id": "e1", "entity_type": "GPE"},
                {"start": 13, "end": 18, "text": "Smith", "mention_id": "m2", "mention_type": "NAM", "entity_id": "e2", "entity_type": "PER"},
                {"start": 20, "end": 26, "text": "Dallas", "mention_id": "m3", "mention_type": "NAM", "entity_id": "e1", "entity_type": "GPE"},
            ],
            "relation_facts": [{
                "relation_type": "ORG-AFF.Employment",
                "head": {"start": 0, "end": 6, "text": "Dallas", "mention_id": "m1", "entity_id": "e1"},
                "tail": {"start": 13, "end": 18, "text": "Smith", "mention_id": "m2", "entity_id": "e2"},
            }],
            "event_facts": [{
                "event_type": "Personnel.End-Position",
                "event_id": "ev1",
                "triggers": [{"trigger_id": "t1", "start": 27, "end": 31, "text": "lost"}],
                "arguments": [{"role": "Entity", "start": 20, "end": 26, "text": "Dallas", "mention_id": "m3", "trigger_id": "t1"}],
            }],
            "clusters": [
                {"cluster_id": "e1", "mentions": [[0, 6], [20, 26]]},
                {"cluster_id": "e2", "mentions": [[13, 18]]},
            ],
        },
    }


def test_loader_keeps_the_string_summary_and_drops_the_facts():
    example = InputExample.from_dict(_office())
    assert example.events[0].triggers == ["lost"]
    assert example.events[0].arguments[0].entity == "Dallas"
    assert isinstance(_parse_gold(example.events[0].triggers[0]), str)
    assert attach(_office()["output"], _office().get("coref_clusters")).get("_coref_clusters") is None
    assert "event_facts" not in example.to_dict()["output"]


def test_adapter_pins_the_second_dallas_and_builds_coref():
    record = adapt.adapt_record(_office(), Counter())
    text = record["input"]
    event = record["output"]["events"][0]
    trigger = event["triggers"][0]
    argument = event["arguments"][0]["entity"]
    assert text[trigger["start"]:trigger["end"]] == "lost"
    assert argument == {"text": "Dallas", "start": 20, "end": 26}
    assert text[argument["start"]:argument["end"]] == "Dallas"
    span = _parse_gold(argument)
    assert isinstance(span, SurfaceSpan)
    assert (span.start, span.end) == (20, 26)
    assert record["coref_clusters"] == [
        [{"text": "Dallas", "start": 0, "end": 6}, {"text": "Dallas", "start": 20, "end": 26}],
        [{"text": "Smith", "start": 13, "end": 18}],
    ]
    attached = attach(record["output"], record["coref_clusters"])
    assert attached["_coref_clusters"][0][1]["start"] == 20
    assert record["output"]["relations"] == [{"ORG-AFF.Employment": {"head": "Dallas", "tail": "Smith"}}]
    assert record["output"]["entities"]["GPE"] == ["Dallas"]
    assert record["output"]["mention_types"]["Smith"] == "NAM"
    assert "entity_facts" not in record["output"]
    assert "clusters" not in record["output"]


def test_second_pass_leaves_the_gliner_record_in_place():
    once = adapt.adapt_record(_office(), Counter())
    twice = adapt.adapt_record(once, Counter())
    assert twice == once


def test_config_flag_adapts_on_read_and_leaves_a_gliner_record_alone(tmp_path):
    office = tmp_path / "ace2005_eng.train.jsonl"
    office.write_text(json.dumps(_office(), ensure_ascii=False) + "\n", encoding="utf-8")
    plain = tmp_path / "ace2005_eng.val.jsonl"
    plain.write_text(json.dumps(adapt.adapt_record(_office(), Counter()), ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        T.bind_data_flags({"data": {"office_ace": True}})
        adapted = T._read_records([str(office)])
        untouched = T._read_records([str(plain)])
    finally:
        T.bind_data_flags({})
    argument = adapted[0]["output"]["events"][0]["arguments"][0]["entity"]
    assert argument["start"] == 20 and argument["text"] == "Dallas"
    assert adapted[0]["coref_clusters"][0][1]["start"] == 20
    assert "event_facts" not in adapted[0]["output"]
    assert untouched[0]["output"]["events"][0]["triggers"][0]["start"] == 27
    T.bind_data_flags({"data": {"office_ace": False}})
    raw = T._read_records([str(office)])
    T.bind_data_flags({})
    assert raw[0]["output"]["events"][0]["triggers"] == ["lost"]
    assert "coref_clusters" not in raw[0]


def test_span_text_that_misses_the_slice_raises():
    office = _office()
    office["output"]["event_facts"][0]["arguments"][0]["text"] = "Smith"
    with pytest.raises(ValueError, match="covers"):
        adapt.adapt_record(office, Counter())


def test_multi_trigger_event_fact_splits_by_trigger_id_and_scores_arguments():
    from gliner2.training.chunking import _filter_events
    from gliner2.training.eval_metrics import score_predictions

    text = "Alpha bombed Baghdad on Monday. Later, Alpha struck Basra on Tuesday."
    office = {
        "input": text,
        "output": {
            "event_facts": [{
                "event_type": "Conflict.Attack",
                "event_id": "E1",
                "triggers": [
                    {"trigger_id": "E1-1", "start": 6, "end": 12, "text": "bombed"},
                    {"trigger_id": "E1-2", "start": 45, "end": 51, "text": "struck"},
                ],
                "arguments": [
                    {"role": "Attacker", "start": 0, "end": 5, "text": "Alpha", "trigger_id": "E1-1"},
                    {"role": "Place", "start": 13, "end": 20, "text": "Baghdad", "trigger_id": "E1-1"},
                    {"role": "Attacker", "start": 39, "end": 44, "text": "Alpha", "trigger_id": "E1-2"},
                    {"role": "Place", "start": 52, "end": 57, "text": "Basra", "trigger_id": "E1-2"},
                ],
            }]
        },
    }
    adapted = adapt.adapt_record(office, Counter())
    events = adapted["output"]["events"]
    assert len(events) == 2
    assert [e["triggers"][0]["text"] for e in events] == ["bombed", "struck"]
    assert [a["entity"]["text"] for a in events[0]["arguments"]] == ["Alpha", "Baghdad"]
    assert [a["entity"]["text"] for a in events[1]["arguments"]] == ["Alpha", "Basra"]

    pred = {
        "event_extraction": {
            "Conflict.Attack": [
                {"triggers": ["bombed"], "arguments": [{"role": "Attacker", "entity": "Alpha"}, {"role": "Place", "entity": "Baghdad"}]},
                {"triggers": ["struck"], "arguments": [{"role": "Attacker", "entity": "Alpha"}, {"role": "Place", "entity": "Basra"}]},
            ]
        }
    }
    metrics = score_predictions([adapted["output"]], [pred], report=False)
    assert metrics["eval_event_argument_strict_micro_f1"] == 1.0
    assert metrics["eval_event_argument_strict_micro_recall"] == 1.0

    # First-sentence chunk keeps E1-1 even though E1-2 lies outside the window.
    kept = _filter_events(events, text[:32], 0, 32)
    assert len(kept) == 1
    assert kept[0]["triggers"][0]["text"] == "bombed"


def test_unaligned_event_argument_is_unresolved_not_absent():
    import torch
    from gliner2.models.boundary.records import (
        _annotated_field_mask,
        _field_target_cols,
        _gold_spans_unresolved,
    )
    from gliner2.processing.boundary_preprocessing import unaligned_event_span_counts
    from gliner2.processor import SchemaTransformer
    from gliner2.training.trainer import ExtractorCollator

    class _Tok:
        def __init__(self):
            self.vocab = {"[P]": 1, "[E]": 2, "[C]": 3, "[L]": 4, "[SEP_STRUCT]": 5, "[SEP_TEXT]": 6,
                          "[CLS]": 101, "[SEP]": 102, "[PAD]": 0, "[UNK]": 100}
            self.pad_token_id = 0
            self.cls_token_id = 101
            self.sep_token_id = 102
            self.unk_token_id = 100
            self._next = 200

        def add_special_tokens(self, d):
            for tok in d.get("additional_special_tokens", []):
                if tok not in self.vocab:
                    self.vocab[tok] = self._next
                    self._next += 1
            return 0

        def convert_tokens_to_ids(self, toks):
            if isinstance(toks, str):
                return self.vocab.get(toks, 100)
            return [self.vocab.get(t, 100) for t in toks]

        def tokenize(self, text):
            return text.split() or [text]

        def encode(self, text, add_special_tokens=False):
            return [self.vocab.setdefault(t, self._next + i) for i, t in enumerate(self.tokenize(text))]

    unaligned_event_span_counts().clear()
    # Event 1 has an unaligned Arabic clitic argument ("القوات" inside "بالقوات") and no Place;
    # Event 2 supplies Place so the event schema includes both Target and Place roles.
    text = "هاجمت القوات بالقوات. قصفت بغداد"
    office = {
        "input": text,
        "output": {
            "event_facts": [{
                "event_type": "Conflict.Attack",
                "event_id": "E2",
                "triggers": [
                    {"trigger_id": "E2-1", "start": 0, "end": 5, "text": "هاجمت"},
                    {"trigger_id": "E2-2", "start": 22, "end": 26, "text": "قصفت"},
                ],
                "arguments": [
                    {"role": "Target", "start": 14, "end": 20, "text": "القوات", "trigger_id": "E2-1"},
                    {"role": "Place", "start": 27, "end": 32, "text": "بغداد", "trigger_id": "E2-2"},
                ],
            }]
        },
    }
    adapted = adapt.adapt_record(office, Counter())
    coll = ExtractorCollator(SchemaTransformer(tokenizer=_Tok()), is_training=True, architecture="boundary", event_records=True)
    batch = coll([(adapted["input"], adapted["output"])])
    spec = next(iter(batch.record_specs[0].values()))
    rec0 = batch.targets.records[0][0]
    by_name = {f.name: f for f in spec.fields}

    target_fspec = by_name["Target"]
    place_fspec = by_name["Place"]
    target_t = rec0.field_for_query(target_fspec.query_id)
    place_t = rec0.field_for_query(place_fspec.query_id)
    assert target_t.values == () and target_t.has_unaligned is True
    assert place_t.values == () and place_t.has_unaligned is False

    t_cols, _ = _field_target_cols(target_fspec, rec0, {})
    p_cols, _ = _field_target_cols(place_fspec, rec0, {})
    assert _gold_spans_unresolved(rec0, target_fspec, t_cols) is True
    assert _gold_spans_unresolved(rec0, place_fspec, p_cols) is False
    mask = _annotated_field_mask([rec0], spec.fields, torch.device("cpu"))
    assert bool(mask[0, spec.fields.index(target_fspec)]) is True
    assert bool(mask[0, spec.fields.index(place_fspec)]) is False
    assert unaligned_event_span_counts()["Target"] == 1


def test_clean_and_rebase_preserves_offsets_across_dumps_record(tmp_path, monkeypatch):
    text = "Alpha\u00ad\u200f \ufb01red at Bagh\u00addad."
    office = {
        "input": text,
        "output": {
            "event_facts": [{
                "event_type": "Conflict.Attack",
                "event_id": "E3",
                "triggers": [{"trigger_id": "E3-1", "start": text.index("\ufb01red"), "end": text.index("\ufb01red") + 4, "text": "\ufb01red"}],
                "arguments": [{"role": "Place", "start": text.index("Bagh"), "end": text.index("dad") + 3, "text": "Bagh\u00addad", "trigger_id": "E3-1"}],
            }],
            "clusters": [{"cluster_id": "C1", "mentions": [[text.index("Bagh"), text.index("dad") + 3]]}],
        },
    }
    inp = tmp_path / "office.json"
    out = tmp_path / "out.jsonl"
    inp.write_text(json.dumps(office, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["adapt_ace2005_office.py", "--input", str(inp), "--out", str(out)])
    adapt.main()
    loaded = json.loads(out.read_text(encoding="utf-8").strip())
    cleaned = loaded["input"]
    trig = loaded["output"]["events"][0]["triggers"][0]
    arg = loaded["output"]["events"][0]["arguments"][0]["entity"]
    cm = loaded["coref_clusters"][0][0]
    assert cleaned[trig["start"]:trig["end"]] == trig["text"] == "fired"
    assert cleaned[arg["start"]:arg["end"]] == arg["text"] == "Baghdad"
    assert cleaned[cm["start"]:cm["end"]] == cm["text"] == "Baghdad"


def test_build_negative_pools_adapts_office_records(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_negative_pools",
        Path(__file__).resolve().parents[1] / "tools" / "data" / "build_negative_pools.py",
    )
    bnp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bnp)

    train_file = tmp_path / "ace_office.train.jsonl"
    rec = {
        "input": "Dallas lost.",
        "output": {
            "entity_facts": [{"start": 0, "end": 6, "text": "Dallas", "entity_type": "GPE"}],
            "event_facts": [{
                "event_type": "Conflict.Attack",
                "triggers": [{"trigger_id": "t1", "start": 7, "end": 11, "text": "lost"}],
                "arguments": [{"role": "Attacker", "start": 0, "end": 6, "text": "Dallas", "trigger_id": "t1"}],
            }],
        },
    }
    train_file.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = {"data": {"office_ace": True, "corpora": [str(tmp_path / "ace_office")]}}
    pools = bnp.build_pools(cfg, tmp_path / "cfg.yaml")
    info = pools["ace_office"]
    assert info["annotates"]["entities"] is True
    assert info["annotates"]["events"] is True
    assert info["entities"] == ["GPE"]
    assert info["events"] == {"Conflict.Attack": ["Attacker"]}


def test_adapt_edge_cases_null_offsets_empty_records_and_authoritative_entities():
    # Null offsets fall back to string text without crashing.
    null_rec = {
        "input": "Dallas lost.",
        "output": {
            "event_facts": [{
                "event_type": "Loss",
                "triggers": [{"trigger_id": "t1", "start": None, "end": None, "text": "lost"}],
                "arguments": [{"role": "Agent", "start": None, "end": None, "text": "Dallas", "trigger_id": "t1"}],
            }]
        },
    }
    adapted_null = adapt.adapt_record(null_rec, Counter())
    assert adapted_null["output"]["events"] == [
        {"event_type": "Loss", "triggers": ["lost"], "arguments": [{"role": "Agent", "entity": "Dallas"}]}
    ]

    # Empty side-list keys are still detected as office records and stripped cleanly.
    empty_rec = {
        "input": "Nothing happened.",
        "output": {"entity_facts": [], "relation_facts": [], "event_facts": [], "clusters": []},
    }
    assert adapt.is_office_record(empty_rec) is True
    assert adapt.adapt_record(empty_rec, Counter()) == {"input": "Nothing happened.", "output": {}}

    # Clusters-only record adapts without raising ValueError.
    clusters_only = {
        "input": "Nothing happened.",
        "output": {"clusters": [{"cluster_id": "c1", "mentions": [[0, 7]]}]},
    }
    adapted_c = adapt.adapt_record(clusters_only, Counter())
    assert adapted_c["output"] == {}
    assert adapted_c["coref_clusters"] == [[{"text": "Nothing", "start": 0, "end": 7}]]

    # Non-empty entity_facts is authoritative and does not merge stale output.entities keys.
    ent_rec = {
        "input": TEXT,
        "output": {
            "entities": {"STALE_TYPE": ["Dallas"]},
            "entity_facts": [{"start": 0, "end": 6, "text": "Dallas", "entity_type": "GPE"}],
        },
    }
    assert adapt.adapt_record(ent_rec, Counter())["output"]["entities"] == {"GPE": ["Dallas"]}


def test_score_predictions_coref_scores_adapts_office_clusters():
    import score_predictions as sp

    row = {
        "input": TEXT,
        "output": {
            "entities": {"GPE": [{"text": "Dallas", "start": 0, "end": 6}, {"text": "Dallas", "start": 20, "end": 26}]},
            "coref_clusters": [[{"text": "Dallas", "start": 0, "end": 6}, {"text": "Dallas", "start": 20, "end": 26}]],
        },
        "gold": {
            "entity_facts": [
                {"start": 0, "end": 6, "text": "Dallas", "entity_type": "GPE"},
                {"start": 20, "end": 26, "text": "Dallas", "entity_type": "GPE"},
            ],
            "clusters": [{"cluster_id": "c1", "mentions": [[0, 6], [20, 26]]}],
        },
    }
    scores = sp.coref_scores([row])
    assert scores["docs"] == 1
    assert scores["all"]["f1"] == 1.0
    assert scores["candidate_recall"] == 1.0
