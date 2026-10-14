"""gold.py: gold placed as training places it, and every span flagged under the scorer's strict keys."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gold  # noqa: E402
from gliner2.training import eval_metrics as M  # noqa: E402

TEXT = "Ed Smith said Houston-area plants in Houston closed. Smith said more."
ROW = {
    "input": TEXT,
    "output": {
        "entities": {"Person": [{"text": "Ed Smith", "start": 0, "end": 8}],
                     "Location": [{"text": "Houston", "start": 37, "end": 44}]},
        "event_extraction": {"Contact.Communicate": [
            {"triggers": [{"text": "said", "start": 9, "end": 13}],
             "arguments": [{"role": "Speaker", "entity": {"text": "Ed Smith", "start": 0, "end": 8}}]}]},
    },
    "gold": {
        "entities": {"Person": ["Ed Smith", "Smith"], "location": ["Houston"], "Org": ["Acme"]},
        "events": [{"event_type": "Contact.Communicate", "triggers": ["said"],
                    "arguments": [{"role": "Speaker", "entity": "Ed Smith"}]}],
    },
}


def view():
    return gold.record_view(json.loads(json.dumps(ROW)))


def test_gold_is_placed_like_training_every_occurrence_and_not_inside_a_hyphenated_word():
    v = view()
    assert [(s["start"], s["end"]) for s in v["gold"]["entities"]["Person"]] == [(0, 8), (3, 8), (53, 58)]
    # "Houston-area" is one word to the splitter, so only the standalone "Houston" is placed.
    assert [(s["start"], s["end"]) for s in v["gold"]["entities"]["location"]] == [(37, 44)]
    assert v["unlocated"] == ["Acme"]


def test_match_flags_agree_with_the_strict_scorer():
    v = view()
    hits = {(label, s["text"]) for label, spans in v["pred"]["entities"].items() for s in spans if s["match"]}
    assert hits == M._gold_entity_set(ROW["gold"]) & M._pred_entity_set(ROW["output"])
    # Case-sensitive like the scorer: gold "location" does not match predicted "Location".
    assert v["pred"]["entities"]["Location"][0]["match"] is False
    mention = v["pred"]["event_extraction"]["Contact.Communicate"][0]
    assert mention["triggers"][0]["match"] and mention["arguments"][0]["entity"]["match"]
    assert v["f1"]["event_argument"] == 1.0


def test_gold_mapped_is_preferred_when_present():
    row = dict(ROW, gold_mapped={"entities": {"Location": ["Houston"]}})
    v = gold.record_view(json.loads(json.dumps(row)))
    assert v["pred"]["entities"]["Location"][0]["match"] is True


def test_record_never_mutates_the_cached_rows(tmp_path):
    path = tmp_path / "preds.jsonl"
    path.write_text(json.dumps(ROW, ensure_ascii=False) + "\n", encoding="utf-8")
    first = gold.record(str(path), 0)
    assert "match" not in json.dumps(gold.rows_of(str(path)))
    assert gold.record(str(path), 0) == first
    assert gold.summary(str(path))["gold_key"] == "gold"
