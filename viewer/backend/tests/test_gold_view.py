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


def test_picker_lists_only_infer_outputs_and_flags_files_without_spans(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "spans.jsonl").write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    bare = dict(ROW, output={"entities": {"Person": ["Ed Smith"]}}, gold_mapped={})
    (tmp_path / "bare.jsonl").write_text(json.dumps(bare) + "\n", encoding="utf-8")
    (tmp_path / "corpus.train.jsonl").write_text(json.dumps({"input": TEXT, "output": {}}) + "\n", encoding="utf-8")
    (tmp_path / "broken.jsonl").write_text("{not json\n", encoding="utf-8")
    got = {f["name"]: (f["spans"], f["gold_mapped"]) for f in gold.list_files(str(tmp_path))}
    assert got == {"sub/spans.jsonl": (True, False), "bare.jsonl": (False, True)}


def test_arguments_link_to_their_entity_on_both_sides_and_scores_do_not_move():
    row = json.loads(json.dumps(ROW))
    row["output"]["event_extraction"]["Contact.Communicate"][0]["arguments"].append(
        {"role": "Place", "entity": {"text": "plants", "start": 27, "end": 33}})
    v = gold.record_view(row)
    pred_arg, derived_arg = v["pred"]["event_extraction"]["Contact.Communicate"][0]["arguments"]
    assert pred_arg["entity_id"] == v["pred"]["entities"]["Person"][0]["id"]
    (d,) = v["pred"]["derived_entities"]
    assert (d["text"], d["id"]) == ("plants", derived_arg["entity_id"])
    gold_arg = v["gold"]["event_extraction"]["Contact.Communicate"][0]["arguments"][0]
    assert gold_arg["entity_id"] == v["gold"]["entities"]["Person"][0]["id"]
    assert v["f1"] == gold._f1(row["gold"], row["output"])



def test_a_training_data_row_shows_its_annotations(tmp_path):
    """A train/val/test row has no gold key: its output IS the gold (bare surfaces), with no prediction.
    Read as a prediction, its surfaces reached argument_links as strings and the record endpoint 500'd."""
    row = {"input": TEXT, "output": {"entities": {"Person": ["Ed Smith"]},
                                     "events": [{"event_type": "Contact.Communicate", "triggers": ["said"],
                                                 "arguments": [{"role": "Speaker", "entity": "Ed Smith"}]}]}}
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert gold.summary(str(p))["data_file"]
    r = gold.record(str(p), 0)
    assert r["data_row"] and not r["has_gold"] and r["diffs"] == []
    ent = r["pred"]["entities"]["Person"][0]
    assert (ent["text"], ent["start"], ent["end"]) == ("Ed Smith", 0, 8)
    mention = r["pred"]["event_extraction"]["Contact.Communicate"][0]
    assert mention["triggers"][0]["start"] == 9 and "trigger_key" not in mention


def _m(text, start):
    return {"text": text, "start": start, "end": start + len(text)}


def test_coref_merge_marks_the_smaller_side_and_pairs_score():
    a1, a2, a3, b1 = _m("Ann", 0), _m("she", 10), _m("Ann", 20), _m("Bea", 30)
    gold_c = [[a1, a2, a3], [b1, _m("Bea", 40)]]
    pred_c = gold.coref_view([[a1, a2, a3, b1]])
    gold._mark_coref(pred_c, gold.coref_view(gold_c))
    assert [m["match"] for m in pred_c[0]] == [True, True, True, False]
    assert gold.coref_view([[a1]]) == []
    assert len(gold.coref_pairs(gold_c)) == 4


def test_data_row_carries_its_coref_clusters(tmp_path):
    row = {"input": "Ann said she left.", "output": {},
           "coref_clusters": [[_m("Ann", 0), _m("she", 9)], [_m("left", 13)]]}
    p = tmp_path / "coref.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    r = gold.record(str(p), 0)
    assert [[m["text"] for m in c] for c in r["pred"]["coref_clusters"]] == [["Ann", "she"]]
