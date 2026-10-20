"""Training validation keeps offset-anchored event gold ({text, start, end}) and still drops misplaced gold.

Until 2026-10-20 the InputExample validator required trigger and argument values to be strings, so every offset-gold
record failed to parse and was DROPPED: eb22c trained with 0 of 200 cmnee_ner_off, 3 of 200 casie_off and 11 of 199
wikievents_off records surviving, i.e. without CMNEE, CASIE and WikiEvents events. Validated records are replaced by
InputExample.to_dict(), so the offsets must also survive that rebuild.
"""
from gliner2.training.data import DataLoader_Factory

TEXT = "Ann met Bob. Ann left the city."


def _record(trigger, entity, role="Agent"):
    return {"input": TEXT, "output": {"events": [{"event_type": "Leave", "triggers": [trigger],
                                                   "arguments": [{"role": role, "entity": entity}]}]}}


def _kept(records):
    valid, _ = DataLoader_Factory._validate_records(records)
    return [records[i] for i in valid]


def test_offset_gold_is_kept_and_survives_the_rebuild():
    trig, ent = {"text": "left", "start": 17, "end": 21}, {"text": "Ann", "start": 13, "end": 16}
    kept = _kept([_record(trig, ent)])
    assert len(kept) == 1
    ev = kept[0]["output"]["events"][0]
    assert ev["triggers"] == [trig] and ev["arguments"][0]["entity"] == ent   # still offsets, still THIS occurrence


def test_misplaced_offset_gold_is_dropped():
    wrong_trigger = {"text": "left", "start": 3, "end": 7}          # " met" is there, not "left"
    assert _kept([_record(wrong_trigger, "Ann")]) == []             # no trigger left: the event, then the record, go
    wrong_arg = {"text": "Ann", "start": 8, "end": 11}              # "Bob" is there
    kept = _kept([_record({"text": "left", "start": 17, "end": 21}, wrong_arg)])
    assert kept and kept[0]["output"]["events"][0]["arguments"] == []   # the argument is dropped, the event kept


def test_the_same_text_at_two_offsets_is_two_arguments_but_a_repeat_is_one():
    first, second = {"text": "Ann", "start": 0, "end": 3}, {"text": "Ann", "start": 13, "end": 16}
    rec = _record({"text": "left", "start": 17, "end": 21}, first)
    rec["output"]["events"][0]["arguments"] += [{"role": "Agent", "entity": second}, {"role": "Agent", "entity": dict(first)}]
    args = _kept([rec])[0]["output"]["events"][0]["arguments"]
    assert [a["entity"] for a in args] == [first, second]


def test_string_gold_is_unchanged():
    kept = _kept([_record("left", "Ann")])
    assert kept[0]["output"]["events"][0] == {"event_type": "Leave", "triggers": ["left"],
                                               "arguments": [{"role": "Agent", "entity": "Ann"}]}


def test_a_corpus_losing_most_of_its_records_is_refused_loudly():
    import pytest
    bad = [dict(_record({"text": "left", "start": 3, "end": 7}, "Ann"), _corpus="broken") for _ in range(3)]
    good = [dict(_record("left", "Ann"), _corpus="fine") for _ in range(3)]
    with pytest.raises(ValueError, match="broken"):
        DataLoader_Factory.load(bad + good, shuffle=False, validate=True)
    assert len(DataLoader_Factory.load(good, shuffle=False, validate=True)) == 3
