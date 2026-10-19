"""Converter --offsets (TODO #26): event gold as {text, start, end} in the CLEANED text dumps_record writes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data"))
import convert_casie as C  # noqa: E402
import convert_cmnee as M  # noqa: E402
import convert_wikievents as W  # noqa: E402
from _split import clean_text  # noqa: E402


def covers(text, span):
    return isinstance(span, dict) and text[span["start"]:span["end"]] == span["text"]


def test_casie_recovers_a_one_char_late_offset_only_on_an_exact_match():
    text = "Hackers held the data to ransom."
    exact = C._offset_gold(text, clean_text(text), "held", 8, 12)
    late = C._offset_gold(text, clean_text(text), "held", 9, 13)          # CASIE's off-by-one: starts one late
    assert covers(clean_text(text), exact) and exact == late
    assert C._offset_gold(text, clean_text(text), "absent", 3, 9) == "absent"   # nothing nearby reads it: a string


def test_cmnee_row_offsets_land_in_the_cleaned_text():
    row = {"text": "6月13日，美国海军执行演习。美国海军",
           "event_list": [{"event_type": "Manoeuvre", "trigger": {"text": "执行", "offset": [10, 12]},
                           "arguments": [{"role": "Subject", "text": "美国海军", "offset": [6, 10]}]}]}
    rec = M.convert_row(row, offsets=True)
    ev = rec["output"]["events"][0]
    text = clean_text(rec["input"])
    assert covers(text, ev["triggers"][0]) and covers(text, ev["arguments"][0]["entity"])
    assert ev["arguments"][0]["entity"]["start"] == 6                     # the annotated one, not the second copy
    assert M.convert_row(row)["output"]["events"][0]["arguments"][0]["entity"] == "美国海军"


def test_wikievents_token_spans_become_character_offsets():
    tokens = ["Bob", "hit", "Ann", ".", "Ann", "left"]
    row = {"text": "Bob hit Ann. Ann left", "tokens": tokens,
           "entity_mentions": [{"id": "m1", "start": 4, "end": 5, "entity_type": "PER", "text": "Ann"}],
           "event_mentions": [{"event_type": "Leave", "trigger": {"text": "left", "start": 5, "end": 6},
                               "arguments": [{"role": "Agent", "text": "Ann", "entity_id": "m1"}]}]}
    rec = W.convert_row(row, offsets=True)
    ev = rec["output"]["events"][0]
    assert covers(clean_text(rec["input"]), ev["arguments"][0]["entity"])
    assert ev["arguments"][0]["entity"]["start"] == 13                    # the SECOND "Ann" (token 4), not the first
    assert ev["triggers"][0] == {"text": "left", "start": 17, "end": 21}
