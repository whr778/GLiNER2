"""Unit tests for the LEVEN converter's parsing logic (no network)."""

import importlib.util
from pathlib import Path

_CONV = Path(__file__).resolve().parents[1] / "tools" / "data" / "convert_leven.py"


def _load():
    spec = importlib.util.spec_from_file_location("convert_leven", _CONV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


leven = _load()

# Two-sentence Chinese doc. Offsets are token offsets within each sentence;
# the second sentence's trigger checks the cross-sentence flat-offset math.
TRAIN_ROW = {
    "content": [
        {"sentence": "被告人为获利诱骗", "tokens": ["被告人", "为", "获利", "诱骗"]},
        {"sentence": "并采用暴力", "tokens": ["并", "采用", "暴力"]},
    ],
    "events": [
        {"type": "获利", "mention": [{"trigger_word": "获利", "sent_id": 0, "offset": [2, 3]}]},
        {"type": "暴力", "mention": [{"trigger_word": "暴力", "sent_id": 1, "offset": [2, 3]}]},
    ],
}

# test.jsonl ships `candidates` instead of `events` -> nothing to emit.
TEST_ROW = {
    "content": [{"sentence": "被告人为获利", "tokens": ["被告人", "为", "获利"]}],
    "candidates": [{"trigger_word": "获利", "sent_id": 0, "offset": [2, 3]}],
}


def test_tokens_join_without_spaces_and_offsets_reconstruct_triggers():
    rec = leven.convert_row(TRAIN_ROW)
    # Chinese tokens concatenated with no separator across both sentences.
    assert rec["input"] == "被告人为获利诱骗并采用暴力"
    events = rec["output"]["events"]
    assert events[0] == {"event_type": "获利", "trigger": "获利", "arguments": []}
    # second sentence trigger resolved via the flat cross-sentence offset
    assert events[1] == {"event_type": "暴力", "trigger": "暴力", "arguments": []}
    assert all(e["trigger"] in rec["input"] for e in events)


def test_test_split_without_events_is_dropped():
    assert leven.convert_row(TEST_ROW) is None
