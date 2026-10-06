"""validate._events keeps EVERY coreferent trigger of one event (the ccnews_english_v2 recipe).

The record format has carried a trigger LIST since 4436653; until 2026-10-06 the annotator was
asked for one trigger and this function wrapped it as [trigger].
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "data" / "synthetic"))
from validate import _events  # noqa: E402

TEXT = "Rebels attacked the base on Monday. The assault killed four soldiers."


def test_trigger_list_keeps_every_verbatim_mention():
    stats = Counter()
    out = _events(TEXT, [{"event_type": "Conflict.Attack", "triggers": ["attacked", "assault", "the raid"],
                          "arguments": [{"role": "Attacker", "entity": "Rebels"}]}], stats)
    assert out[0]["triggers"] == ["attacked", "assault"]           # "the raid" is not in the text
    assert stats["coreferent_triggers"] == 1 and stats["triggers_dropped"] == 1


def test_legacy_single_trigger_still_parses():
    out = _events(TEXT, [{"event_type": "Conflict.Attack", "trigger": "attacked", "arguments": []}], Counter())
    assert out[0]["triggers"] == ["attacked"]


def test_event_with_no_verbatim_trigger_is_dropped():
    stats = Counter()
    assert _events(TEXT, [{"event_type": "Conflict.Attack", "triggers": ["bombed"], "arguments": []}], stats) == []
    assert stats["events_dropped"] == 1
