"""The near-dup gate catches a re-published story that exact-match dedup misses."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data"))

from near_dup_eval import containment

STORY = "The council voted on Tuesday to close the old bridge after inspectors found deep cracks in two of its supports."


def test_same_story_with_a_new_header_is_contained():
    assert containment(["BREAKING NEWS: " + STORY], [STORY + " More to follow."])[0] > 0.8


def test_unrelated_text_is_not():
    other = "Shares of the airline rose sharply in early trading as fuel prices fell for a third straight week."
    assert containment([STORY], [other])[0] == 0.0
