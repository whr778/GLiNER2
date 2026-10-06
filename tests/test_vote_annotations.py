"""vote_annotations: coreferent event identity, 2-of-3 voting, event-level agreement."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "data"))
import vote_annotations as V  # noqa: E402

A = {"event_type": "Conflict.Attack", "triggers": ["attacked", "the assault"],
     "arguments": [{"role": "Attacker", "entity": "Rebels"}, {"role": "Target", "entity": "base"}]}
A2 = {"event_type": "Conflict.Attack", "triggers": ["assault"], "arguments": [{"role": "Attacker", "entity": "Rebels"}]}
DIE = {"event_type": "Life.Die", "triggers": ["killed"], "arguments": []}


def test_shared_or_nested_mention_is_the_same_event_but_type_must_match():
    assert V.same_event(A, A2)                                  # "assault" inside "the assault"
    assert not V.same_event(A, {**A2, "event_type": "Life.Injure"})


def test_two_of_three_keeps_the_event_and_only_majority_arguments():
    out = V.vote_events([[A], [A2], [DIE]], need=2)
    assert len(out) == 1 and out[0]["event_type"] == "Conflict.Attack" and out[0]["votes"] == 2
    assert out[0]["arguments"] == [{"role": "Attacker", "entity": "Rebels"}]   # Target given once


def test_event_found_in_one_run_is_dropped():
    assert V.vote_events([[DIE], [], []], need=2) == []


def test_agreement_is_one_on_identical_and_lower_on_a_missing_event():
    assert V.event_agreement([{"events": [A, DIE]}], [{"events": [A, DIE]}])["events_f1"] == 1.0
    assert V.event_agreement([{"events": [A, DIE]}], [{"events": [A]}])["events_f1"] < 1.0


def test_structures_vote_by_name_and_anchor_value():
    meta = {"person_profile": {"mode": "natural", "anchor": "name"}}
    run = lambda role: {"json_structures": [{"person_profile": {"name": "Elena Krylova", "role": role}}], "record_metadata": meta}
    out, md = V.vote_structures([run("spokeswoman"), run("spokeswoman"), run("aide")], need=2)
    assert out == [{"person_profile": {"name": "Elena Krylova", "role": "spokeswoman"}}] and md == meta
    assert V.vote_structures([run("x"), {"json_structures": []}, {}], need=2) == ([], {})
