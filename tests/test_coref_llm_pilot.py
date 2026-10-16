"""The coref LLM pilot: its prompt comes from GUIDELINES.md, and its scorer can fail."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data"))

import coref_llm_pilot as P  # noqa: E402

REC = {"input": "Ann met Bea. She said it rained.",
       "output": {"entities": {"PER": ["Ann", "Bea"]}},
       "coref_clusters": [[{"text": "Ann", "start": 0, "end": 3}, {"text": "She", "start": 13, "end": 16}]]}


def test_system_prompt_carries_the_guidelines_block():
    assert "same kind of thing" in P.SYSTEM and "single JSON object" in P.SYSTEM


def test_candidates_add_unclustered_surfaces_and_pronouns_as_distractors():
    c = P.candidates(REC)
    assert [(m["text"], m["cluster"]) for m in c] == [("Ann", 0), ("Bea", None), ("She", 0), ("it", None)]
    assert P.render(REC["input"], c) == "<m id=0>Ann</m> met <m id=1>Bea</m>. <m id=2>She</m> said <m id=3>it</m> rained."


def test_parse_keeps_valid_ids_once_and_rejects_garbage():
    assert P.parse('{"clusters": [[0, 2], [2, 3], [9]]}', 4) == [0, 5, 0, 1]
    assert P.parse("no json", 4) is None


def test_distractor_pairs_are_unscored():
    docs = [{"cands": P.candidates(REC)}]
    gold = lambda d: [m["cluster"] for m in docs[d]["cands"]]  # noqa: E731
    linked_to_distractor = {0: [0, 0, 0, 0]}
    assert P.score(linked_to_distractor, docs, gold)["all"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert P.distractor_links(linked_to_distractor, docs) == 0.8333
    assert P.score({0: [0, 1, 2, 3]}, docs, gold)["pronoun"]["recall"] == 0.0


def test_parse_takes_the_last_answer_when_the_model_corrects_itself():
    raw = '{"clusters": [[0, 1], [1, 2]]}\n\nWait, ids can\'t repeat. Corrected:\n\n{"clusters": [[0, 1]]}'
    assert P.parse(raw, 3) == [0, 0, 5]
