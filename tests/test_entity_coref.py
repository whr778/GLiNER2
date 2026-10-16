"""Entity coref at inference: candidate mentions and best-antecedent clustering."""
import torch

from gliner2.inference.entity_coref import best_antecedent, candidate_mentions


def test_candidates_are_entity_spans_then_pronouns_without_overlap():
    text = "Ann Lee said she met Bob. He agreed with Lee."
    ents = {"Person": [{"text": "Ann Lee", "start": 0, "end": 7}, {"text": "Ann", "start": 0, "end": 3},
                       {"text": "Bob", "start": 21, "end": 24}]}
    got = [m["text"] for m in candidate_mentions(text, ents)]
    assert got == ["Ann Lee", "she", "Bob", "He"]


def test_best_antecedent_links_each_mention_once_above_threshold():
    logits = torch.tensor([[0.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0],
                           [1.0, 3.0, 0.0]])
    assert best_antecedent(logits, 0.5) == [0, 0, 0]
    assert best_antecedent(logits, 2.5) == [0, 1, 1]
    assert best_antecedent(logits, 9.0) == [0, 1, 2]
