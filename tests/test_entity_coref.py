"""Entity coref at inference: candidate mentions, event argument mentions, and in-engine clustering."""
import torch

from gliner2.inference.entity_coref import best_antecedent, candidate_mentions, resolve_from_states
from gliner2.models.boundary.mention_link import MentionLink


def test_candidates_are_entity_spans_then_pronouns_without_overlap():
    text = "Ann Lee said she met Bob. He agreed with Lee."
    ents = {"Person": [{"text": "Ann Lee", "start": 0, "end": 7}, {"text": "Ann", "start": 0, "end": 3},
                       {"text": "Bob", "start": 21, "end": 24}]}
    got = [m["text"] for m in candidate_mentions(text, ents)]
    assert got == ["Ann Lee", "she", "Bob", "He"]


def test_candidates_include_non_overlapping_event_arguments_and_let_entities_win_on_overlap():
    text = "President Joe Biden met the envoy in Paris. He thanked her."
    # Entity has "Joe Biden" (10..19); event argument has overlapping "President Joe Biden" (0..19)
    # and non-overlapping "the envoy" (24..33) and "Paris" (37..42).
    ents = {"Person": [{"text": "Joe Biden", "start": 10, "end": 19}]}
    events = {
        "Meet": [{
            "triggers": [{"text": "met", "start": 20, "end": 23}],
            "arguments": [
                {"role": "Participant", "entity": {"text": "President Joe Biden", "start": 0, "end": 19}},
                {"role": "Participant", "entity": {"text": "the envoy", "start": 24, "end": 33}},
                {"role": "Place", "entity": {"text": "Paris", "start": 37, "end": 42}},
            ],
        }]
    }
    got = [m["text"] for m in candidate_mentions(text, ents, events)]
    # "Joe Biden" wins over overlapping "President Joe Biden" (Case C 1A), while
    # "the envoy" and "Paris" are added alongside pronouns "He" and "her".
    assert got == ["Joe Biden", "the envoy", "Paris", "He", "her"]


def test_best_antecedent_links_each_mention_once_above_threshold():
    logits = torch.tensor([[0.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0],
                           [1.0, 3.0, 0.0]])
    assert best_antecedent(logits, 0.5) == [0, 0, 0]
    assert best_antecedent(logits, 2.5) == [0, 1, 1]
    assert best_antecedent(logits, 9.0) == [0, 1, 2]


def test_resolve_from_states_clusters_identical_word_vectors():
    # 4 words: "Ann" (0..3), "met" (4..7), "Bob" (8..11), "she" (13..16)
    starts = [0, 4, 8, 13]
    ends = [3, 7, 11, 16]
    # Give word 0 ("Ann") and word 3 ("she") identical states along axis 0, orthogonal to word 2 ("Bob")
    doc_states = torch.tensor([
        [4.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
    ])
    link = MentionLink(4, 4)
    with torch.no_grad():
        link.w.weight.copy_(torch.eye(4))
        link.w.bias.zero_()
    mentions = [
        {"text": "Ann", "start": 0, "end": 3},
        {"text": "Bob", "start": 8, "end": 11},
        {"text": "she", "start": 13, "end": 16},
    ]
    clusters = resolve_from_states(link, doc_states, starts, ends, mentions, threshold=1.0)
    multi = [c for c in clusters if len(c) > 1]
    assert len(multi) == 1
    assert [m["text"] for m in multi[0]] == ["Ann", "she"]
