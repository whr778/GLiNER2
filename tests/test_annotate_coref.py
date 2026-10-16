"""Candidate building and windowing for the bought coref corpus."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data"))

import annotate_coref as A


def test_pronouns_match_lower_and_capitalized_but_never_all_caps_or_inside_names():
    text = 'He said Toys "R" Us and the US told us.'
    spans = A.mention_spans(text, {'Toys "R" Us'}, [])
    assert [text[s:e] for s, e in spans] == ["He", 'Toys "R" Us', "us"]


def test_windows_cut_at_sentence_ends_and_cover_the_text():
    text = "Aa. Bb. Cc. Dd."
    ws = A.windows(text, 8)
    assert "".join(text[s:e] for s, e in ws) == text and all(e - s <= 8 for s, e in ws)
    assert ws[0] == (0, 8) and text[ws[1][0]] == "C"


def test_units_rebase_candidates_per_window_and_keep_gold_mentions():
    rec = {"input": "Ann ran. She won.", "output": {"entities": {"person": ["Ann"]}},
           "coref_clusters": [[{"text": "Ann", "start": 0, "end": 3}, {"text": "She", "start": 9, "end": 12}]]}
    us = A.units(rec, 9)
    assert [(u["offset"], [(m["start"], m["text"]) for m in u["cands"]]) for u in us] == [(0, [(0, "Ann")]), (9, [(0, "She")])]


def test_clusters_keep_singletons_as_negatives():
    cands = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    assert A.clusters([0, 9, 0], cands) == [[{"text": "a"}, {"text": "c"}], [{"text": "b"}]]


def test_vote_keeps_majority_pairs_and_closes_transitively():
    import annotate_coref as A
    # draws agree 0~1 twice; 1~2 twice; 0~2 never directly -> one cluster by closure; 3 alone
    draws = [[0, 0, 5, 7], [0, 0, 0, 7], [4, 5, 5, 6]]
    labels = A.vote(draws, 4)
    assert labels[0] == labels[1] == labels[2] and labels[3] != labels[0]


def test_vote_drops_a_pair_only_one_draw_links():
    import annotate_coref as A
    labels = A.vote([[0, 0, 2], [0, 1, 2], [0, 1, 2]], 3)
    assert len(set(labels)) == 3
