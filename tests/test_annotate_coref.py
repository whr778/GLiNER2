"""Candidate building and windowing for the bought coref corpus."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data"))

import annotate_coref as A  # noqa: E402


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
