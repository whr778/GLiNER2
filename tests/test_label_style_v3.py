"""label_style_v3: the v3 generator refuses blank review decisions and maps that fail its gates."""
import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
import label_style_v3 as V  # noqa: E402

STYLE = {"name": "title_snake", "acronyms": [], "words": []}


def test_gate_passes_a_styled_closed_map():
    V.gate("entities", {"person": "Person", "LOC": "Location"}, Counter({"person": 1, "LOC": 1, "Person": 3}), STYLE)


@pytest.mark.parametrize("m,uses", [
    ({"person": "person"}, Counter({"person": 1})),                                   # target not styled
    ({"LOC": "Loc", "Loc": "Location"}, Counter({"LOC": 1, "Loc": 1})),              # not closed
    ({"LOC": "Location"}, Counter({"LOC": 1, "street address": 1})),                 # a label left unstyled
])
def test_gate_refuses(m, uses):
    with pytest.raises(SystemExit):
        V.gate("entities", m, uses, STYLE)


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def test_review_refuses_a_blank_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "REVIEW", tmp_path)
    _write(tmp_path / "squashed_segments.tsv", [{"category": "events", "segment": "judgecourt", "decision": ""}])
    _write(tmp_path / "suspect_groups.tsv", [{"category": "events", "spelling": "proposal", "decision": "merge"}])
    with pytest.raises(SystemExit):
        V.review()


def test_review_reads_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "REVIEW", tmp_path)
    _write(tmp_path / "squashed_segments.tsv", [{"category": "events", "segment": "judgecourt", "decision": "Judge_Court"},
                                               {"category": "events", "segment": "detainee", "decision": "one word"}])
    _write(tmp_path / "suspect_groups.tsv", [{"category": "events", "spelling": "proposal",
                                             "decision": "keep apart: Marriage_Proposal"}])
    _write(tmp_path / "all_caps_tokens.tsv", [{"token": "LAW", "decision": "word"}])
    _write(tmp_path / "structure_dot_pairs.tsv", [{"group": "hotel.name | hotel_name", "decision": ""}])
    squashed, apart, verdicts, winners = V.review()
    assert squashed["events"] == {"judgecourt": "Judge_Court", "detainee": "Detainee"}
    assert apart == {("events", "proposal"): "Marriage_Proposal"} and verdicts == {"LAW": False} and winners == {}
