"""convert_corefud / convert_gap: mentions land on their exact surfaces, clusters group by entity id."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data"))
import convert_corefud as C  # noqa: E402
import convert_gap as G  # noqa: E402

CONLLU = """# newdoc id = GUM_news_test
# meta::genre = news
# text = Ann met Eye-Tracking fans.
1\tAnn\t_\t_\t_\t_\t_\t_\t_\tEntity=(e1-person-1-)
2\tmet\t_\t_\t_\t_\t_\t_\t_\t_
3\tEye\t_\t_\t_\t_\t_\t_\t_\tEntity=(e2-abstract-1-(e3-object-1-)|SpaceAfter=No
4\t-\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No
5\tTracking\t_\t_\t_\t_\t_\t_\t_\tEntity=e2)
6\tfans\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No
7\t.\t_\t_\t_\t_\t_\t_\t_\t_

# text = It's her.
1-2\tIt's\t_\t_\t_\t_\t_\t_\t_\t_
1\tIt\t_\t_\t_\t_\t_\t_\t_\tEntity=(e2-abstract-1-)
2\t's\t_\t_\t_\t_\t_\t_\t_\t_
3\ther\t_\t_\t_\t_\t_\t_\t_\tEntity=(e1-person-1-)|SpaceAfter=No
4\t.\t_\t_\t_\t_\t_\t_\t_\t_
"""


def test_corefud_nested_single_and_contraction_mentions(tmp_path):
    f = tmp_path / "x-corefud-train.conllu"
    f.write_text(CONLLU, encoding="utf-8")
    (doc,) = C.read_docs(f)
    assert doc["text"].strip() == "Ann met Eye-Tracking fans. It's her."
    rec = C.to_record(doc, "test", {"news": "CC-BY-2.5"}, {"kept": 0, "dropped": 0})
    clusters = sorted([[m["text"] for m in c] for c in rec["coref_clusters"]])
    assert clusters == [["Ann", "her"], ["Eye"], ["Eye-Tracking", "It"]]
    assert all(rec["input"][m["start"]:m["end"]] == m["text"] for c in rec["coref_clusters"] for m in c)
    assert rec["provenance"]["license"] == "CC-BY-2.5" and rec["output"] == {}


def test_gap_pronoun_joins_the_coreferent_name_and_the_other_is_a_singleton():
    text = "Ann met Bea. She left."
    row = {"ID": "t-1", "Text": text, "URL": "u", "Pronoun": "She", "Pronoun-offset": 13,
           "A": "Ann", "A-offset": 0, "A-coref": False, "B": "Bea", "B-offset": 8, "B-coref": True}
    rec = G.to_record(pd.Series(row), {"kept": 0, "dropped": 0})
    assert [[m["text"] for m in c] for c in rec["coref_clusters"]] == [["Ann"], ["Bea", "She"]]
    row.update({"B-coref": False})
    assert len(G.to_record(pd.Series(row), {"kept": 0, "dropped": 0})["coref_clusters"]) == 3
