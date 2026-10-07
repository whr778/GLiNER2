"""Menu modes (MENU_SPEC.md): what is offered per document, and what an application menu refuses."""
import json
import sys
from pathlib import Path

import pytest

from gliner2.training import eval_metrics as E

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "train"))
sys.path.insert(0, str(ROOT / "tools"))
import train as T  # noqa: E402
import score_predictions as S  # noqa: E402
import infer  # noqa: E402

GOLD = {"entities": {"Person": ["Ann"]},
        "events": [{"event_type": "Conflict.Attack", "triggers": ["shelled"],
                    "arguments": [{"role": "Attacker", "entity": "rebels"}, {"role": "Weapon", "entity": "mortar"}]}],
        "relations": [{"works_for": {"head": "Ann", "tail": "UN"}}]}
APP = {"name": "news", "exhaustive_for": ["news_corpus"],
       "schema": {"entities": ["Person", "Location"], "events": {"Conflict.Attack": ["Attacker"], "Sport.Compete": ["Winner"]}}}
POOL = {"entities": ["Person", "Location", "Date"], "events": {"Conflict.Attack": ["Attacker"], "Life.Die": []}}


def test_parse_menu():
    assert E.parse_menu("gold") == ("gold", None) and E.parse_menu("widened:20") == ("widened", "20")
    assert E.parse_menu("app:news55") == ("app", "news55") and E.parse_menu("corpus_full") == ("corpus_full", None)
    for bad in ("widened", "app:", "full", "union"):
        with pytest.raises(ValueError):
            E.parse_menu(bad)


def test_widen_offers_dimensions_the_document_lacks():
    """An event-free document of an event corpus is still asked about events."""
    out = E.widen_from_pool({"entities": {"Person": ""}}, POOL, max_absent=10)
    assert set(out["entities"]) == {"Person", "Location", "Date"}
    assert set(out["events"]) == {"Conflict.Attack", "Life.Die"}


def test_widen_caps_absents_and_keeps_gold():
    out = E.widen_from_pool({"entities": {"Person": ""}}, POOL, max_absent=1, index=3)
    assert "Person" in out["entities"] and len(out["entities"]) == 2
    assert out == E.widen_from_pool({"entities": {"Person": ""}}, POOL, max_absent=1, index=3)


def test_project_gold_keeps_only_what_the_menu_asks():
    g = E.project_gold(GOLD, APP["schema"])
    assert g["entities"] == {"Person": ["Ann"]} and "relations" not in g
    assert g["events"][0]["arguments"] == [{"role": "Attacker", "entity": "rebels"}]


def test_build_menus_gold_mode_is_the_gold_menu():
    recs, menus, rep = E.build_menus([{"output": GOLD}], ["c"], "gold")
    assert menus == [E._schema_from_gold(GOLD)] and rep["scored"] == 1


def test_app_menu_refuses_corpora_it_is_not_exhaustive_for():
    recs, menus, rep = E.build_menus([{"output": GOLD}, {"output": GOLD}], ["news_corpus", "casie"], "app:news", app=APP)
    assert rep["scored"] == 1 and rep["refused"] == {"casie": 1}
    assert menus[0] == APP["schema"] and "relations" not in recs[0]["output"]


def test_compute_metrics_offers_the_given_menus():
    seen = []

    class M:
        def batch_extract(self, texts, schemas, **kw):
            seen.extend(schemas)
            return [{} for _ in texts]
    E.compute_metrics(M(), [("Ann shelled.", GOLD)], menus=[APP["schema"]], report=False)
    assert seen == [APP["schema"]]


def test_score_predictions_app_mode_keeps_event_free_docs_and_checks_the_app_menu():
    rows = [{"input": "a", "output": {"entities": {"Location": ["X"]}}, "gold": {}}]
    records, dropped = S.prepare(rows, "gold", APP)
    assert len(records) == 1 and dropped["no_gold"] == 0
    assert S.off_menu(records, APP) == 0
    assert S.off_menu([{"output": {}, "pred": {"entities": {"Weapon": ["gun"]}}}], APP) == 1


def test_corpus_pools_keep_roleless_types_and_skip_partial_dimensions(tmp_path):
    rec = {"input": "x", "output": {"entities": {"Person": ["A"]}, "events": [{"event_type": "Life.Die", "triggers": ["died"]}]}}
    (tmp_path / "c.train.jsonl").write_text(json.dumps(rec) + "\n")
    cfg = {"data": {"corpora": [str(tmp_path / "c")]}}
    assert T.corpus_pools(cfg, str(tmp_path / "x.yaml")) == {"c": {"entities": ["Person"], "events": {"Life.Die": []}}}
    cfg["data"]["partial_annotation"] = {"c": ["entities"]}
    assert T.corpus_pools(cfg, str(tmp_path / "x.yaml")) == {"c": {"events": {"Life.Die": []}}}


def test_check_menus_refuses_an_unknown_mode():
    with pytest.raises(ValueError):
        T.check_menus({"eval": {"menu": "union"}}, "x.yaml")


def test_infer_reads_an_application_menu_file(tmp_path):
    f = tmp_path / "news.json"
    f.write_text(json.dumps(APP))
    args = infer._parse_args(["--model", "m", "--input", "x", "--schema-json", str(f)])
    assert infer._build_schema(args) == APP["schema"]
