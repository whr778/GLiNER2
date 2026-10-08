"""tools/train/score_predictions.py: scoring an infer.py file must equal scoring in compute_metrics."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
import score_predictions as S  # noqa: E402
import train as T  # noqa: E402

from gliner2.training.eval_metrics import compute_metrics, score_predictions  # noqa: E402

GOLD = {"entities": {"person": ["Olaf Scholz"], "location": ["Berlin"]},
        "events": [{"event_type": "Contact.Meet", "triggers": ["met"],
                    "arguments": [{"role": "Participant", "entity": "Olaf Scholz"}]}]}
PRED = {"entities": {"person": [{"text": "Olaf Scholz", "start": 0, "end": 11, "confidence": 0.9}], "location": []},
        "event_extraction": {"Contact.Meet": [{"triggers": ["met"], "arguments": [{"role": "Participant", "entity": "Olaf Scholz"}]}]}}


class _Model:
    """batch_extract returns fixed predictions, so compute_metrics' scoring half is what differs."""
    def __init__(self, preds):
        self.preds = preds

    def batch_extract(self, texts, schemas, **kw):
        return self.preds[: len(texts)]


def test_score_predictions_equals_compute_metrics():
    golds, preds = [GOLD, {"entities": {"person": ["Ann"]}}], [PRED, {"entities": {"person": ["Bob"]}}]
    data = [("Olaf Scholz met in Berlin.", golds[0]), ("Ann", golds[1])]
    assert compute_metrics(_Model(preds), data, report=False) == score_predictions(golds, preds, report=False)


def test_gold_key_auto_mapped_and_mixed():
    assert S.gold_key([{"gold": {}, "gold_mapped": {}}], "auto") == "gold_mapped"
    assert S.gold_key([{"gold": {}}], "auto") == "gold"
    with pytest.raises(SystemExit):
        S.gold_key([{"gold": {}, "gold_mapped": {}}, {"gold": {}}], "auto")


def test_prepare_drops_no_gold_and_exact_duplicates():
    rows = [{"input": "a", "output": PRED, "gold": GOLD}, {"input": "a", "output": PRED, "gold": GOLD},
            {"input": "b", "output": {}, "gold": {}}]
    records, dropped = S.prepare(rows, "gold")
    assert len(records) == 1 and dropped == {"no_gold": 1, "duplicate": 1}


def test_off_menu_counts_predictions_outside_the_gold_menu():
    on = [{"output": GOLD, "pred": PRED}]
    off = [{"output": GOLD, "pred": {"entities": {"weapon": ["gun"]}}}]
    assert S.off_menu(on) == 0 and S.off_menu(off) == 1


def test_by_language_block_projects_and_names_folded():
    block = T.by_language_block({"eng": {"eval_x": 1.0, "eval_y_classification_report": "...", "other": 2}}, {"kor": 1})
    assert block == {"by_language": {"eng": {"eval_x": 1.0}}, "by_language_folded": {"kor": 1}}


def test_cli_end_to_end(tmp_path):
    p = tmp_path / "preds.jsonl"
    p.write_text(json.dumps({"input": "Olaf Scholz met in Berlin.", "output": PRED, "gold": GOLD}) + "\n", encoding="utf-8")
    out = tmp_path / "m.json"
    S.main(["--predictions", str(p), "--out", str(out)])
    m = json.loads(out.read_text())
    assert m["eval_entity_strict_micro_f1"] == score_predictions([GOLD], [PRED], report=False)["eval_entity_strict_micro_f1"]
    assert m["scored_from"]["records"] == 1


def test_card_is_the_metrics_without_model_card_prose():
    m = score_predictions([GOLD], [PRED], report=False)
    table = S.metrics_table(m, "All records (preds.jsonl, gold `gold`, menu app:news55.json)")
    assert "own gold" not in table and "eval.py" not in table and "menu app:news55.json" in table
    assert f"| entity (strict -> relaxed) | {m['eval_entity_strict_micro_precision']:.3f} -> " in table
    assert "| event_argument (strict -> relaxed) |" in table and "| Arg-C (OneIE) |" in table


def test_span_and_confidence_output_scores_like_plain_output():
    """infer.py --include-spans --include-confidence wraps argument entities as {"text", ...} and
    classification answers as {"label", "confidence"}: Arg-I/Arg-C read 0.000 and every answer
    scored as a structure false positive (2026-10-08)."""
    gold = dict(GOLD, classifications=[{"task": "genre", "labels": ["news", "blog"], "true_label": ["news"]}])
    plain = dict(PRED, genre="news")
    spans = {"entities": PRED["entities"], "genre": {"label": "news", "confidence": 0.9},
             "event_extraction": {"Contact.Meet": [{"triggers": [{"text": "met", "start": 12, "end": 15}],
                                                    "arguments": [{"role": "Participant", "entity": {
                                                        "text": "Olaf Scholz", "start": 0, "end": 11, "confidence": 0.8}}]}]}}
    a, b = score_predictions([gold], [plain], report=False), score_predictions([gold], [spans], report=False)
    assert a["eval_argc_external_micro_f1"] == 1.0 and a == b
