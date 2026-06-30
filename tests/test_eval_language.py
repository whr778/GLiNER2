"""Tests for language detection and per-language blind-test evaluation in train.py."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_TRAIN_PY = Path(__file__).resolve().parents[1] / "tools" / "train" / "train.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_cli", _TRAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load_train_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_lid_ctx(lang2="en", confidence=0.95, alpha3="eng"):
    """Context manager that stubs lumi_language_id + langcodes in sys.modules."""
    mock_lumi = MagicMock()
    mock_lumi.detect_language.return_value = (lang2, confidence)

    mock_lang_obj = MagicMock()
    mock_lang_obj.to_alpha3.return_value = alpha3
    mock_langcodes = MagicMock()
    mock_langcodes.Language.get.return_value = mock_lang_obj

    return patch.dict(sys.modules, {"lumi_language_id": mock_lumi, "langcodes": mock_langcodes})


def _make_records(langs):
    """Minimal records pre-annotated with _lang."""
    return [{"input": f"text in {lang}", "output": {}, "_lang": lang} for lang in langs]


# ---------------------------------------------------------------------------
# _detect_lang
# ---------------------------------------------------------------------------

def test_detect_lang_english():
    with _mock_lid_ctx(lang2="en", alpha3="eng"):
        assert train._detect_lang("Hello world") == "eng"


def test_detect_lang_french():
    with _mock_lid_ctx(lang2="fr", alpha3="fra"):
        assert train._detect_lang("Bonjour le monde") == "fra"


def test_detect_lang_und_passthrough():
    """LID returning 'und' (low confidence) flows through to langcodes as-is."""
    with _mock_lid_ctx(lang2="und", alpha3="und"):
        assert train._detect_lang("gibberish") == "und"


def test_detect_lang_passes_text_to_lid():
    mock_lumi = MagicMock()
    mock_lumi.detect_language.return_value = ("en", 0.99)
    mock_lang_obj = MagicMock()
    mock_lang_obj.to_alpha3.return_value = "eng"
    mock_langcodes = MagicMock()
    mock_langcodes.Language.get.return_value = mock_lang_obj

    with patch.dict(sys.modules, {"lumi_language_id": mock_lumi, "langcodes": mock_langcodes}):
        train._detect_lang("specific test input")

    mock_lumi.detect_language.assert_called_once_with("specific test input")


def test_detect_lang_passes_lang2_to_langcodes():
    mock_lumi = MagicMock()
    mock_lumi.detect_language.return_value = ("ja", 0.97)
    mock_lang_obj = MagicMock()
    mock_lang_obj.to_alpha3.return_value = "jpn"
    mock_langcodes = MagicMock()
    mock_langcodes.Language.get.return_value = mock_lang_obj

    with patch.dict(sys.modules, {"lumi_language_id": mock_lumi, "langcodes": mock_langcodes}):
        result = train._detect_lang("日本語のテキスト")

    mock_langcodes.Language.get.assert_called_once_with("ja")
    assert result == "jpn"


# ---------------------------------------------------------------------------
# _annotate_languages
# ---------------------------------------------------------------------------

def test_annotate_uses_input_field():
    records = [{"input": "Hello world", "output": {}}]
    with patch.object(train, "_detect_lang", return_value="eng"):
        train._annotate_languages(records)
    assert records[0]["_lang"] == "eng"


def test_annotate_falls_back_to_text_field():
    records = [{"text": "Bonjour le monde", "schema": {}}]
    with patch.object(train, "_detect_lang", return_value="fra"):
        train._annotate_languages(records)
    assert records[0]["_lang"] == "fra"


def test_annotate_prefers_input_over_text():
    """'input' takes precedence over 'text' when both are present."""
    records = [{"input": "English text", "text": "French text", "output": {}}]
    with patch.object(train, "_detect_lang", return_value="eng") as mock_fn:
        train._annotate_languages(records)
    mock_fn.assert_called_once_with("English text")


def test_annotate_empty_text_becomes_und():
    records = [{"input": "", "output": {}}]
    with patch.object(train, "_detect_lang", side_effect=AssertionError("must not be called")):
        train._annotate_languages(records)
    assert records[0]["_lang"] == "und"


def test_annotate_whitespace_only_becomes_und():
    records = [{"input": "   ", "output": {}}]
    with patch.object(train, "_detect_lang", side_effect=AssertionError("must not be called")):
        train._annotate_languages(records)
    assert records[0]["_lang"] == "und"


def test_annotate_missing_text_field_becomes_und():
    records = [{"output": {}}]
    with patch.object(train, "_detect_lang", side_effect=AssertionError("must not be called")):
        train._annotate_languages(records)
    assert records[0]["_lang"] == "und"


def test_annotate_modifies_in_place_and_returns_same_list():
    records = [{"input": "Hello", "output": {}}, {"input": "Bonjour", "output": {}}]
    with patch.object(train, "_detect_lang", side_effect=["eng", "fra"]):
        result = train._annotate_languages(records)
    assert result is records
    assert records[0]["_lang"] == "eng"
    assert records[1]["_lang"] == "fra"


def test_annotate_mixed_record_formats():
    """Both {input/output} and {text/schema} record shapes are handled."""
    records = [
        {"input": "Hello", "output": {}},
        {"text": "Hola", "schema": {}},
    ]
    with patch.object(train, "_detect_lang", side_effect=["eng", "spa"]):
        train._annotate_languages(records)
    assert records[0]["_lang"] == "eng"
    assert records[1]["_lang"] == "spa"


def test_annotate_lang_survives_list_shuffle():
    """_lang stored on each dict stays with its record through a shuffle."""
    import random
    records = [{"input": f"text {i}", "output": {}} for i in range(10)]
    expected = [f"l{i:02d}" for i in range(10)]
    with patch.object(train, "_detect_lang", side_effect=list(expected)):
        train._annotate_languages(records)
    random.shuffle(records)
    for rec in records:
        idx = int(rec["input"].split()[-1])
        assert rec["_lang"] == f"l{idx:02d}"


def test_annotate_prints_progress(capsys):
    records = [{"input": "hello", "output": {}}]
    with patch.object(train, "_detect_lang", return_value="eng"):
        train._annotate_languages(records)
    assert "[lang]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _print_blind_test
# ---------------------------------------------------------------------------

def test_print_blind_test_empty_does_nothing(capsys):
    train._print_blind_test({})
    assert capsys.readouterr().out == ""


def test_print_blind_test_prints_headers(capsys):
    with patch("gliner2.training.metrics._print_micro_report"):
        train._print_blind_test({"eval_entity_f1": 0.9})
    out = capsys.readouterr().out
    assert "===== Blind test metrics =====" in out
    assert "===== Blind test summary =====" in out


def test_print_blind_test_float_four_decimals(capsys):
    with patch("gliner2.training.metrics._print_micro_report"):
        train._print_blind_test({"eval_entity_f1": 0.91234})
    out = capsys.readouterr().out
    assert "0.9123" in out


def test_print_blind_test_int_no_decimal(capsys):
    with patch("gliner2.training.metrics._print_micro_report"):
        train._print_blind_test({"eval_entity_count": 42})
    out = capsys.readouterr().out
    assert "42" in out
    assert "42." not in out


def test_print_blind_test_classification_report_printed(capsys):
    metrics = {
        "eval_entity_f1": 0.9,
        "eval_entity_strict_classification_report": "precision: 0.9\nrecall: 0.8",
    }
    with patch("gliner2.training.metrics._print_micro_report"):
        train._print_blind_test(metrics)
    out = capsys.readouterr().out
    assert "--- entity strict classification report ---" in out
    assert "precision: 0.9" in out


def test_print_blind_test_calls_micro_report():
    metrics = {"eval_entity_f1": 0.9}
    with patch("gliner2.training.metrics._print_micro_report") as mock_fn:
        train._print_blind_test(metrics)
    mock_fn.assert_called_once_with(metrics)


# ---------------------------------------------------------------------------
# _blind_test_by_language
# ---------------------------------------------------------------------------

def test_blind_test_by_language_returns_aggregate_metrics(tmp_path):
    records = _make_records(["eng", "fra"])
    mock_metrics = {"eval_entity_f1": 0.8}

    with patch.object(train, "_annotate_languages", side_effect=lambda r: r), \
         patch("gliner2.GLiNER2.from_pretrained", return_value=MagicMock()), \
         patch("gliner2.training.metrics.compute_metrics", return_value=mock_metrics), \
         patch("gliner2.training.trainer.ExtractorDataset", return_value=MagicMock()), \
         patch.object(train, "_print_blind_test"):
        result = train._blind_test_by_language(tmp_path, records, eval_bs=4, eval_thr=0.5)

    assert result == mock_metrics


def test_blind_test_by_language_alphabetical_order(tmp_path, capsys):
    records = _make_records(["spa", "eng", "fra"])

    with patch.object(train, "_annotate_languages", side_effect=lambda r: r), \
         patch("gliner2.GLiNER2.from_pretrained", return_value=MagicMock()), \
         patch("gliner2.training.metrics.compute_metrics", return_value={}), \
         patch("gliner2.training.trainer.ExtractorDataset", return_value=MagicMock()), \
         patch.object(train, "_print_blind_test"):
        train._blind_test_by_language(tmp_path, records, eval_bs=4, eval_thr=0.5)

    out = capsys.readouterr().out
    assert out.find("eng") < out.find("fra") < out.find("spa")


def test_blind_test_by_language_loads_model_once(tmp_path):
    records = _make_records(["eng", "fra", "spa"])

    with patch.object(train, "_annotate_languages", side_effect=lambda r: r), \
         patch("gliner2.GLiNER2.from_pretrained", return_value=MagicMock()) as mock_load, \
         patch("gliner2.training.metrics.compute_metrics", return_value={}), \
         patch("gliner2.training.trainer.ExtractorDataset", return_value=MagicMock()), \
         patch.object(train, "_print_blind_test"):
        train._blind_test_by_language(tmp_path, records, eval_bs=4, eval_thr=0.5)

    assert mock_load.call_count == 1


def test_blind_test_by_language_combined_pass(tmp_path, capsys):
    records = _make_records(["eng", "fra"])

    with patch.object(train, "_annotate_languages", side_effect=lambda r: r), \
         patch("gliner2.GLiNER2.from_pretrained", return_value=MagicMock()), \
         patch("gliner2.training.metrics.compute_metrics", return_value={}), \
         patch("gliner2.training.trainer.ExtractorDataset", return_value=MagicMock()), \
         patch.object(train, "_print_blind_test"):
        train._blind_test_by_language(tmp_path, records, eval_bs=4, eval_thr=0.5)

    assert "All languages combined" in capsys.readouterr().out


def test_blind_test_by_language_compute_metrics_per_lang_plus_combined(tmp_path):
    """compute_metrics is called once per distinct language + once for all combined."""
    records = _make_records(["eng", "eng", "fra"])

    with patch.object(train, "_annotate_languages", side_effect=lambda r: r), \
         patch("gliner2.GLiNER2.from_pretrained", return_value=MagicMock()), \
         patch("gliner2.training.metrics.compute_metrics", return_value={}) as mock_cm, \
         patch("gliner2.training.trainer.ExtractorDataset", return_value=MagicMock()), \
         patch.object(train, "_print_blind_test"):
        train._blind_test_by_language(tmp_path, records, eval_bs=4, eval_thr=0.5)

    # 2 languages (eng, fra) + 1 combined = 3 calls
    assert mock_cm.call_count == 3


def test_blind_test_by_language_materialises_file_paths(tmp_path):
    """File path strings are read and converted to dicts before annotation."""
    jsonl = tmp_path / "test.jsonl"
    record = {"input": "Hello world", "output": {}}
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with patch.object(train, "_annotate_languages", side_effect=lambda r: r) as mock_ann, \
         patch("gliner2.GLiNER2.from_pretrained", return_value=MagicMock()), \
         patch("gliner2.training.metrics.compute_metrics", return_value={}), \
         patch("gliner2.training.trainer.ExtractorDataset", return_value=MagicMock()), \
         patch.object(train, "_print_blind_test"):
        train._blind_test_by_language(tmp_path, [str(jsonl)], eval_bs=4, eval_thr=0.5)

    called_arg = mock_ann.call_args[0][0]
    assert isinstance(called_arg[0], dict)
    assert called_arg[0]["input"] == "Hello world"


# ---------------------------------------------------------------------------
# eval_by_language config option
# ---------------------------------------------------------------------------

def test_eval_by_language_defaults_false():
    assert {}.get("eval_by_language", False) is False


def test_eval_by_language_reads_true():
    assert {"eval_by_language": True}.get("eval_by_language", False) is True


def test_eval_by_language_reads_false_explicit():
    cfg = {"eval_by_language": False, "batch_size": 8, "threshold": 0.5}
    assert cfg.get("eval_by_language", False) is False
