"""build_schema: the train gold's schema, and val/test labels it does not offer."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
import build_schema as B  # noqa: E402


def _write(path, outputs):
    path.write_text("".join(json.dumps({"input": "x", "output": o}) + "\n" for o in outputs), encoding="utf-8")


def test_schema_from_train_reports_eval_only_labels(tmp_path, monkeypatch, capsys):
    ev = lambda t, r: {"event_type": t, "arguments": [{"role": r, "text": "x"}]}  # noqa: E731
    _write(tmp_path / "c.train.jsonl", [{"entities": {"Person": ["x"]}, "events": [ev("Attack", "Victim")]}])
    _write(tmp_path / "c.dev.jsonl", [{"entities": {"Malware": ["x"]}}])
    _write(tmp_path / "c.test.jsonl", [{"events": [ev("Attack", "Tool")]}])
    out = tmp_path / "schemas" / "c.json"
    monkeypatch.setattr(sys, "argv", ["build_schema.py", str(tmp_path / "c"), "--out", str(out)])
    assert B.main() == 0
    menu = json.loads(out.read_text(encoding="utf-8"))
    assert menu["schema"] == {"entities": ["Person"], "events": {"Attack": ["Victim"]}}
    assert menu["name"] == "c" and menu["exhaustive_for"] == []
    assert "not offered: 2" in capsys.readouterr().out


def test_include_eval_offers_them(tmp_path, monkeypatch):
    _write(tmp_path / "c.train.jsonl", [{"entities": {"Person": ["x"]}}])
    _write(tmp_path / "c.test.jsonl", [{"entities": {"Malware": ["x"]}}])
    out = tmp_path / "c.json"
    monkeypatch.setattr(sys, "argv", ["build_schema.py", str(tmp_path / "c"), "--out", str(out), "--include-eval",
                                      "--exhaustive"])
    B.main()
    menu = json.loads(out.read_text(encoding="utf-8"))
    assert menu["schema"]["entities"] == ["Malware", "Person"] and menu["exhaustive_for"] == ["c"]
