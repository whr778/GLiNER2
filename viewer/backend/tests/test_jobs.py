"""Inference tickets write the same JSONL line tools/infer.py --output writes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gold  # noqa: E402
import jobs  # noqa: E402
from infer import _gold_schema, prediction_row  # noqa: E402

CASIE = Path(__file__).resolve().parents[3] / "data" / "casie.test.jsonl"
PANEL = {"entities": ["person", "organization"]}


def setup_function():
    jobs.clear()


def _casie_records(n):
    lines = [ln for ln in CASIE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines[:n]]


def test_casie_row_is_an_infer_output_and_gold_schema_names_its_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "predictions_dir", lambda: tmp_path)
    rec = _casie_records(1)[0]
    schema = _gold_schema(rec)
    assert schema["entities"] and schema["events"]
    pred = {"entities": {"Attacker": [{"text": "Hackers", "start": 0, "end": 7, "confidence": 0.4}]}}
    hand = {"input": rec["input"], "output": pred, "gold": rec["output"]}
    assert prediction_row(rec, pred) == hand
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps(prediction_row(rec, pred), ensure_ascii=False) + "\n", encoding="utf-8")
    summary = gold.summary(str(path))
    assert summary["count"] == 1 and summary["data_file"] is False
    assert "entity" in summary["records"][0]["f1"]


def test_jsonl_job_writes_a_file_the_predictions_screen_can_open(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "predictions_dir", lambda: tmp_path)
    seen = []

    def predict(texts, schemas):
        seen.append((texts[0][:12], sorted(schemas[0])))
        return [{"entities": {}} for _ in texts]

    job = jobs.create("jsonl", None, "data/casie.test.jsonl", "gold", {}, {}, 2, "unit-model")
    jobs.execute(job, predict, batch_size=1)
    assert job.status == "done" and job.done == 2 and job.total == 2
    assert seen[0][0] == "Hackers logg"
    assert "entities" in seen[0][1] and "events" in seen[0][1]
    summary = gold.summary(job.output_path)
    assert summary["count"] == 2 and summary["data_file"] is False
    listed = {entry["path"] for entry in gold.list_files(str(tmp_path))}
    assert job.output_path in listed


def test_stop_during_the_only_batch_is_not_marked_done(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "predictions_dir", lambda: tmp_path)
    job = jobs.create("text", "Ann said she left.", None, "panel", PANEL, {}, 0, "unit-model")

    def predict(texts, schemas):
        jobs.cancel(job.id)
        assert job.status == "stopping"
        return [{"entities": {"person": ["Ann"]}} for _ in texts]

    jobs.execute(job, predict, batch_size=1)
    assert job.status == "cancelled"
    assert job.result["result"]["entities"]["person"] == ["Ann"]


def test_cancel_stops_before_the_next_record(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "predictions_dir", lambda: tmp_path)
    job = jobs.create("jsonl", None, "data/casie.test.jsonl", "panel", PANEL, {}, 4, "unit-model")

    def predict(texts, schemas):
        jobs.cancel(job.id)
        return [{"entities": {}} for _ in texts]

    jobs.execute(job, predict, batch_size=1)
    assert job.status == "cancelled" and job.done == 1
    rows = [json.loads(ln) for ln in Path(job.output_path).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["gold"]["events"]


def test_delete_removes_the_ticket_and_its_output(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "predictions_dir", lambda: tmp_path)
    job = jobs.create("jsonl", None, "data/casie.test.jsonl", "panel", PANEL, {}, 1, "unit-model")
    jobs.execute(job, lambda texts, schemas: [{"entities": {}} for _ in texts], batch_size=1)
    out = Path(job.output_path)
    assert out.is_file()
    jobs.delete(job.id)
    assert jobs.get(job.id) is None
    assert not out.exists()
    assert not (tmp_path / "jobs" / f"{job.id}.json").exists()


def test_delete_while_running_removes_the_file_when_the_batch_ends(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "predictions_dir", lambda: tmp_path)
    job = jobs.create("jsonl", None, "data/casie.test.jsonl", "panel", PANEL, {}, 3, "unit-model")

    def predict(texts, schemas):
        jobs.delete(job.id)
        assert job.delete_after and job.status == "stopping"
        return [{"entities": {}} for _ in texts]

    jobs.execute(job, predict, batch_size=1)
    assert jobs.get(job.id) is None
    assert not Path(job.output_path).exists()


def test_dataset_path_outside_data_root_is_refused():
    try:
        jobs.resolve_dataset("/etc/passwd")
    except ValueError as exc:
        assert "data" in str(exc).lower() or "inside" in str(exc).lower()
    else:
        raise AssertionError("path outside data_root was accepted")


def test_queue_lists_the_ticket_before_it_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "predictions_dir", lambda: tmp_path)
    job = jobs.create("text", "Ann said she left.", None, "panel", PANEL, {"device": "cpu"}, 0, "unit-model")
    assert job.status == "queued"
    assert [item.id for item in jobs.list_jobs()] == [job.id]
    jobs.execute(job, lambda texts, schemas: [{"entities": {"person": ["Ann"]}} for _ in texts])
    assert job.status == "done" and job.result["result"]["entities"]["person"] == ["Ann"]


def test_a_job_left_running_reloads_as_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "predictions_dir", lambda: tmp_path)
    job = jobs.create("text", "Ann said she left.", None, "panel", PANEL, {}, 0, "unit-model")
    job.status = "running"
    jobs.save(job)
    jobs.clear()
    jobs.load()
    restored = jobs.get(job.id)
    assert restored is not None and restored.status == "interrupted"
    assert jobs._claim() is None


def test_dataset_search_finds_casie_off():
    hits = jobs.list_datasets("casie_off.test")
    assert any(item["path"].endswith("casie_off.test.jsonl") for item in hits)
