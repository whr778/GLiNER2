"""Inference jobs for the viewer.

A job is a ticket: text or a JSONL dataset, a model, a schema mode. The ticket
comes back immediately. One worker runs the queue, because the loaded model and
its gates are one shared object -- a second run would change them under the first.

JSONL output is the same line ``tools/infer.py --output`` writes, so Predictions
vs gold can open it, including a file the run has not finished.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import config

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from infer import _gold_schema, prediction_row  # noqa: E402

Predict = Callable[[List[str], List[Dict[str, Any]]], List[Dict[str, Any]]]

_LOCK = threading.Lock()
_JOBS: Dict[str, "Job"] = {}
_ORDER: List["Job"] = []
_CANCEL: Dict[str, threading.Event] = {}
_STOP = threading.Event()
_THREAD: Optional[threading.Thread] = None


def predictions_dir() -> Path:
    return config.predictions_root()


@dataclass
class Job:
    id: str
    kind: str
    status: str
    label: str
    model: str
    source: str
    schema_mode: str
    created: float
    done: int = 0
    total: int = 0
    output_path: Optional[str] = None
    error: Optional[str] = None
    device: Optional[str] = None
    ran_on: Optional[str] = None
    limit: int = 0
    text: Optional[str] = None
    schema: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    delete_after: bool = False
    label_map: Any = None
    label_style: Any = None

    def public(self, full: bool = False) -> Dict[str, Any]:
        out = {
            "id": self.id, "kind": self.kind, "status": self.status, "label": self.label,
            "model": self.model, "source": self.source, "schema_mode": self.schema_mode,
            "created": self.created, "done": self.done, "total": self.total,
            "output_path": self.output_path, "error": self.error, "device": self.device,
            "ran_on": self.ran_on, "limit": self.limit,
            "delete_after": self.delete_after,
        }
        if full:
            out["text"] = self.text
            out["result"] = self.result
            out["schema"] = self.schema
        return out


def _state_path(job_id: str) -> Path:
    return predictions_dir() / "jobs" / f"{job_id}.json"


def save(job: Job) -> None:
    path = _state_path(job.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = job.public(full=True)
    body["schema"] = job.schema
    body["options"] = job.options
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")


def resolve_dataset(path: str) -> Path:
    """A JSONL under ``data_root``. Anything else is refused."""
    if not path or not str(path).strip():
        raise ValueError("dataset path is empty")
    root = config.data_root().resolve()
    raw = Path(path).expanduser()
    candidate = raw if raw.is_absolute() else (REPO / raw)
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"dataset must stay inside {root}")
    if candidate.suffix != ".jsonl" or not candidate.is_file():
        raise ValueError(f"not a jsonl file: {path}")
    return candidate


def list_datasets(query: str = "", limit: int = 200) -> List[Dict[str, Any]]:
    """JSONL files under ``data_root`` whose path contains ``query``."""
    root = config.data_root()
    q = (query or "").lower()
    out = []
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*.jsonl")):
        rel = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
        if q and q not in rel.lower():
            continue
        out.append({"path": rel, "name": path.name, "bytes": path.stat().st_size})
        if len(out) >= limit:
            break
    return out


def _new_id() -> str:
    for _ in range(5):
        job_id = uuid.uuid4().hex[:8]
        if job_id not in _JOBS and not _state_path(job_id).exists():
            return job_id
    return uuid.uuid4().hex


def create(kind: str, text: Optional[str], path: Optional[str], schema_mode: str,
           schema: Dict[str, Any], options: Dict[str, Any], limit: int, model: str) -> Job:
    """Validate and enqueue. Raises ``ValueError`` on a bad request."""
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if kind == "text":
        if not (text or "").strip():
            raise ValueError("text is empty")
        schema_mode = "panel"
        if not schema:
            raise ValueError("schema is empty")
        label = " ".join(text.split())[:72]
        source = "text"
        dataset = None
    elif kind == "jsonl":
        dataset = resolve_dataset(path or "")
        if schema_mode not in ("gold", "panel"):
            raise ValueError("schema_mode must be gold or panel")
        if schema_mode == "panel" and not schema:
            raise ValueError("schema is empty")
        label = dataset.name
        source = str(dataset.relative_to(REPO)) if dataset.is_relative_to(REPO) else str(dataset)
    else:
        raise ValueError("kind must be text or jsonl")
    job = Job(
        id=_new_id(), kind=kind, status="queued", label=label or kind, model=model,
        source=source, schema_mode=schema_mode, created=time.time(), limit=limit,
        text=text if kind == "text" else None, schema=schema or {}, options=options or {},
        device=(options or {}).get("device"),
    )
    if kind == "jsonl":
        slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model)[:60]
        out = predictions_dir() / f"{dataset.stem}__{slug}__{job.id}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        job.output_path = str(out)
    with _LOCK:
        _JOBS[job.id] = job
        _ORDER.append(job)
        _CANCEL[job.id] = threading.Event()
    save(job)
    return job


def clear() -> None:
    """Drop in-memory tickets. Tests use this; it does not delete files."""
    with _LOCK:
        _JOBS.clear()
        _ORDER.clear()
        _CANCEL.clear()


def get(job_id: str) -> Optional[Job]:
    with _LOCK:
        return _JOBS.get(job_id)


def list_jobs() -> List[Job]:
    with _LOCK:
        return sorted(_ORDER, key=lambda j: j.created, reverse=True)


def cancel(job_id: str) -> Optional[Job]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        if job.status == "queued":
            job.status = "cancelled"
            save(job)
        elif job.status == "running":
            # The forward already in flight finishes. The loop stops before the next batch.
            job.status = "stopping"
            _CANCEL[job_id].set()
            save(job)
        return job


def _unlink(job: Job) -> None:
    """Remove the ticket file and its output, when the output is under predictions_dir."""
    state = _state_path(job.id)
    if state.is_file():
        state.unlink()
    raw = job.output_path
    if not raw:
        return
    root = predictions_dir().resolve()
    path = Path(raw).resolve()
    if path != root and root not in path.parents:
        return
    if path.is_file():
        path.unlink()


def _drop(job: Job) -> None:
    """Forget the ticket and delete its files. Caller holds ``_LOCK``."""
    job.status = "cancelled"
    _JOBS.pop(job.id, None)
    _ORDER[:] = [item for item in _ORDER if item.id != job.id]
    _CANCEL.pop(job.id, None)
    _unlink(job)


def delete(job_id: str) -> Optional[Job]:
    """Remove a ticket and its output file.

    A job inside the model is cancelled first. Its files go when that batch ends,
    so the writer is not deleting a file it still has open until the batch returns.
    """
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        if job.status in ("running", "stopping"):
            job.delete_after = True
            job.status = "stopping"
            _CANCEL[job_id].set()
            save(job)
            return job
        _drop(job)
        return job


def load() -> None:
    """Reload tickets. A job that was queued or running when the process died stays interrupted."""
    folder = predictions_dir() / "jobs"
    if not folder.is_dir():
        return
    found = []
    for path in folder.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("delete_after"):
            ghost = Job(
                id=data["id"], kind=data.get("kind") or "text", status="cancelled",
                label=data.get("label") or data["id"], model=data.get("model") or "",
                source=data.get("source") or "", schema_mode=data.get("schema_mode") or "panel",
                created=float(data.get("created") or 0), output_path=data.get("output_path"),
            )
            _unlink(ghost)
            continue
        if data.get("status") in ("queued", "running", "stopping"):
            data["status"] = "interrupted"
            data["error"] = "the server stopped while this job was in the queue"
        job = Job(
            id=data["id"], kind=data.get("kind") or "text", status=data.get("status") or "interrupted",
            label=data.get("label") or data["id"], model=data.get("model") or "",
            source=data.get("source") or "", schema_mode=data.get("schema_mode") or "panel",
            created=float(data.get("created") or 0), done=int(data.get("done") or 0),
            total=int(data.get("total") or 0), output_path=data.get("output_path"),
            error=data.get("error"), device=data.get("device"), ran_on=data.get("ran_on"),
            limit=int(data.get("limit") or 0), text=data.get("text"),
            schema=data.get("schema") or {}, options=data.get("options") or {},
            result=data.get("result"), delete_after=bool(data.get("delete_after")),
        )
        found.append(job)
        save(job)
    found.sort(key=lambda j: j.created)
    with _LOCK:
        for job in found:
            if job.id not in _JOBS:
                _JOBS[job.id] = job
                _ORDER.append(job)
                _CANCEL.setdefault(job.id, threading.Event())


def _records(job: Job) -> List[Dict[str, Any]]:
    if job.kind == "text":
        return [{"input": job.text or ""}]
    path = resolve_dataset(job.source)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if job.limit and len(rows) >= job.limit:
                break
    return rows


def _schema(job: Job, rec: Dict[str, Any]) -> Dict[str, Any]:
    from gliner2.inference.label_map import apply_label_map

    raw = _gold_schema(rec) if job.schema_mode == "gold" else job.schema
    mapped, _applied = apply_label_map(raw or {}, job.label_map, job.label_style)
    return mapped


def _stop_requested(job_id: str) -> bool:
    event = _CANCEL.get(job_id)
    return event is not None and event.is_set()


def _mark_cancelled(job: Job) -> None:
    job.status = "cancelled"
    save(job)


def _stopped(job: Job) -> bool:
    """End the job when stop was asked. A delete removes the files instead of keeping a cancelled ticket."""
    if not (_stop_requested(job.id) or job.delete_after):
        return False
    if job.delete_after:
        with _LOCK:
            if job.id in _JOBS:
                _drop(job)
        return True
    _mark_cancelled(job)
    return True


def execute(job: Job, predict: Predict, *, batch_size: int = 4) -> None:
    """Run one claimed job. ``predict`` decodes a batch of texts against already-mapped schemas.

    Stop finishes the batch already inside the model, writes it, and does not start another.
    A one-record job is that single batch, so the ticket ends cancelled rather than done.
    """
    if _stopped(job):
        return
    job.status = "running"
    save(job)
    handle = None
    try:
        records = _records(job)
        job.total = len(records)
        save(job)
        step = max(1, batch_size)
        for start in range(0, len(records), step):
            if _stopped(job):
                return
            chunk = records[start:start + step]
            schemas = [_schema(job, rec) for rec in chunk]
            keep = [i for i, schema in enumerate(schemas) if schema]
            preds: List[Dict[str, Any]] = [{} for _ in chunk]
            if keep:
                got = predict([chunk[i]["input"] for i in keep], [schemas[i] for i in keep])
                if len(got) != len(keep):
                    raise RuntimeError(f"predict returned {len(got)} results for {len(keep)} texts")
                for i, pred in zip(keep, got):
                    preds[i] = pred or {}
            if job.output_path:
                if handle is None:
                    Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
                    handle = open(job.output_path, "w", encoding="utf-8")
                for rec, pred in zip(chunk, preds):
                    handle.write(json.dumps(prediction_row(rec, pred), ensure_ascii=False) + "\n")
                handle.flush()
            else:
                job.result = {"text": chunk[0]["input"], "result": preds[0]}
            job.done = min(start + step, len(records))
            save(job)
            if _stopped(job):
                return
        job.status = "done"
        save(job)
    except Exception as exc:
        if job.delete_after:
            with _LOCK:
                if job.id in _JOBS:
                    _drop(job)
            return
        job.status = "failed"
        job.error = getattr(exc, "detail", None) or str(exc)
        save(job)
        raise
    finally:
        if handle is not None:
            handle.close()


def _claim() -> Optional[Job]:
    with _LOCK:
        for job in _ORDER:
            if job.status == "queued":
                job.status = "running"
                save(job)
                return job
    return None


def _loop(runner: Callable[[Job], None]) -> None:
    while not _STOP.is_set():
        job = _claim()
        if job is None:
            _STOP.wait(0.25)
            continue
        try:
            runner(job)
        except Exception as exc:
            if job.status == "running":
                job.status = "failed"
                job.error = getattr(exc, "detail", None) or str(exc)
                save(job)


def start(runner: Callable[[Job], None]) -> None:
    global _THREAD
    if _THREAD and _THREAD.is_alive():
        return
    _STOP.clear()
    _THREAD = threading.Thread(target=_loop, args=(runner,), name="inference-jobs", daemon=True)
    _THREAD.start()


def stop() -> None:
    _STOP.set()
