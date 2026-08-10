"""EKF tracking runs for the viewer: news feed -> observations -> tracked timeline.

Wraps ``tools/ekf_showcase/run_pipeline.py`` so the viewer can run the whole research
line on a JSONL feed. A run is minutes long (N articles x several model calls), far
past any sensible HTTP timeout, so it executes on a worker thread and the frontend
polls: ``POST /ekf-track`` returns a job id immediately, ``GET /ekf-track/{id}``
returns progress and then the result.

Feed paths are resolved against the repo root and must stay inside it -- the viewer
is a local tool, but an endpoint that takes a path should still not read /etc.
"""
from __future__ import annotations

import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
SHOWCASE = REPO / "tools" / "ekf_showcase"
for extra in (str(SHOWCASE), str(REPO / "datasets" / "disaster_streams")):
    if extra not in sys.path:
        sys.path.insert(0, extra)


@dataclass
class Job:
    id: str
    status: str = "running"          # running | done | error
    stage: str = "starting"
    done: int = 0
    total: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    log: List[str] = field(default_factory=list)

    def public(self) -> Dict[str, Any]:
        return {"job_id": self.id, "status": self.status, "stage": self.stage,
                "done": self.done, "total": self.total, "error": self.error,
                "log": self.log[-12:], "result": self.result}


_JOBS: Dict[str, Job] = {}
_LOCK = threading.Lock()


def resolve_feed(feed: str) -> Path:
    """Repo-relative (or absolute) feed path, constrained to the repo."""
    p = Path(feed).expanduser()
    p = p if p.is_absolute() else (REPO / p)
    p = p.resolve()
    if REPO not in p.parents and p != REPO:
        raise ValueError(f"feed path must stay inside the repository: {feed}")
    if not p.is_file():
        raise ValueError(f"feed not found: {feed}")
    return p


def resolve_model(name: str) -> str:
    """Local checkpoint paths -> absolute; Hub ids pass through unchanged.

    The backend runs with CWD=viewer/backend, so a repo-relative checkpoint like
    ``out/casualty-multievent/best`` does not exist from here and huggingface_hub
    then rejects it as a malformed repo id. Resolve against the repo root first,
    and only fall back to treating it as a Hub id.
    """
    if not name:
        return name
    candidate = Path(name).expanduser()
    for p in (candidate, REPO / name):
        if p.exists():
            return str(p.resolve())
    return name


def list_feeds() -> List[Dict[str, Any]]:
    """JSONL files that look like a feed: objects carrying `t_hours` and `text`."""
    import json
    out: List[Dict[str, Any]] = []
    for d in (REPO / "datasets" / "ekf_showcase", REPO / "datasets"):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jsonl")):
            try:
                with p.open(encoding="utf-8") as fh:
                    first = json.loads(fh.readline() or "{}")
            except Exception:
                continue
            if "t_hours" in first and "text" in first:
                n = sum(1 for _ in p.open(encoding="utf-8"))
                truth = p.with_name(p.stem + ".truth.jsonl")
                out.append({"path": str(p.relative_to(REPO)), "articles": n,
                            "truth": str(truth.relative_to(REPO)) if truth.is_file() else None})
        if out:
            break
    return out


def _run(job: Job, params: Dict[str, Any]) -> None:
    import json
    import run_pipeline as rp
    from gliner2 import AutoExtractor

    try:
        feed_path = resolve_feed(params["feed"])
        feed = [json.loads(l) for l in feed_path.open(encoding="utf-8") if l.strip()]
        feed.sort(key=lambda r: r["t_hours"])
        limit = int(params.get("limit") or 0)
        if limit:
            feed = feed[:limit]
        texts = [r["text"] for r in feed]
        job.total = len(feed)
        device = params.get("device") or "cpu"

        job.stage = "gate"
        job.log.append(f"gate: {params['gate_model']} on {len(feed)} articles")
        gate_model = AutoExtractor.from_pretrained(resolve_model(params["gate_model"]), map_location=device)
        gates = rp.gate(gate_model, texts, float(params.get("gate_threshold", 0.5)))
        kept = [i for i, g in enumerate(gates) if g["relevant"]]
        job.log.append(f"gate kept {len(kept)}/{len(feed)}")

        events: List[Dict[str, Any]] = [{} for _ in feed]
        if params.get("event_model"):
            job.stage = "events"
            job.log.append(f"events: {params['event_model']}")
            ev_model = AutoExtractor.from_pretrained(resolve_model(params["event_model"]), map_location=device)
            found = rp.extract_stage1(ev_model, [texts[i] for i in kept],
                                      float(params.get("event_threshold", 0.3)))
            for i, e in zip(kept, found):
                events[i] = e
            del ev_model

        job.stage = "casualty"
        job.log.append(f"casualty: {params['casualty_model']}")
        cas_model = AutoExtractor.from_pretrained(resolve_model(params["casualty_model"]), map_location=device)
        # associate: the viewer shipped 6h BEFORE association landed in the pipeline and
        # then never caught up, so every observation was pooled into one stream -- the
        # exact failure association fixed (multi-event nRMSE 102 -> 27.975). Reuse the
        # pipeline's own keying rather than re-deriving it here, which is how it drifted.
        associate = params.get("associate") or "none"
        cas_schema = rp.build_casualty_schema(with_location=associate == "record")
        modes = [params.get("normalizer") or "hybrid"]
        cls_schema = (rp.build_normalizer_schema(gate_model)
                      if modes[0] in ("classify", "hybrid") else None)

        per_mode: Dict[str, List[Dict]] = {m: [] for m in modes}
        articles: List[Dict[str, Any]] = []
        window = params.get("window") or "article"
        for i, row in enumerate(feed):
            entry = {"t_hours": row["t_hours"], "text": row["text"], **gates[i],
                     "events": events[i], "observations": []}
            if gates[i]["relevant"]:
                if window == "lead":
                    envelopes = [{"text": row["text"][: int(params.get("lead_chars", 1100))]}]
                else:
                    envelopes = (rp.casualty_windows(row["text"], events[i],
                                                     int(params.get("envelope_margin", 60)))
                                 if window == "event" and events[i] else [])
                entry["envelopes"] = envelopes
                for env in (envelopes or [{"text": row["text"]}]):
                    read_text = env["text"]
                    records = (cas_model.extract(read_text, cas_schema,
                                                 include_confidence=True)
                               .get("casualty_report") or [])
                    for rec in (records or [{}]):
                        key = (rp.record_key(events[i], rec) if associate == "record"
                               else rp.association_key(events[i], associate, env))
                        rp._emit(rec, read_text, row, entry, modes, per_mode,
                                 gate_model, cls_schema, event_key=key)
            articles.append(entry)
            job.done = i + 1

        job.stage = "tracking"
        t0, t1 = feed[0]["t_hours"], feed[-1]["t_hours"]
        step = float(params.get("grid_step", 6.0))
        grid = [t0 + k * step for k in range(int((t1 - t0) / step) + 1)]
        tracked = {m: rp.track(per_mode[m], grid) for m in modes}
        # Per-stream tracking is the point of associating at all; track_by_event also
        # folds clipped location keys (syr -> syria) via merge_prefix_keys.
        tracked_by_event = {m: rp.track_by_event(per_mode[m], grid) for m in modes}

        result: Dict[str, Any] = {
            "feed": str(feed_path.relative_to(REPO)), "grid": grid,
            "articles": articles, "n_articles": len(feed), "n_relevant": len(kept),
            "tracked": tracked, "tracked_by_event": tracked_by_event,
            "associate": associate, "mode": modes[0],
            "n_observations": {m: len(per_mode[m]) for m in modes},
        }
        truth_path = params.get("truth") or str(
            feed_path.with_name(feed_path.stem + ".truth.jsonl").relative_to(REPO))
        tp = REPO / truth_path
        if tp.is_file():
            truth = [json.loads(l) for l in tp.open(encoding="utf-8") if l.strip()]
            result["truth"] = {role: [rp._truth_at(truth, role, t) for t in grid]
                               for role in rp.ROLES}

        job.result = result
        job.stage = "done"
        job.status = "done"
        job.log.append(f"done: {sum(len(v) for v in per_mode.values())} observations")
    except Exception as exc:                                   # surfaced to the UI
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.log.append(traceback.format_exc().splitlines()[-1])


def start(params: Dict[str, Any]) -> Job:
    job = Job(id=uuid.uuid4().hex[:12])
    with _LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=_run, args=(job, params), daemon=True).start()
    return job


def get(job_id: str) -> Optional[Job]:
    with _LOCK:
        return _JOBS.get(job_id)
