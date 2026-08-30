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


# Per-feed recommended settings, and WHY each one is there. Every entry is a documented
# configuration from the research runs, not a preference: getting these wrong does not
# degrade a result, it produces a silently empty or wrong one.
#
#   window/associate  `long` + `record` is the research default. `long` takes Helene
#                     `dead` observations 25 -> 106; `record` binds the number to the
#                     place the same decode step produced.
#   rollup            Helene and Aegean key streams on a bare lowercased place, which is
#                     what a collapse_type rollup emits. Turkiye keys on
#                     "Earthquakes|<place>" and its rollup would STRIP the type -- get it
#                     wrong and every stream drops out silently as nan.
#   event_year        Turkiye only. Without it the 1999 Izmit toll is tracked as a 2023
#                     figure.
#   max_plausible     Helene only, and event-specific by nature. 2,000 removes a 94,000
#                     that is Asheville's POPULATION read as a death toll, taking ungated
#                     per-place error 378.809 -> 18.190.
#   gate_threshold    0.9 chosen 2026-08-28 by gate_threshold_sweep.py.
FEED_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "helene2024": {
        "window": "long", "associate": "record", "gate_threshold": 0.9,
        "max_plausible": 2000,
        "why": "Research configuration. max_plausible 2000 removes a 94,000 that is "
               "Asheville's population read as a death toll (per-place error 378.8 -> 18.2).",
    },
    "aegean2020": {
        "window": "long", "associate": "record", "gate_threshold": 0.9,
        "why": "Research configuration. Keys streams on a bare place, so the rollup "
               "beside the feed applies.",
    },
    "turkey2023": {
        "window": "long", "associate": "record", "gate_threshold": 0.9,
        "rollup": "", "event_year": 2023,
        "why": "Rollup DISABLED: this event keys on 'Earthquakes|<place>' and its rollup "
               "would strip the type, dropping every stream as nan. event_year 2023 keeps "
               "the 1999 Izmit toll from being tracked as a 2023 figure.",
    },
}


def recommend_for(feed_path: str) -> Optional[Dict[str, Any]]:
    """Recommended settings for a feed, keyed by its event directory."""
    parts = Path(feed_path).parts
    for token in parts:
        if token in FEED_RECOMMENDATIONS:
            return FEED_RECOMMENDATIONS[token]
    return None


def list_feeds() -> List[Dict[str, Any]]:
    """JSONL files that look like a feed: objects carrying `t_hours` and `text`.

    Searched RECURSIVELY, and every directory is searched. The previous version used a
    non-recursive glob and stopped at the first directory that matched, so it only ever
    saw ``datasets/ekf_showcase/*.jsonl`` -- three synthetic feeds -- while the three REAL
    events this project is about sit one level deeper in ``datasets/<event>/_cache/`` and
    were invisible: Helene (70 articles), Aegean (71), Turkiye (16).

    Truth is looked up two ways because the two families name it differently: the showcase
    feeds use ``<stem>.truth.jsonl`` beside the feed, the real events use
    ``ground_truth.json`` at the event root (the path ``scope_gate_test.DATASETS`` points
    at). Reporting truth as absent when it exists would hide the tracking error column.
    """
    import json
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for d in (REPO / "datasets" / "ekf_showcase", REPO / "datasets"):
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.jsonl")):
            if p in seen or p.name.endswith(".truth.jsonl"):
                continue
            # Skip training corpora. Recursion also reaches datasets/disaster_streams_*/
            # {train,val,test}/, whose observation files carry t_hours and text and so
            # look exactly like feeds -- but at 10k-45k rows they are training data, not
            # events, and would bury the three real feeds in the dropdown.
            if {"train", "val", "test"} & {q.name for q in p.parents}:
                continue
            try:
                with p.open(encoding="utf-8") as fh:
                    first = json.loads(fh.readline() or "{}")
            except Exception:
                continue
            if not ("t_hours" in first and "text" in first):
                continue
            seen.add(p)
            n = sum(1 for _ in p.open(encoding="utf-8"))
            truth = p.with_name(p.stem + ".truth.jsonl")
            if not truth.is_file():
                # datasets/<event>/_cache/feed.jsonl -> datasets/<event>/ground_truth.json
                alt = p.parent.parent / "ground_truth.json"
                truth = alt if alt.is_file() else truth
            out.append({"path": str(p.relative_to(REPO)), "articles": n,
                        "truth": str(truth.relative_to(REPO)) if truth.is_file() else None,
                        "recommended": recommend_for(str(p.relative_to(REPO)))})
    # Feeds WITH ground truth first: those are the ones that can report tracking error,
    # which is the whole point of the panel.
    out.sort(key=lambda f: (f["truth"] is None, f["path"]))
    return out


def _ground_truth_series(path: Path, grid: List[float]) -> Optional[Dict[str, Any]]:
    """Read a real event's ``ground_truth.json`` onto the chart grid.

    A DIFFERENT format from the showcase feeds' ``<stem>.truth.jsonl``, which is why the
    three real events showed no gold overlay at all: the runner only ever looked for the
    JSONL form, so Helene, Aegean and Turkiye silently had no truth to draw.

    Snapshots are absolute timestamps, so they are converted to hours from ``onset_utc``
    -- the same conversion ``scope_gate_test.truth`` does. Helene and Aegean carry a
    ``Total`` inside ``deaths``; Turkiye ships flat ``turkey``/``syria`` keys and no total,
    so the combined series is their sum.

    Only DEATHS exist in these files. `injured` and `missing` return None rather than
    zero: a flat zero line would read as "ground truth says nobody was injured" instead
    of "this event's truth table does not record injuries".
    """
    import json
    from datetime import datetime

    raw = json.loads(path.read_text(encoding="utf-8"))
    onset_raw = raw.get("onset_utc")
    if not onset_raw or not raw.get("points"):
        return None
    onset = datetime.fromisoformat(onset_raw.replace("Z", "+00:00"))

    points: List[tuple] = []
    for pt in raw["points"]:
        snap = pt.get("snapshot")
        if not snap:
            continue
        hours = (datetime.fromisoformat(snap.replace("Z", "+00:00")) - onset).total_seconds() / 3600.0
        if "deaths" in pt:
            deaths = pt["deaths"]
            if "Total" in deaths:
                total = float(deaths["Total"])
            else:  # no Total row: sum the places, skipping private keys
                total = sum(float(v) for k, v in deaths.items() if not k.startswith("_"))
        else:
            total = sum(float(pt[k]) for k in ("turkey", "syria") if k in pt)
        points.append((hours, total))
    if not points:
        return None
    points.sort()

    # Step interpolation, not linear: a cumulative toll holds its last reported value
    # until the next report. Linear would invent a smooth rise nobody reported.
    def at(t: float) -> Optional[float]:
        last = None
        for h, v in points:
            if h <= t:
                last = v
            else:
                break
        return last

    return {"dead": [at(t) for t in grid],
            "injured": [None] * len(grid),
            "missing": [None] * len(grid)}


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
        # Resolve through app._map_location, NOT straight into map_location: "auto" is not
        # a torch device and would raise, and "cuda" on a box without CUDA would raise
        # rather than falling back. The single-document path already resolved it this way;
        # this path did not, so exposing a device control here without this makes the
        # "auto" option an error. Imported lazily -- app imports this module.
        from app import _map_location
        device = _map_location(params.get("device") or "cpu")

        # Administrative rollup: folds city/county keys up to their state and multi-state
        # phrases to __aggregate__. Resolved beside the feed by convention when not given,
        # so the Helene and Turkiye feeds pick theirs up automatically.
        rollup = None
        rollup_path = params.get("rollup")
        if rollup_path is None:
            guess = feed_path.parent.parent / "rollup.json"
            rollup_path = str(guess.relative_to(REPO)) if guess.is_file() else ""
        if rollup_path:
            rp_file = REPO / rollup_path
            if rp_file.is_file():
                rollup = json.loads(rp_file.read_text(encoding="utf-8"))
                job.log.append(f"rollup: {rollup_path}")

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
                if window == "long":
                    # The research configuration, and the viewer lacked it entirely.
                    # extract_long chunks the WHOLE document with overlap; on Helene it
                    # took `dead` observations 25 -> 106, so a viewer without it cannot
                    # reproduce any published number.
                    chunk = int(params.get("chunk_size", 200))
                    records = (cas_model.extract_long(
                        row["text"], cas_schema,
                        threshold=float(params.get("event_threshold", 0.3)),
                        chunk_size=chunk,
                        chunk_overlap=int(params.get("chunk_overlap", 50)),
                    ).get("casualty_report") or [])
                    entry["envelopes"] = [{"text": f"<extract_long chunks of {chunk} words>"}]
                    for rec in records:
                        key = (rp.record_key(events[i], rec) if associate == "record"
                               else rp.association_key(events[i], associate, {}))
                        rp._emit(rec, row["text"], row, entry, modes, per_mode,
                                 gate_model, cls_schema, event_key=key,
                                 event_year=int(params.get("event_year", 0) or 0))
                else:
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
                                     gate_model, cls_schema, event_key=key,
                                     event_year=int(params.get("event_year", 0) or 0))
            articles.append(entry)
            job.done = i + 1

        job.stage = "tracking"

        # Plausibility ceiling: drop anything above the largest credible toll for THIS
        # event, before tracking. Measured 2026-08-20 on Helene -- dropping a single
        # 94,000 (Asheville's POPULATION, read as a death toll) takes ungated per-place
        # error from 378.809 to 18.287. Off by default: it is prior knowledge about one
        # event, not a general rule.
        ceiling = float(params.get("max_plausible", 0) or 0)
        culled = 0
        if ceiling > 0:
            for m in modes:
                before = len(per_mode[m])
                per_mode[m] = [o for o in per_mode[m] if float(o["value"]) <= ceiling]
                culled += before - len(per_mode[m])
            # TAG rather than delete. The tracker must not see these, but deleting them
            # from the article makes "why did this document contribute nothing" an
            # unanswerable question -- only an aggregate count survived, so a document
            # that produced a 94,000 and a document that produced nothing looked
            # identical. The per-document trace needs the difference.
            for entry in articles:
                for o in entry["observations"]:
                    if float(o["value"]) > ceiling:
                        o["dropped"] = f"above plausibility ceiling {ceiling:.0f}"
            job.log.append(f"plausibility ceiling {ceiling:.0f}: dropped {culled}")

        t0, t1 = feed[0]["t_hours"], feed[-1]["t_hours"]
        step = float(params.get("grid_step", 6.0))
        grid = [t0 + k * step for k in range(int((t1 - t0) / step) + 1)]
        tracked = {m: rp.track(per_mode[m], grid) for m in modes}
        # Per-stream tracking is the point of associating at all; track_by_event also
        # folds clipped location keys (syr -> syria) via merge_prefix_keys, and applies
        # the administrative rollup when one is given -- the viewer omitted the rollup
        # entirely, so city/county keys never folded up to their state and per-state
        # streams could not be compared with the research runs.
        tracked_by_event = {m: rp.track_by_event(per_mode[m], grid, rollup) for m in modes}

        result: Dict[str, Any] = {
            "feed": str(feed_path.relative_to(REPO)), "grid": grid,
            "articles": articles, "n_articles": len(feed), "n_relevant": len(kept),
            "tracked": tracked, "tracked_by_event": tracked_by_event,
            "associate": associate, "mode": modes[0],
            "n_observations": {m: len(per_mode[m]) for m in modes},
            # Record what produced this, for the reason PROVENANCE.md gives: the archived
            # 2026-08-10 Helene artifact stored only `associate`, and is now unreproducible.
            "invocation": {"params": dict(params), "rollup": rollup_path or None,
                           "plausibility_ceiling": ceiling or None,
                           "plausibility_dropped": culled,
                           "git_commit": rp._git_commit()},
        }
        # Two truth formats, and only the JSONL one was ever resolved -- so the three real
        # events had no gold overlay despite shipping ground truth.
        truth_path = params.get("truth")
        if not truth_path:
            jsonl = feed_path.with_name(feed_path.stem + ".truth.jsonl")
            gt = feed_path.parent.parent / "ground_truth.json"
            truth_path = str((jsonl if jsonl.is_file() else gt).relative_to(REPO))
        tp = REPO / truth_path
        if tp.is_file() and tp.suffix == ".json":
            series = _ground_truth_series(tp, grid)
            if series:
                result["truth"] = series
                result["truth_note"] = "deaths only; this event's truth table records no injured/missing"
        elif tp.is_file():
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
