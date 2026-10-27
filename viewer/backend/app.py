"""FastAPI backend for the GLiNER2 extraction viewer.

Loads a GLiNER2 model and exposes extraction over the long-document path so the
frontend can render entities, relations, events, classifications, and
structures. Reuses gliner2's ``Schema.from_dict`` + ``batch_extract_long`` (the
same call as ``tools/infer.py``); the library itself is untouched.

Run: ``uv run uvicorn app:app --reload --port 8000`` (from viewer/backend/).
The default model and the data/model roots are read from ``viewer/config.yaml``
(see ``config.py`` / ``viewer/config.example.yaml``); ``GLINER2_MODEL`` overrides
the default model.
"""

from __future__ import annotations

import os
import threading

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

import config

# Log through uvicorn's configured logger so lines like "Loaded model ... on
# device mps" actually surface in the server console (a bare getLogger has no
# handler under uvicorn and is swallowed).
logger = logging.getLogger("uvicorn.error")

DEFAULT_MODEL = config.default_model()

# Loaded models cached by id (loading torch weights is expensive).
_models: Dict[str, Any] = {}


def _map_location(requested: Optional[str]) -> Optional[str]:
    """Requested device -> from_pretrained map_location. ``None``/``auto`` lets the
    library pick (CUDA -> MPS -> CPU); an unavailable accelerator falls back to
    auto rather than erroring. Note: for many-label event models MPS is markedly
    slower than CPU (per-op sync overhead), so 'cpu' is often the fast choice."""
    import torch

    if requested in (None, "", "auto"):
        return None
    if requested == "cuda" and not torch.cuda.is_available():
        return None
    if requested == "mps" and not torch.backends.mps.is_available():
        return None
    return requested


def get_model(model_id: str, device: Optional[str] = None):
    # AutoExtractor, not GLiNER2: `GLiNER2` IS the span class, so loading a
    # boundary checkpoint through it dies on `config.max_width` -- a span-only
    # field -- long after the download succeeded. The error names the config and
    # reads like a bad repo id, which it is not. `ekf.py` already used
    # AutoExtractor; this loader was the last span-only path in the viewer.
    from gliner2 import AutoExtractor

    from models import hub_load_kwargs

    map_location = _map_location(device)
    extra = hub_load_kwargs(model_id)
    key = (model_id, extra.get("subfolder", ""), map_location or "auto")
    if key not in _models:
        model = AutoExtractor.from_pretrained(model_id, map_location=map_location, **extra)
        dev = next(model.parameters()).device
        logger.info(
            "Loaded model %s subfolder=%s on device %s (requested %s)",
            model_id, extra.get("subfolder") or "<root>", dev, device or "auto",
        )
        _models[key] = model
    return _models[key]


@asynccontextmanager
async def _lifespan(_app):
    import jobs

    jobs.load()
    jobs.start(_run_inference_job)
    yield
    jobs.stop()


app = FastAPI(title="GLiNER2 Viewer API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    # The frontend proxies /api/* to this backend same-origin (see the viewer's
    # next.config.mjs), so browser calls need no CORS. This localhost allowlist
    # only covers pointing a browser directly at the backend during local dev;
    # it deliberately does NOT allow arbitrary cross-origin reads of this
    # unauthenticated API.
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Options(BaseModel):
    # ``model`` collides with pydantic's protected ``model_`` namespace; allow it.
    model_config = ConfigDict(protected_namespaces=())
    threshold: float = 0.5
    chunk_size: int = 384
    chunk_overlap: int = 128
    global_decode: bool = False
    beam_width: int = 8
    model: Optional[str] = None
    device: Optional[str] = None  # auto | cpu | mps | cuda (unavailable -> auto)
    # TODO 19/22: the record (instance) gate and the argument (field) gate, separate from
    # the span threshold. None = the model's own setting (its checkpoint config).
    record_threshold: Optional[float] = None
    argument_threshold: Optional[float] = None
    # None keeps the checkpoint. Same fields as infer.py --merge-coreferent / --coref-threshold.
    merge_coreferent: Optional[Literal["off", "args", "link"]] = None
    coref_threshold: Optional[float] = None
    # None leaves entity mention clusters out of the result. A number runs the mention link.
    entity_coref_threshold: Optional[float] = None


class ExtractRequest(BaseModel):
    # Accept the JSON key ``schema`` without shadowing BaseModel.schema.
    model_config = ConfigDict(populate_by_name=True)
    text: str
    input_schema: Dict[str, Any] = Field(alias="schema")
    options: Options = Field(default_factory=Options)


class UrlRequest(BaseModel):
    url: str


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "data_root": str(config.data_root()),
        "models_root": str(config.models_root()),
        "loaded_models": list(_models),
    }


def _declare_records(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Give every structure an explicit ``mode``/``anchor`` when it has none.

    On the BOUNDARY architecture a structure without record metadata cannot be decoded,
    and the failure is SILENT -- the extraction returns ``{}`` with no error, which in the
    viewer reads as "this model found nothing" rather than "this schema was undecodable".
    Measured on `whr778/gliner2-tr-dose-15000`, same text and threshold:

        without mode/anchor -> {}
        with mode/anchor    -> {'casualty_report': [{'dead': '22', 'injured': '40', ...}]}

    Declaring it is safe on span models, whose legacy structure path ignores the metadata
    -- `build_casualty_schema` records that plain and declared schemas give byte-identical
    output there. So this defaults rather than asking the user to know which architecture
    they loaded. An explicit mode in the request is always left alone.
    """
    structures = input_schema.get("structures")
    if not isinstance(structures, dict):
        return input_schema
    out = dict(input_schema)
    out["structures"] = {}
    for name, spec in structures.items():
        if not isinstance(spec, dict) or spec.get("mode"):
            out["structures"][name] = spec
            continue
        fields = spec.get("fields") or []
        first = next((f.get("name") for f in fields if isinstance(f, dict) and f.get("name")), None)
        spec = dict(spec, mode="natural")
        if first and not spec.get("anchor"):
            spec["anchor"] = first
        out["structures"][name] = spec
    return out



_GATE_LOCK = threading.Lock()
# apply_boundary_overrides only writes keys. A checkpoint that omitted one
# has to have that key removed again, or the override stays on the cached model.
_DROP = object()


def _gate_overrides(options: "Options") -> Dict[str, Any]:
    """Boundary-head overrides for this request's record and argument gates (TODO 19/22).

    Merge fields match ``infer.py --merge-coreferent`` and ``--coref-threshold``.
    None leaves the checkpoint's ``record_merge_coreferent`` and threshold.
    """
    ov: Dict[str, Any] = {}
    if options.record_threshold is not None:
        ov.update(record_anchor_threshold=options.record_threshold,
                  record_anchor_proposal_threshold=options.record_threshold,
                  record_anchor_threshold_wins=True)
    if options.argument_threshold is not None:
        ov.update(record_field_threshold=options.argument_threshold, record_field_threshold_wins=True)
    if options.merge_coreferent is not None:
        ov["record_merge_coreferent"] = options.merge_coreferent
    if options.coref_threshold is not None:
        ov["record_coref_link_threshold"] = options.coref_threshold
    return ov


def _attach_entity_coref(model, text: str, result: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Cluster predicted entity spans and pronouns with the trained mention link.

    The same call as ``infer.py --entity-coref``: multi-mention clusters only.
    It does not add those mentions as event-argument fillers.
    """
    if getattr(model, "mention_link", None) is None:
        raise HTTPException(
            status_code=422,
            detail="entity mention clusters need a checkpoint with a trained mention link",
        )
    from gliner2.inference.entity_coref import candidate_mentions, resolve

    mentions = candidate_mentions(text, result.get("entities"))
    result["coref_clusters"] = [
        c for c in resolve(model, text, mentions, threshold) if len(c) > 1
    ]
    return result


def _apply_gates(model, gates: Dict[str, Any]) -> Dict[str, Any]:
    """Apply gates to a boundary model; return the overrides that restore its prior state.

    A key the checkpoint did not have is returned as ``_DROP``. The restore call
    removes it and rebuilds the head, because ``apply_boundary_overrides`` cannot
    delete a key it just wrote.
    """
    current = getattr(model.config, "boundary_head", None)
    if not gates or not current:
        return {}
    from gliner2.training.eval_metrics import apply_boundary_overrides
    drops = [k for k, v in gates.items() if v is _DROP]
    present = {k: v for k, v in gates.items() if v is not _DROP}
    if drops:
        current = {k: v for k, v in current.items() if k not in drops}
        model.config.boundary_head = current
    restore = {k: current[k] if k in current else _DROP for k in present}
    if present:
        apply_boundary_overrides(model, present)
    elif drops and current:
        key = next(iter(current))
        apply_boundary_overrides(model, {key: current[key]})
    return restore


def _run_inference_job(job) -> None:
    """Decode one queued job on the cached model. The gate lock is held for the whole job."""
    import jobs
    from gliner2.inference.global_decode import GlobalDecodeConfig
    from gliner2.inference.schema import Schema

    opts = Options(**(job.options or {}))
    model = get_model(job.model, opts.device)
    job.label_map = getattr(model.config, "label_map", None)
    job.label_style = getattr(model.config, "label_style", None)
    job.ran_on = str(next(model.parameters()).device)
    if opts.merge_coreferent == "link":
        bh = getattr(model.config, "boundary_head", None) or {}
        if not bh.get("record_coref_link"):
            raise RuntimeError("trigger merge link needs a checkpoint trained with record_coref_link")

    def predict(texts, schemas):
        prepared = [Schema.from_dict(_declare_records(schema)) for schema in schemas]
        out = model.batch_extract_long(
            texts,
            prepared,
            batch_size=len(texts),
            threshold=opts.threshold,
            chunk_size=opts.chunk_size,
            chunk_overlap=opts.chunk_overlap,
            include_spans=True,
            include_confidence=True,
            global_decode=opts.global_decode,
            global_decode_config=GlobalDecodeConfig(beam_width=opts.beam_width),
        )
        if opts.entity_coref_threshold is not None:
            out = [_attach_entity_coref(model, text, result, opts.entity_coref_threshold)
                   for text, result in zip(texts, out)]
        return out

    with _GATE_LOCK:
        restore = _apply_gates(model, _gate_overrides(opts))
        try:
            jobs.execute(job, predict)
        finally:
            _apply_gates(model, restore)


@app.post("/extract")
def extract(req: ExtractRequest) -> Dict[str, Any]:
    from gliner2.inference.global_decode import GlobalDecodeConfig
    from gliner2.inference.label_map import apply_label_map
    from gliner2.inference.schema import Schema

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    # Strip stray whitespace/newlines (a trailing \r from a shell pipeline makes
    # os.path.isdir False -> gliner2 treats it as an HF id -> cryptic
    # HFValidationError on a path that otherwise looks correct).
    model_id = (req.options.model or DEFAULT_MODEL).strip()
    try:
        model = get_model(model_id, req.options.device)
    except Exception as e:  # noqa: BLE001 - surface the real load error, not a bare 500
        raise HTTPException(
            status_code=502,
            detail=(
                f"could not load model '{model_id}': {e}. Pass a local checkpoint "
                "directory that exists in the server (see GET /models for valid "
                "paths) or a HuggingFace repo id 'namespace/name'."
            ),
        ) from e
    # Labels are an INPUT: send the spellings this model trained on (its config.label_map).
    input_schema, label_map_applied = apply_label_map(
        req.input_schema, getattr(model.config, "label_map", None), getattr(model.config, "label_style", None))
    try:
        schema = Schema.from_dict(_declare_records(input_schema))
    except Exception as e:  # noqa: BLE001 - surface schema errors to the client
        raise HTTPException(status_code=422, detail=f"invalid schema: {e}") from e
    gates = _gate_overrides(req.options)
    if req.options.merge_coreferent == "link":
        bh = getattr(model.config, "boundary_head", None) or {}
        if not bh.get("record_coref_link"):
            raise HTTPException(
                status_code=422,
                detail="trigger merge link needs a checkpoint trained with record_coref_link",
            )
    try:
        # The cached model is shared across requests: apply this request's gates, extract,
        # and restore, under one lock so a concurrent request never sees another's gates.
        with _GATE_LOCK:
            restore = _apply_gates(model, gates)
            try:
                result = model.batch_extract_long(
                    [req.text],
                    schema,
                    threshold=req.options.threshold,
                    chunk_size=req.options.chunk_size,
                    chunk_overlap=req.options.chunk_overlap,
                    include_spans=True,
                    include_confidence=True,
                    global_decode=req.options.global_decode,
                    global_decode_config=GlobalDecodeConfig(beam_width=req.options.beam_width),
                )[0]
            finally:
                _apply_gates(model, restore)
            if req.options.entity_coref_threshold is not None:
                result = _attach_entity_coref(
                    model, req.text, result, req.options.entity_coref_threshold)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - surface extraction errors to the client
        raise HTTPException(status_code=500, detail=f"extraction failed: {e}") from e
    device = str(next(model.parameters()).device)
    return {"text": req.text, "result": result, "device": device,
            "label_map_applied": label_map_applied}


@app.post("/import-url")
def import_url(req: UrlRequest) -> Dict[str, str]:
    import trafilatura

    downloaded = trafilatura.fetch_url(req.url)
    if not downloaded:
        raise HTTPException(status_code=400, detail=f"could not fetch: {req.url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="no extractable text at that url")
    return {"text": text}


@app.get("/presets")
def presets() -> Dict[str, Any]:
    from presets import list_presets

    return {"presets": list_presets()}


class ModelEntry(BaseModel):
    path: str
    label: Optional[str] = None


@app.get("/models")
def get_models() -> Dict[str, Any]:
    from models import list_models

    return {"models": list_models(DEFAULT_MODEL)}


@app.get("/model-schema")
def model_schema(model: str) -> Dict[str, Any]:
    """The extraction schema co-located in a model's config, if any.

    Read from config.json only (no weights loaded) so the frontend can offer the
    model's own ontology the instant a model is selected. ``None`` for models
    saved before schemas were co-located; the frontend then falls back to
    corpus-name matching.
    """
    from gliner2.configuration import ExtractorConfig
    from models import hub_load_kwargs

    cfg = ExtractorConfig.from_pretrained(model, **hub_load_kwargs(model))
    bh = getattr(cfg, "boundary_head", None) or {}
    return {"model": model, "schema": getattr(cfg, "default_schema", None),
            # The model's own gates; None means "follows the span threshold".
            "record_defaults": {
                "record_threshold": bh.get("record_anchor_threshold") if bh.get("record_anchor_threshold_wins") else None,
                "argument_threshold": bh.get("record_field_threshold") if bh.get("record_field_threshold_wins") else None,
                "boundary": bool(bh)},
            # Absent merge mode is off. The threshold default is the head's own 0.5.
            "coref": {
                "record_merge_coreferent": bh.get("record_merge_coreferent") or "off",
                "record_coref_link": bool(bh.get("record_coref_link")),
                "record_coref_link_threshold": float(bh.get("record_coref_link_threshold", 0.5)),
                "entity_coref_link": bool(bh.get("entity_coref_link")),
            },
            "inference_defaults": getattr(cfg, "inference_defaults", None),
            "has_label_map": bool(getattr(cfg, "label_map", None))}


@app.post("/models")
def create_model(entry: ModelEntry) -> Dict[str, Any]:
    from models import add_model, list_models

    if not entry.path.strip():
        raise HTTPException(status_code=400, detail="path is empty")
    add_model(entry.path, entry.label)
    return {"models": list_models(DEFAULT_MODEL)}


@app.delete("/models")
def delete_model(path: str) -> Dict[str, Any]:
    from models import list_models, remove_model

    remove_model(path)
    return {"models": list_models(DEFAULT_MODEL)}

# --------------------------------------------------------------------------- #
# EKF tracking (news feed -> observations -> tracked timeline)
# --------------------------------------------------------------------------- #
class EkfRequest(BaseModel):
    """A tracking run. Defaults mirror tools/ekf_showcase/run_pipeline.py."""
    model_config = ConfigDict(protected_namespaces=())

    feed: str
    truth: Optional[str] = None
    gate_model: str = "fastino/gliner2-base-v1"
    casualty_model: str = "whr778/gliner2-base-v1-casualty-docee"
    event_model: Optional[str] = None
    window: str = "article"          # article | lead | event | long
    normalizer: str = "hybrid"
    gate_threshold: float = 0.5
    event_threshold: float = 0.3
    grid_step: float = 6.0
    device: Optional[str] = "cpu"
    limit: int = 0
    # Everything below was READ by ekf.py and never DECLARED here, so model_dump() dropped
    # it and the runner silently used its fallback. `associate` was the worst case: the
    # viewer could not associate at all, which is the failure association exists to fix
    # (multi-event nRMSE 102 -> 27.975). Declared 2026-08-20.
    associate: str = "none"          # none | type | type+location | envelope | record
    rollup: Optional[str] = None     # None resolves one beside the feed; "" disables
    lead_chars: int = 1100
    envelope_margin: int = 60
    chunk_size: int = 200            # --window long: the framing curve's band, not 384
    chunk_overlap: int = 50
    # Drop observations above the largest credible toll for THIS event, before tracking.
    # 0 disables. On Helene a ceiling of 2,000 removes a 94,000 that is Asheville's
    # POPULATION and takes ungated per-place error 378.809 -> 18.190.
    max_plausible: float = 0.0
    # Reject casualty figures whose nearest date predates this year. 0 disables. Needed by
    # Turkiye: without it the 1999 Izmit toll was tracked as a 2023 figure in every
    # configuration. Read but never declared until 2026-08-30, so it silently did nothing.
    event_year: int = 0


class JobRequest(BaseModel):
    """A text extraction or a JSONL evaluation. The response is the ticket, not the result."""
    model_config = ConfigDict(populate_by_name=True)
    kind: Literal["text", "jsonl"]
    text: Optional[str] = None
    path: Optional[str] = None
    schema_mode: Literal["gold", "panel"] = "panel"
    input_schema: Dict[str, Any] = Field(default_factory=dict, alias="schema")
    options: Options = Field(default_factory=Options)
    limit: int = 0


@app.get("/datasets")
def datasets(q: str = "") -> Dict[str, Any]:
    """JSONL files under data_root. ``q`` matches the repo-relative path."""
    import jobs

    return {"datasets": jobs.list_datasets(q)}


@app.post("/jobs")
def create_job(req: JobRequest) -> Dict[str, Any]:
    import jobs

    model = (req.options.model or DEFAULT_MODEL).strip()
    try:
        job = jobs.create(
            req.kind, req.text, req.path, req.schema_mode, req.input_schema,
            req.options.model_dump(), req.limit, model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job.public()


@app.get("/jobs")
def get_jobs() -> Dict[str, Any]:
    import jobs

    return {"jobs": [job.public() for job in jobs.list_jobs()]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    import jobs

    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return job.public(full=True)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    import jobs

    job = jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return job.public()


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> Dict[str, Any]:
    import jobs

    job = jobs.delete(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return job.public()


@app.get("/predictions")
def predictions(path: str) -> Dict[str, Any]:
    """Open a ``tools/infer.py --output`` file: per-record strict F1 for the pager."""
    import gold

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"no such file: {path}")
    return gold.summary(path)


@app.get("/predictions/files")
def prediction_files() -> Dict[str, Any]:
    """The infer.py output files under the configured ``predictions_root``, newest first."""
    import gold

    root = config.predictions_root()
    return {"root": str(root), "files": gold.list_files(str(root)) if root.is_dir() else []}


@app.get("/predictions/record")
def prediction_record(path: str, i: int) -> Dict[str, Any]:
    """One record of an infer.py output file: prediction and gold, every span flagged ``match``."""
    import gold

    return gold.record(path, i)


@app.get("/ekf-feeds")
def ekf_feeds() -> Dict[str, Any]:
    import ekf

    return {"feeds": ekf.list_feeds()}


@app.post("/ekf-track")
def ekf_track(req: EkfRequest) -> Dict[str, Any]:
    """Start a run. Returns a job id immediately -- a run is minutes long, well
    past any sensible request timeout, so the frontend polls /ekf-track/{id}."""
    import ekf

    try:
        ekf.resolve_feed(req.feed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    job = ekf.start(req.model_dump())
    return job.public()


@app.get("/ekf-track/{job_id}")
def ekf_status(job_id: str) -> Dict[str, Any]:
    import ekf

    job = ekf.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return job.public()
