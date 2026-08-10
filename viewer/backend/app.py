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

import logging
from typing import Any, Dict, Optional

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
    from gliner2 import GLiNER2

    map_location = _map_location(device)
    key = (model_id, map_location or "auto")  # cache one copy per (model, device)
    if key not in _models:
        model = GLiNER2.from_pretrained(model_id, map_location=map_location)
        dev = next(model.parameters()).device
        logger.info("Loaded model %s on device %s (requested %s)", model_id, dev, device or "auto")
        _models[key] = model
    return _models[key]


app = FastAPI(title="GLiNER2 Viewer API")
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


@app.post("/extract")
def extract(req: ExtractRequest) -> Dict[str, Any]:
    from gliner2.inference.global_decode import GlobalDecodeConfig
    from gliner2.inference.schema import Schema

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    try:
        schema = Schema.from_dict(req.input_schema)
    except Exception as e:  # noqa: BLE001 - surface schema errors to the client
        raise HTTPException(status_code=422, detail=f"invalid schema: {e}") from e

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
    except Exception as e:  # noqa: BLE001 - surface extraction errors to the client
        raise HTTPException(status_code=500, detail=f"extraction failed: {e}") from e
    device = str(next(model.parameters()).device)
    return {"text": req.text, "result": result, "device": device}


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

    cfg = ExtractorConfig.from_pretrained(model)
    return {"model": model, "schema": getattr(cfg, "default_schema", None)}


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
    window: str = "article"
    normalizer: str = "hybrid"
    gate_threshold: float = 0.5
    event_threshold: float = 0.3
    grid_step: float = 6.0
    device: Optional[str] = "cpu"
    limit: int = 0


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
