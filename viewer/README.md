# GLiNER2 Viewer

An interactive app to visualize what a GLiNER2 model extracts — **entities,
relations, events (trigger + arguments), classifications, and structures** —
over text you **paste, upload (.txt), or import from a URL**.

- **Backend** (`backend/`): FastAPI service that loads a GLiNER2 model and runs
  extraction (reuses `Schema.from_dict` + `batch_extract_long`; the `gliner2`
  library is untouched).
- **Frontend** (`frontend/`): NextJS app — input → **Extract** → annotated text
  with color-coded spans, a per-task breakdown, and confidences.

```
Browser ─▶ NextJS (:3000) ─fetch─▶ FastAPI (:8000) ─▶ GLiNER2 model
```

## Run locally

One-time setup (installs frontend deps): `cd viewer/frontend && npm install`.
The backend's Python env (torch + the model) is set up on its first start.

Then start and stop **both** services with a single script:

```bash
bash viewer/viewer.sh start      # backend :8000 + frontend :3000 (waits until up)
bash viewer/viewer.sh stop       # stop both
bash viewer/viewer.sh restart
bash viewer/viewer.sh status
bash viewer/viewer.sh logs       # tail both logs (Ctrl-C to quit)
```

Then open **http://localhost:3000**, pick a schema preset (or edit the schema
JSON), enter text, and click **Extract**.

Choose the model (blank uses `fastino/gliner2-base-v1`):

```bash
GLINER2_MODEL=out/fastino/gliner2-base-v1-wikievents/best bash viewer/viewer.sh start
```

Logs + PIDs live in `viewer/.run/` (git-ignored). Requires `uv` and `npm` on PATH.

<details>
<summary>Or run the two services by hand</summary>

```bash
# backend
cd viewer/backend && uv run uvicorn app:app --host 127.0.0.1 --port 8000
# frontend (separate terminal)
cd viewer/frontend && npm run dev
```
Point the frontend at a non-default backend with `NEXT_PUBLIC_API_BASE`
(e.g. `NEXT_PUBLIC_API_BASE=http://host:8000 npm run dev`).
</details>

## Schema (what to extract)

GLiNER2 is schema-driven, so you tell it *what* to look for. The Schema panel
offers:

- **Presets** — built-ins (General entities, Sentiment, ACE-style events,
  Contact structure) **plus one auto-derived from every `data/<name>.train.jsonl`
  in the repo** (e.g. `corpus: wikievents`, `corpus: casie`, `corpus: duee`), so
  you can one-click the exact ontology a trained model expects.
- **Schema JSON** editor — the `SchemaInput` form:

```json
{
  "entities": ["person", "organization", "location"],
  "events": {"Attack": ["Attacker", "Target", "Place", "Time"]},
  "relations": ["works_for", "located_in"],
  "classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"]}],
  "structures": {"product": {"fields": [{"name": "name", "dtype": "str"}, {"name": "price", "dtype": "str"}]}}
}
```

## API (backend)

| Endpoint | Body | Returns |
|---|---|---|
| `POST /extract` | `{text, schema, options}` | `{text, result}` (flat task-keyed dict) |
| `POST /import-url` | `{url}` | `{text}` (article text via trafilatura) |
| `GET /presets` | — | `{presets: [{name, schema, source}]}` |
| `GET /health` | — | `{status, default_model, loaded_models}` |

`options`: `threshold`, `chunk_size`, `chunk_overlap`, `global_decode`,
`beam_width`, `model`.

## Deploy as a webapp

- **Backend**: any Python host. `uv run uvicorn app:app --host 0.0.0.0 --port
  8000` (CPU is fine for demos; a GPU host speeds large models). Containerize
  with a small Dockerfile, or run on HF Spaces (Docker), Render, or Railway.
- **Frontend**: `npm run build && npm run start` (or a static/Node host like
  Vercel). Set `NEXT_PUBLIC_API_BASE` to the deployed backend URL.
- Update the backend CORS `allow_origins` (`app.py`) to include the deployed
  frontend origin.

## Notes

- The result JSON has **no copy of the input text** — the UI pairs
  `result` with the text it sent, using each span's char `start`/`end` to
  highlight in place.
- Trigger-only / classification-only corpora (e.g. `mendeley_ed`) whose derived
  schema has no role-bearing events won't appear as a corpus preset (their
  schema is empty after role-less event types are dropped) — build a schema by
  hand for those.
