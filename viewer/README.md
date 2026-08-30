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

## Run with Docker

One container runs both services (backend :8000 + frontend :3000). `data/` and
the model checkpoints (`out/`) are **mounted**, not baked into the image, and
GLiNER2 auto-selects the GPU when the container has one.

Once the image is built, `viewer/docker.sh` is the easy way to run it (it wraps
the `docker run` below with the ports, mounts, HF-cache volume, and GPU flag):

```bash
bash viewer/docker.sh start      # run the container, wait until it's up
bash viewer/docker.sh stop       # stop + remove it
bash viewer/docker.sh restart
bash viewer/docker.sh status
bash viewer/docker.sh logs       # follow container logs
```

GPU is auto-detected (adds `--gpus all` when `nvidia-smi` is present); force it
with `GPU=1`, disable with `GPU=0`, and pick the model with `GLINER2_MODEL=...`.
Or run the raw commands:

```bash
# build (from the repo root -- the backend installs the local gliner2 package):
docker build -f viewer/Dockerfile -t gliner2-viewer .

# run on CPU (data/ + out/ mounted read-only; HF cache persists downloads).
# Only :3000 is published -- the browser reaches the backend via the /api proxy;
# :8000 is loopback-only (the backend is unauthenticated, keep it off the network):
docker run --rm -p 3000:3000 -p 127.0.0.1:8000:8000 \
  -v "$PWD/data:/app/data:ro" -v "$PWD/out:/app/out:ro" \
  -v gliner2-hf-cache:/root/.cache/huggingface \
  gliner2-viewer

# run on GPU: just add --gpus all -- the image ships CUDA torch, no rebuild.
docker run --gpus all --rm -p 3000:3000 -p 127.0.0.1:8000:8000 \
  -v "$PWD/data:/app/data:ro" -v "$PWD/out:/app/out:ro" \
  -v gliner2-hf-cache:/root/.cache/huggingface \
  gliner2-viewer
```

Then open **http://localhost:3000**. Set the default model with
`-e GLINER2_MODEL=<hf-id-or-/app/out/...>`. Notes:

- The GPU box is x86_64; on an arm64 machine build for it with
  `docker build --platform=linux/amd64 ...`, or build on the box itself.
- Serving the UI from a remote host? Nothing to rebuild. The browser calls
  `/api/*` on the page's own origin and Next proxies to the co-located backend,
  so `http://SERVER:3000` works as-is (no baked backend host, no CORS). Just make
  sure the host firewall allows port 3000 (`firewalld` on EL9 blocks it by
  default), or tunnel it: `ssh -L 3000:localhost:3000 SERVER`.

### On an EL9 / Intel-64 GPU box

`viewer/docker-build.sh` builds a `linux/amd64` image by default:

```bash
viewer/docker-build.sh                                  # -> gliner2-viewer
API_BASE=http://THIS_HOST:8000 viewer/docker-build.sh   # if accessed remotely
```

Build on the box for a native, fast build (cross-building amd64 from an arm64
Mac works but is slow -- then `docker save gliner2-viewer | ssh box docker load`).

The image uses a **Red Hat UBI9 base**, so it builds cleanly even on a
**FIPS-enabled RHEL9** host (a Debian base aborts here: `apt`'s libgcrypt refuses
FIPS-disabled MD5 -> `_gcry_fatal_error`, exit 134). If you prefer not to build
on the FIPS host at all, build off-host and `docker save | docker load` (the
runtime only needs TLS, which is FIPS-fine).

Host prerequisites on EL9 (once): Docker CE + the NVIDIA Container Toolkit.

```bash
# Docker CE
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo systemctl enable --now docker

# NVIDIA Container Toolkit (needs the NVIDIA driver already installed)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # verify
```

Then build (script above) and `docker run --gpus all ...` as shown earlier.

#### Custom mount paths (restrictive EL9 hosts)

On a locked-down box, `docker run` may fail preparing the `data/`/`out/` bind
mounts — `error creating source path ... mkdir ...: permission denied` on a path
that *already exists*, even though your own perms are fine. That's the engine
(not you) unable to reach the source: an **NFS home with `root_squash`**, an
**SELinux-restricted `/home`**, or **rootless Docker** (its user namespace maps
your primary uid/gid but not your supplementary groups, and can't map NFS homes).

Fix: mount from a **local path you own** and force the SELinux relabel.
`docker.sh` takes `DATA_DIR`/`OUT_DIR` (mount sources) and `SELINUX_RELABEL=1`
(force the `:z` relabel):

```bash
WORK=/var/tmp/gliner2                     # a LOCAL path you own (not NFS, not /home)
mkdir -p "$WORK/data" "$WORK/out"
cp -a data/*.jsonl "$WORK/data/" 2>/dev/null || true   # optional: schema presets
cp -a out/*        "$WORK/out/"  2>/dev/null || true   # optional: your checkpoints

bash viewer/docker-build.sh               # build the image (once)
DATA_DIR="$WORK/data" OUT_DIR="$WORK/out" SELINUX_RELABEL=1 GPU=1 \
  bash viewer/docker.sh start
```

Empty `data/`/`out/` are fine — the default `fastino/gliner2-base-v1` still runs
(downloaded into the `gliner2-hf-cache` volume). The equivalent raw run:

```bash
docker run -d --name gliner2-viewer --gpus all -p 3000:3000 -p 127.0.0.1:8000:8000 \
  -v "$WORK/data:/app/data:ro,z" -v "$WORK/out:/app/out:ro,z" \
  -v gliner2-hf-cache:/root/.cache/huggingface gliner2-viewer
```

## Configuration

The backend reads the data corpus and model-checkpoint locations from an
optional `viewer/config.yaml` (git-ignored) — copy `viewer/config.example.yaml`
and edit, or point `$GLINER2_VIEWER_CONFIG` at any file. Without one it uses
repo-relative defaults, so it works out of the box.

```yaml
data_root: data      # scanned for <name>.train.jsonl -> corpus schema presets
models_root: out     # scanned for **/best -> auto-discovered models
default_model: fastino/gliner2-base-v1   # HF id or a checkpoint path
```

Relative paths resolve against the repo root; `~` and absolute paths work — so
you can point the viewer at data/models anywhere. `GLINER2_MODEL` still overrides
`default_model`. `GET /health` reports the resolved `data_root`/`models_root`.

### Model list

The combobox merges three sources: the server default, anything auto-discovered
under `models_root` (`**/best`), and a saved registry at
`viewer/backend/models.json` (git-ignored, written by the Model Manager UI and
by `POST /models`).

`viewer/backend/models.example.json` holds the **60 published checkpoints** —
every `whr778/*` model on the Hub plus the Fastino bases. Copy it to seed the
registry:

```bash
cp viewer/backend/models.example.json viewer/backend/models.json
```

**Most of these are PRIVATE — 28 of the 53 `whr778/*` entries — so the backend
needs a Hugging Face token or they fail to load with a 401.** Export one before
starting; read scope is enough:

```bash
export HF_TOKEN=hf_...        # or: huggingface-cli login
bash viewer/viewer.sh start
```

Without this you only get the default plus local checkpoints, which means the
casualty models the EKF panel needs are not selectable. Labels are family-prefixed
(`casualty:`, `boundary:`, `warmstart:`, `scaling:`, `tier2:`) so the list sorts
into groups; `../tools/events_working_papers/MODEL_LINEAGE.md` says which config
trained each one.

<details>
<summary>Or run the two services by hand</summary>

```bash
# backend
cd viewer/backend && uv run uvicorn app:app --host 127.0.0.1 --port 8000
# frontend (separate terminal)
cd viewer/frontend && npm run dev
```
The frontend proxies `/api/*` to `http://127.0.0.1:8000` by default; point that
proxy at a non-default backend with `GLINER2_BACKEND_ORIGIN`
(e.g. `GLINER2_BACKEND_ORIGIN=http://host:8000 npm run dev`). To make the browser
call an external backend directly (bypassing the proxy), set
`NEXT_PUBLIC_API_BASE` to an absolute URL instead.
</details>

## Schema (what to extract)

GLiNER2 is schema-driven, so you tell it *what* to look for. The Schema panel
offers:

- **Presets** — built-ins (General entities, Sentiment, ACE-style events,
  Contact structure, plus the EKF pipeline's own two: **Casualty report
  (stage 2)** and **Relevance gate (stage 0)**) **plus one auto-derived from every `data/<name>.train.jsonl`
  in the repo** (e.g. `corpus: wikievents`, `corpus: casie`, `corpus: duee`), so
  you can one-click the exact ontology a trained model expects.
- **Schema JSON** editor — the `SchemaInput` form:

```json
{
  "entities": ["person", "organization", "location"],
  "events": {"Attack": ["Attacker", "Target", "Place", "Time"]},
  "relations": ["works_for", "located_in"],
  "classifications": [{"task": "sentiment", "labels": ["positive", "negative", "neutral"]}],
  "structures": {"product": {"mode": "natural", "anchor": "name",
                             "fields": [{"name": "name", "dtype": "str"},
                                        {"name": "price", "dtype": "str"}]}}
}
```

**`mode` and `anchor` matter on boundary models.** A structure declared without
them cannot be decoded on the BOUNDARY architecture, and the failure is *silent* —
extraction returns `{}` with no error, which reads in the UI as "the model found
nothing". Measured on `whr778/gliner2-tr-dose-15000`, same text and threshold:

| schema | result |
|---|---|
| no `mode`/`anchor` | `{}` |
| `mode: natural, anchor: dead` | `{"casualty_report": [{"dead": "22", "injured": "40", ...}]}` |

The backend fills them in when omitted (`_declare_records` in `app.py`), so the
editor is forgiving — but declare them anyway in schemas you copy elsewhere.
Span models ignore the metadata, so declaring it is free.

## EKF tracking panel

The viewer also drives the **event-tracking** line, not just single-document
extraction: a time-ordered news feed in, a tracked casualty timeline out. The
backend imports `tools/ekf_showcase/run_pipeline.py` directly, so the panel and
the CLI run the *same* code — see `tools/ekf_showcase/README.md`.

**Fixed 2026-08-30 — the three real event feeds were not selectable.** Feed discovery
used a non-recursive glob and stopped at the first directory that matched, so it only
ever offered `datasets/ekf_showcase/*.jsonl`. The events this project is actually about
sit one level deeper and were invisible: **Helene (70 articles), Aegean (71), Türkiye
(16)**. The rollup logic below already resolved paths like
`datasets/helene2024/_cache/feed.jsonl`, so the backend was ready for feeds the dropdown
could never offer. Discovery is now recursive over every directory, skips `train/val/test`
splits (whose observation files look like feeds but are 10k–45k-row training corpora),
finds `ground_truth.json` as well as `<stem>.truth.jsonl`, and lists truth-bearing feeds
first. Feeds are labelled by event directory, since all three real ones are named
`feed.jsonl` and would otherwise render as three identical entries.

**Model selection is a dropdown**, fed by the same registry as the main panel, rather than
two free-text boxes that had to be typed exactly. A value not in the registry (a local
path, or a checkpoint added since load) stays selectable, so switching model never happens
silently. The stage-1 event model has an explicit `none — skip stage 1` option.

**Realigned with the research pipeline 2026-08-20**, after it had drifted for the
second time. Four gaps are closed, each of which made the panel unable to reproduce a
published number:

- **`window: long`** now exists here. It chunks the whole document with overlap and is
  the research default; on Helene it takes `dead` observations 25 → 106. Without it the
  panel could not get near a published figure.
- **The administrative rollup is applied.** It is resolved beside the feed by
  convention, so `datasets/helene2024/_cache/feed.jsonl` picks up
  `datasets/helene2024/rollup.json` automatically. Previously omitted entirely, so
  city and county keys never folded up to their state.
- **`associate` actually reaches the runner.** The frontend had always sent it and the
  runner had always read it, but it was never declared on `EkfRequest`, so Pydantic
  dropped it and every run silently pooled into one stream — the exact failure
  association exists to fix. Six other read-but-undeclared fields were fixed with it.
- **A per-event plausibility ceiling** (`Max plausible`, 0 = off). Anything above the
  largest credible toll for the event is dropped before tracking. On Helene a ceiling
  of 2,000 removes a 94,000 that is *Asheville's population* read as a death toll, and
  is worth roughly 20× on ungated per-place error.

Every run now records its full parameters, the rollup used, the ceiling, and the git
commit — with a `-dirty` marker — in `result.invocation`. That exists because the
archived 2026-08-10 Helene artifact stored only `associate` and is consequently
unreproducible; see `tools/ekf_showcase/muting_arm_results/PROVENANCE.md`.

Pick a feed, pick the models, press run. Four stages: **gate** (is this a
mass-casualty report?) → **event** (type + "Casualties and Losses" spans) →
**extract** (bind numbers to `{dead, injured, missing}`) → **track** (EKF plus a
`last_value` baseline).

Feeds are auto-discovered from `datasets/ekf_showcase/*.jsonl`, then
`datasets/*.jsonl`. A file qualifies if its first record has `t_hours` and
`text`; a sibling `<stem>.truth.jsonl` is picked up as ground truth
automatically.

A run takes **minutes**, well past any sensible request timeout, so `POST
/ekf-track` returns a job id immediately and the frontend polls
`GET /ekf-track/{job_id}` for stage, progress and log lines.

Defaults mirror the CLI: gate `fastino/gliner2-base-v1`, casualty
`whr778/gliner2-base-v1-casualty-docee`, `normalizer: hybrid`,
`window: article`, `device: cpu`. **CPU is deliberate** — MPS is 3-4x *slower*
for many-label event extraction, because the per-op sync overhead dominates.

## API (backend)

| Endpoint | Body | Returns |
|---|---|---|
| `POST /extract` | `{text, schema, options}` | `{text, result}` (flat task-keyed dict) |
| `POST /import-url` | `{url}` | `{text}` (article text via trafilatura) |
| `GET /presets` | — | `{presets: [{name, schema, source}]}` |
| `GET /models` | — | `{models: [{path, label, source}]}` — default, saved, discovered |
| `POST /models` | `{path, label}` | adds a saved model entry |
| `DELETE /models` | `{path}` | removes a saved entry (saved only) |
| `GET /model-schema` | `?model=` | the ontology a checkpoint expects |
| `GET /ekf-feeds` | — | `{feeds: [{path, articles, truth}]}` |
| `POST /ekf-track` | `EkfRequest` | `{job_id, ...}` — starts an async run |
| `GET /ekf-track/{job_id}` | — | `{stage, done, total, log, result}` |
| `GET /health` | — | `{status, default_model, loaded_models}` |

`options`: `threshold`, `chunk_size`, `chunk_overlap`, `global_decode`,
`beam_width`, `model`.

`EkfRequest`: `feed` (required), `truth`, `gate_model`, `casualty_model`,
`event_model`, `window`, `normalizer`, `gate_threshold`, `event_threshold`,
`grid_step`, `device`, `limit`.

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
- Every event corpus gets a schema preset. Full-event sets (casie, rams, duee,
  cmnee, wikievents) expose their `events` (type + roles). Trigger-only sets
  (maven, mendeley_ed) have no roles, so event detection is offered as
  **entities** (one label per event type) — the model detects the triggers as
  typed spans.
