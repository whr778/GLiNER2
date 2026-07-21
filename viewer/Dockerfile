# GLiNER2 viewer -- one container running the FastAPI backend (:8000) and the
# Next.js frontend (:3000). The build CONTEXT is the repo root, because the
# backend installs the local gliner2 package (editable ../..) and reads scripts/.
#
#   # build (from the repo root):
#   docker build -f viewer/Dockerfile -t gliner2-viewer .
#
#   # run (data/ + out/ are mounted, not baked in):
#   docker run --rm -p 3000:3000 -p 8000:8000 \
#     -v "$PWD/data:/app/data:ro" -v "$PWD/out:/app/out:ro" \
#     -v gliner2-hf-cache:/root/.cache/huggingface \
#     gliner2-viewer
#
#   # add --gpus all for GPU (the image ships CUDA torch; without the flag it
#   # falls back to CPU automatically -- GLiNER2 picks the device on load):
#   docker run --gpus all --rm -p 3000:3000 -p 8000:8000 ... gliner2-viewer
#
# For a remote host, rebuild the frontend against the reachable backend URL:
#   docker build -f viewer/Dockerfile \
#     --build-arg NEXT_PUBLIC_API_BASE=http://SERVER:8000 -t gliner2-viewer .
#
# The GPU box is x86_64; on an arm64 machine build for it with
# --platform=linux/amd64 (or just build on the box).

FROM python:3.12-slim-bookworm

# Node 20 (Next 15 needs >= 18.18); curl for the NodeSource installer; git for
# any VCS-sourced pip deps; build-essential to compile the fasttext C++ ext
# (gliner2 -> lumi-language-id-2 -> fasttext-numpy2 ships sdist only).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# uv (Python package manager).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# --- backend deps (installs the local gliner2[local] -> torch/transformers) ---
# The gliner2 editable build reads pyproject (version attr in __init__, LICENSE,
# README), so copy those + the package before `uv sync`.
COPY pyproject.toml README.md LICENSE ./
COPY gliner2/ ./gliner2/
COPY viewer/backend/pyproject.toml viewer/backend/uv.lock ./viewer/backend/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd viewer/backend && uv sync --frozen --no-dev
COPY viewer/backend/ ./viewer/backend/
COPY scripts/ ./scripts/

# --- frontend build (Next production build; API base is inlined at build) ---
COPY viewer/frontend/package.json viewer/frontend/package-lock.json ./viewer/frontend/
RUN --mount=type=cache,target=/root/.npm \
    cd viewer/frontend && npm ci
COPY viewer/frontend/ ./viewer/frontend/
ARG NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
RUN cd viewer/frontend && npm run build

COPY viewer/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Persist HF downloads across restarts by mounting this path.
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 3000 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/health || exit 1
CMD ["/app/start.sh"]
