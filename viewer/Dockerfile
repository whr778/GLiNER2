# GLiNER2 viewer -- one container running the FastAPI backend (:8000) and the
# Next.js frontend (:3000). Red Hat UBI9 base so it builds cleanly on a
# FIPS-enabled RHEL9 host: dnf/rpm/OpenSSL are FIPS-certified, whereas the
# earlier Debian base tripped libgcrypt's MD5-in-FIPS abort during `apt`
# (_gcry_fatal_error "requested algo not in md context", SIGABRT/134). UBI
# images pull freely (no subscription) and build anywhere, so this is also fine
# on non-RHEL Docker hosts. Build CONTEXT is the repo root (the backend installs
# the local gliner2 package [editable ../..] and reads scripts/).
#
#   # build (from the repo root):
#   docker build -f viewer/Dockerfile -t gliner2-viewer .
#
#   # run (data/ + out/ are mounted, not baked in; --gpus all for GPU). Only
#   # :3000 needs publishing -- the browser reaches the backend through the
#   # /api proxy; :8000 is loopback-only for local debugging:
#   docker run --gpus all --rm -p 3000:3000 -p 127.0.0.1:8000:8000 \
#     -v "$PWD/data:/app/data:ro" -v "$PWD/out:/app/out:ro" \
#     -v gliner2-hf-cache:/root/.cache/huggingface \
#     gliner2-viewer
#   # drop --gpus all for CPU -- GLiNER2 picks the device on load.
#
# Remote host? Nothing to rebuild: the browser calls /api/* on the page's own
# origin and Next proxies to the co-located backend, so accessing the viewer at
# http://SERVER:3000 works out of the box (no baked backend host, no CORS).

FROM registry.access.redhat.com/ubi9/python-312

USER 0
WORKDIR /app

# gcc-c++/make + python3.12-devel (Python.h) compile the fasttext C++ ext
# (gliner2 -> lumi-language-id-2 -> fasttext-numpy2, sdist only); nodejs from
# UBI9 appstream is >= 18.20, which satisfies Next 15's Node >= 18.18; git for
# any VCS-sourced deps.
RUN dnf install -y --setopt=install_weak_deps=False \
        gcc-c++ make git nodejs npm python3.12-devel \
    && dnf clean all

# uv (copied from its published image -- no build-time network crypto here).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

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

# --- frontend build (Next production build) ---
COPY viewer/frontend/package.json viewer/frontend/package-lock.json ./viewer/frontend/
RUN --mount=type=cache,target=/root/.npm \
    cd viewer/frontend && npm ci
COPY viewer/frontend/ ./viewer/frontend/
# No backend host is baked into the CLIENT: it calls /api/* same-origin and Next
# proxies to the co-located backend (next.config.mjs), so the viewer works
# however it is reached. Leave NEXT_PUBLIC_API_BASE empty; set it to an ABSOLUTE
# URL only to bypass the proxy and point the client straight at an external
# backend. GLINER2_BACKEND_ORIGIN retargets the proxy destination and is baked
# into the build here (next start reads the built manifest, not a runtime env),
# so a split-container deploy passes --build-arg GLINER2_BACKEND_ORIGIN=...;
# empty -> the default http://127.0.0.1:8000 (co-located backend).
ARG NEXT_PUBLIC_API_BASE=
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
ARG GLINER2_BACKEND_ORIGIN=
ENV GLINER2_BACKEND_ORIGIN=$GLINER2_BACKEND_ORIGIN
RUN cd viewer/frontend && npm run build

COPY viewer/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Persist HF downloads across restarts by mounting this path.
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 3000 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/health || exit 1
# Clear the UBI s2i entrypoint so the container runs start.sh directly.
ENTRYPOINT []
CMD ["/app/start.sh"]
