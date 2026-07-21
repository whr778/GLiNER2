#!/usr/bin/env bash
#
# Build the GLiNER2 viewer image for the EL9 / Intel-64 GPU box (linux/amd64).
# The build context is the repo root, so this can be run from anywhere.
#
#   viewer/docker-build.sh                       # linux/amd64 (the EL9 box)
#   PLATFORM=linux/arm64 viewer/docker-build.sh  # build for this arch instead
#   API_BASE=http://myserver:8000 viewer/docker-build.sh   # remote frontend URL
#   TAG=gliner2-viewer:el9 viewer/docker-build.sh
#
# Building on the EL9 box itself is native + fast. Cross-building amd64 from an
# Apple-Silicon Mac works via emulation (Docker Desktop / buildx) but is slow;
# afterwards ship it with:  docker save "$TAG" | ssh el9box docker load
#
# GPU is a run-time flag (`docker run --gpus all ...`), not a build option: the
# image always ships CUDA torch and GLiNER2 falls back to CPU without a GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PLATFORM="${PLATFORM:-linux/amd64}"
TAG="${TAG:-gliner2-viewer}"
API_BASE="${API_BASE:-http://127.0.0.1:8000}"

echo "Building $TAG for $PLATFORM  (NEXT_PUBLIC_API_BASE=$API_BASE)"
docker build \
  --platform "$PLATFORM" \
  --build-arg NEXT_PUBLIC_API_BASE="$API_BASE" \
  -f viewer/Dockerfile \
  -t "$TAG" \
  .

echo
echo "Done. Run it (add --gpus all on the GPU box):"
echo "  docker run --gpus all --rm -p 3000:3000 -p 8000:8000 \\"
echo "    -v \"\$PWD/data:/app/data:ro\" -v \"\$PWD/out:/app/out:ro\" \\"
echo "    -v gliner2-hf-cache:/root/.cache/huggingface \\"
echo "    $TAG"
