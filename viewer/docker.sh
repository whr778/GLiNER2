#!/usr/bin/env bash
#
# Start / stop the GLiNER2 viewer as a Docker container -- backend (:8000) +
# frontend (:3000) in one image. Mirrors viewer/viewer.sh, but drives the
# container instead of host processes.
#
# Usage (from anywhere):
#   bash viewer/docker.sh start      # run the container, wait until it's up
#   bash viewer/docker.sh stop       # stop + remove the container
#   bash viewer/docker.sh restart
#   bash viewer/docker.sh status
#   bash viewer/docker.sh logs       # follow the container logs (Ctrl-C to quit)
#
# data/ and out/ (models) are mounted read-only -- with an SELinux :z relabel
# when SELinux is enforcing (e.g. EL9), else the daemon rejects the bind mount.
# HF downloads persist in the gliner2-hf-cache volume. GPU is auto-detected
# (adds --gpus all when nvidia-smi
# is present) -- force with GPU=1, disable with GPU=0. Pick the model with
# GLINER2_MODEL=<hf-id-or-/app/out/...>. Override the image/name with IMAGE=/NAME=.
#
# Build the image first:  docker build -f viewer/Dockerfile -t gliner2-viewer .
#                         (or: bash viewer/docker-build.sh)

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-gliner2-viewer}"
NAME="${NAME:-gliner2-viewer}"
# Host paths bind-mounted to /app/data and /app/out (default: the repo's).
# Override when the Docker engine can't reach the repo -- a common EL9 failure is
#   "error creating source path ... mkdir ...: permission denied"
# on a path that already exists, even though your own perms look fine. Causes:
#   - NFS home with root_squash: a rootful daemon runs as root -> squashed to
#     nobody, so it cannot touch your home.
#   - SELinux-restricted /home (the :z relabel below handles the local-fs case).
#   - Rootless Docker: its user namespace maps your PRIMARY uid/gid to root-in-ns
#     but NOT your supplementary groups, and can't map NFS homes -- so a shared,
#     group-owned source (or anything on NFS) is unreachable. `namei -l "$PWD/out"`
#     shows which component/group is the problem; `docker info` says if rootless.
# Fix (no sudo, no daemon restart): point these at a LOCAL directory you OWN
# outright (not a shared-group dir, not NFS home):
#   mkdir -p /tmp/gliner2/{data,out}
#   DATA_DIR=/tmp/gliner2/data OUT_DIR=/tmp/gliner2/out bash viewer/docker.sh start
DATA_DIR="${DATA_DIR:-$REPO/data}"
OUT_DIR="${OUT_DIR:-$REPO/out}"
BACK_PORT=8000
FRONT_PORT=3000
BACK_HEALTH="http://127.0.0.1:$BACK_PORT/health"
FRONT_URL="http://localhost:$FRONT_PORT"

is_up()   { curl -sf --max-time 2 "$1" >/dev/null 2>&1; }
running() { [ -n "$(docker ps -q -f "name=^${NAME}$" 2>/dev/null)" ]; }
exists()  { [ -n "$(docker ps -aq -f "name=^${NAME}$" 2>/dev/null)" ]; }

start() {
  if running; then
    echo "Container '$NAME' is already running. Run 'stop' first."; return 1
  fi
  exists && docker rm "$NAME" >/dev/null 2>&1  # clear a previous, stopped run
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image '$IMAGE' not found. Build it first:" >&2
    echo "  docker build -f viewer/Dockerfile -t $IMAGE .   (or bash viewer/docker-build.sh)" >&2
    return 1
  fi
  mkdir -p "$DATA_DIR" "$OUT_DIR" 2>/dev/null || true

  # GPU: auto (nvidia-smi present) unless forced with GPU=1 / disabled with GPU=0.
  local gpu=()
  case "${GPU:-auto}" in
    1) gpu=(--gpus all) ;;
    0) gpu=() ;;
    *) command -v nvidia-smi >/dev/null 2>&1 && gpu=(--gpus all) ;;
  esac

  local mode="CPU"; [ ${#gpu[@]} -gt 0 ] && mode="GPU"

  # SELinux (EL9 defaults to Enforcing) denies the container runtime access to
  # bind-mount sources unless they're relabeled -- ":z" tells Docker to relabel
  # them to container_file_t. Without it you get a daemon-side "error creating
  # mount source path ... mkdir ... permission denied" (DAC perms look fine
  # because it is SELinux/MAC denying). Skipped when SELinux isn't enforcing.
  local vsuf="ro"
  [ "$(getenforce 2>/dev/null)" = "Enforcing" ] && vsuf="ro,z"

  local args=(run -d --name "$NAME")
  [ ${#gpu[@]} -gt 0 ] && args+=("${gpu[@]}")
  args+=(-p "$FRONT_PORT:3000" -p "$BACK_PORT:8000"
         -v "$DATA_DIR:/app/data:$vsuf" -v "$OUT_DIR:/app/out:$vsuf"
         -v gliner2-hf-cache:/root/.cache/huggingface)
  [ -n "${GLINER2_MODEL:-}" ] && args+=(-e "GLINER2_MODEL=$GLINER2_MODEL")
  args+=("$IMAGE")

  echo "Starting '$NAME' from '$IMAGE' ($mode)  model=${GLINER2_MODEL:-<default>}"
  docker "${args[@]}" >/dev/null || { echo "docker run failed"; return 1; }

  printf "Waiting for backend"
  for _ in $(seq 1 60); do
    is_up "$BACK_HEALTH" && break
    running || { echo; echo "  container exited early -- see 'logs'"; return 1; }
    printf "."; sleep 1
  done; echo
  printf "Waiting for frontend"
  for _ in $(seq 1 90); do is_up "$FRONT_URL" && break; printf "."; sleep 1; done; echo

  is_up "$BACK_HEALTH" && echo "  backend  OK  -> http://127.0.0.1:$BACK_PORT" \
                       || echo "  backend  FAILED (see 'logs')"
  is_up "$FRONT_URL"   && echo "  frontend OK  -> $FRONT_URL" \
                       || echo "  frontend FAILED (see 'logs')"
  echo "Open $FRONT_URL"
}

stop() {
  echo "Stopping container '$NAME'..."
  docker stop "$NAME" >/dev/null 2>&1 || true
  docker rm "$NAME"   >/dev/null 2>&1 || true
  status
}

status() {
  running && docker ps -f "name=^${NAME}$" --format "container {{.Names}}: {{.Status}}" \
          || echo "container $NAME: not running"
  is_up "$BACK_HEALTH" && echo "backend  ($BACK_HEALTH): UP"   || echo "backend  ($BACK_HEALTH): down"
  is_up "$FRONT_URL"   && echo "frontend ($FRONT_URL): UP" || echo "frontend ($FRONT_URL): down"
}

logs() { docker logs -f "$NAME"; }

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  logs)    logs ;;
  *) echo "usage: bash viewer/docker.sh {start|stop|restart|status|logs}"; exit 1 ;;
esac
