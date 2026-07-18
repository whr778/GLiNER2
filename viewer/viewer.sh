#!/usr/bin/env bash
#
# Start / stop the GLiNER2 viewer -- FastAPI backend (:8000) + NextJS frontend
# (:3000) -- with one command.
#
# Usage (from anywhere):
#   bash viewer/viewer.sh start      # start both, wait until they're up
#   bash viewer/viewer.sh stop       # stop both
#   bash viewer/viewer.sh restart
#   bash viewer/viewer.sh status
#   bash viewer/viewer.sh logs       # tail both logs (Ctrl-C to quit)
#
# Choose the model: export GLINER2_MODEL=<hf-id-or-path> before `start`.
# Requires: uv, npm (and `npm install` already run once in viewer/frontend).

set -uo pipefail

VIEWER_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN="$VIEWER_DIR/.run"
mkdir -p "$RUN"

BACK_PORT=8000
FRONT_PORT=3000
BACK_HEALTH="http://127.0.0.1:$BACK_PORT/health"
FRONT_URL="http://localhost:$FRONT_PORT"

is_up() { curl -sf --max-time 2 "$1" >/dev/null 2>&1; }

start() {
  if is_up "$BACK_HEALTH" || is_up "$FRONT_URL"; then
    echo "Viewer already running (or ports $BACK_PORT/$FRONT_PORT in use). Run 'stop' first."
    return 1
  fi

  echo "Starting backend  (uvicorn :$BACK_PORT)  model=${GLINER2_MODEL:-<default>}"
  nohup bash -c 'cd "$1" && exec uv run uvicorn app:app --host 127.0.0.1 --port "$2"' \
    _ "$VIEWER_DIR/backend" "$BACK_PORT" >"$RUN/backend.log" 2>&1 &
  echo $! >"$RUN/backend.pid"

  echo "Starting frontend (next dev :$FRONT_PORT)"
  nohup bash -c 'cd "$1" && exec npm run dev' \
    _ "$VIEWER_DIR/frontend" >"$RUN/frontend.log" 2>&1 &
  echo $! >"$RUN/frontend.pid"

  printf "Waiting for backend"
  for _ in $(seq 1 40); do is_up "$BACK_HEALTH" && break; printf "."; sleep 1; done; echo
  printf "Waiting for frontend"
  for _ in $(seq 1 90); do is_up "$FRONT_URL" && break; printf "."; sleep 1; done; echo

  is_up "$BACK_HEALTH" && echo "  backend  OK  -> http://127.0.0.1:$BACK_PORT" \
                       || echo "  backend  FAILED (see $RUN/backend.log)"
  is_up "$FRONT_URL"   && echo "  frontend OK  -> $FRONT_URL" \
                       || echo "  frontend FAILED (see $RUN/frontend.log)"
  echo "Open $FRONT_URL"
}

kill_pidfile() {
  local pidfile="$1"
  [ -f "$pidfile" ] || return 0
  local pid; pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    pkill -P "$pid" 2>/dev/null || true   # children (uvicorn / next worker)
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
}

stop() {
  echo "Stopping viewer..."
  kill_pidfile "$RUN/frontend.pid"
  kill_pidfile "$RUN/backend.pid"
  # Fallbacks for detached children the pidfile parent may have spawned.
  pkill -f "next dev" 2>/dev/null || true
  pkill -f "next-server" 2>/dev/null || true
  pkill -f "uvicorn app:app" 2>/dev/null || true
  sleep 1
  status
}

status() {
  is_up "$BACK_HEALTH" && echo "backend  ($BACK_HEALTH): UP"   || echo "backend  ($BACK_HEALTH): down"
  is_up "$FRONT_URL"   && echo "frontend ($FRONT_URL): UP" || echo "frontend ($FRONT_URL): down"
}

logs() { tail -n 30 -F "$RUN/backend.log" "$RUN/frontend.log"; }

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  logs)    logs ;;
  *) echo "usage: bash viewer/viewer.sh {start|stop|restart|status|logs}"; exit 1 ;;
esac
