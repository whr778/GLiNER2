#!/usr/bin/env bash
#
# Container entrypoint: run the GLiNER2 viewer backend (uvicorn :8000) and
# frontend (next :3000) side by side. If either process exits, tear the whole
# container down so the failure is visible (rather than a half-dead viewer).

set -uo pipefail

( cd /app/viewer/backend && exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 ) &
backend=$!

( cd /app/viewer/frontend && exec node_modules/.bin/next start -H 0.0.0.0 -p 3000 ) &
frontend=$!

shutdown() { kill "$backend" "$frontend" 2>/dev/null; }
trap shutdown TERM INT

# Wake as soon as either child exits, then stop the other.
wait -n
shutdown
wait
