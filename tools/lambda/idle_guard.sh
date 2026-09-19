#!/usr/bin/env bash
# STOP 4: the box is up but NO JOB EVER STARTED.
#
# Stops 1-3 (job timeout, hard-deadline watchdog, terminate on the normal path) all live
# INSIDE box_run.sh, so not one of them is armed until the job starts. That leaves a real
# window: if the laptop's link drops during the ssh that starts the job, the box sits idle
# with no watchdog at all -- and the laptop cannot tell "no runner" from "cannot reach the
# box". Both absneg arms hit exactly that on 2026-09-19.
#
# Armed at provision time, once ~/.lambda_key and ~/.instance_id are verified and before
# the bootstrap. After IDLE_GRACE it asks ONE question: is a runner alive? If yes it stands
# down and never touches the run. If no, it terminates the instance itself. It needs
# nothing from the laptop, which is the entire point.
set -uo pipefail

IDLE_GRACE=${IDLE_GRACE:-1200}   # 20 min: bootstrap measures ~5 min, job start seconds

# PATH-ANCHORED on purpose. A bare `box_run.sh` pattern matches any process whose command
# line merely CONTAINS the name -- it matched the test harness and the laptop's own ssh
# clients, reporting a runner alive on a box that had none. That is the one answer this
# guard must never give wrongly. Overridable so the test can inject a unique token and be
# immune to whatever else is running on the host.
RUNNER_PATTERN=${RUNNER_PATTERN:-'tools/lambda/box_run\.sh'}

sleep "$IDLE_GRACE"

if pgrep -f "$RUNNER_PATTERN" >/dev/null; then
  echo "[guard] runner alive after ${IDLE_GRACE}s -- standing down"
  exit 0
fi

ID=$(cat ~/.instance_id 2>/dev/null)
if ! printf '%s' "$ID" | grep -qE '^[0-9a-f]{32}$'; then
  echo "[guard] FATAL: no valid instance id; cannot terminate" >&2
  exit 1
fi

echo "[guard] no runner after ${IDLE_GRACE}s -- terminating $ID"
curl -s -u "$(cat ~/.lambda_key):" -X POST \
  https://cloud.lambda.ai/api/v1/instance-operations/terminate \
  -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$ID\"]}"
