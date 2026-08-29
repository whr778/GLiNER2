#!/bin/bash
# Self-contained Lambda job: score, publish, TERMINATE. Designed so that losing the
# operator's network cannot leave an instance billing.
#
# THREE INDEPENDENT STOPS, because any one of them can fail:
#   1. `timeout` on the job itself, so the log survives a hang.
#   2. A detached hard-deadline watchdog that terminates regardless of the job.
#   3. Terminate at the end of the normal path.
# The instance terminates ITSELF via the Lambda API using a key written to ~/.lambda_key
# (mode 600) at launch, so no laptop involvement is required at any point.
#
# TRAP, previously paid for: `pkill -f box_run` also matches the incoming ssh command
# string and kills the caller's own session mid-command. Collect pids with
# `pgrep -f "[b]ox_run"` and kill by pid.
set -uo pipefail

POOL=${POOL:-$HOME/pool_prefiltered.jsonl}
OUT=${OUT:-$HOME/pool_scores.jsonl}
REPO=${REPO:-whr778/turkish-pool-gate-scores}
JOB_TIMEOUT=${JOB_TIMEOUT:-14400}      # 4h on the job
HARD_DEADLINE=${HARD_DEADLINE:-18000}  # 5h absolute, whatever happens

terminate() {
  local id="${INSTANCE_ID:-}"
  [ -z "$id" ] && id=$(cat ~/.instance_id 2>/dev/null)
  echo "[box] terminating $id"
  curl -s -u "$(cat ~/.lambda_key):" -X POST \
    https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$id\"]}"
}

# PRE-FLIGHT: prove we can kill ourselves BEFORE starting anything expensive. An
# unkillable box bills until someone notices, which is the failure this whole script
# exists to prevent -- so refuse to start rather than run un-terminable.
INSTANCE_ID=$(curl -s --max-time 5 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
[ -z "$INSTANCE_ID" ] && INSTANCE_ID=$(cat ~/.instance_id 2>/dev/null)
if [ -z "$INSTANCE_ID" ]; then
  echo "[box] FATAL: cannot determine instance id; refusing to start" >&2; exit 3
fi
if ! curl -sf -u "$(cat ~/.lambda_key):" https://cloud.lambda.ai/api/v1/instances \
     | grep -q "$INSTANCE_ID"; then
  echo "[box] FATAL: Lambda API cannot see instance $INSTANCE_ID; refusing to start" >&2
  exit 3
fi
echo "[box] pre-flight OK: can terminate $INSTANCE_ID"

# STOP 2: hard deadline, detached, survives the job dying in any manner.
( sleep "$HARD_DEADLINE"; echo "[box] HARD DEADLINE"; terminate ) >> ~/watchdog.log 2>&1 &
disown

# The publish step is the only thing standing between a finished job and a destroyed
# disk, so fail loudly here rather than after 45 minutes of GPU time.
if [ ! -s ~/.hf_token ]; then
  echo "[box] FATAL: no ~/.hf_token; results could not be published" >&2
  terminate; exit 4
fi
export HF_TOKEN=$(cat ~/.hf_token)
export PATH="$HOME/.local/bin:$PATH"

cd ~/gliner2
echo "[box] starting $(date -u)"
# STOP 1: bounded job. rc=124 means it hit the cap; the log and checkpoints still exist.
timeout "$JOB_TIMEOUT" ./.venv/bin/python -u tools/lambda/score_pool.py \
  --pool "$POOL" --out "$OUT" --repo "$REPO" 2>&1 | tee ~/score.log
rc=${PIPESTATUS[0]}
echo "[box] job rc=$rc"

# Publish whatever exists even on failure -- partial scores beat none, and the disk dies
# with the instance.
if [ "$rc" -ne 0 ] && [ -s "$OUT" ]; then
  echo "[box] job failed; publishing partial results"
  ./.venv/bin/python - <<'PY' || echo "[box] partial publish FAILED"
import os
from huggingface_hub import HfApi
api = HfApi()
repo = os.environ.get("REPO", "whr778/turkish-pool-gate-scores")
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for f in ("pool_scores.jsonl", "score.log"):
    p = os.path.expanduser(f"~/{f}")
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"partial/{f}",
                        repo_id=repo, repo_type="dataset")
        print("uploaded", f)
PY
fi

# STOP 3: normal path.
echo "[box] finished $(date -u); terminating"
terminate
