#!/bin/bash
# Take a LAUNCHED Lambda instance from bare to running a job, and refuse to proceed past
# any step that did not actually work.
#
#   INSTANCE_ID=<32 hex> JOB="bash tools/lambda/pool_ab.sh" \
#     CFG=tools/train/config/ab/pool-shared.yaml CKPT=whr778/gliner2-... \
#     bash tools/lambda/provision.sh
#
# WRITTEN AFTER A PROVISION THAT REPORTED SUCCESS AND HAD SHIPPED NO CREDENTIALS. On
# 2026-09-17 an A10 was launched, bootstrapped, and told to start; the run died at the data
# restore with `httpx.LocalProtocolError: Illegal header value b'Bearer '`. The cause was one
# line: the token was read from `~/.hf_token`, which does not exist on this laptop, so an
# EMPTY string was written to the box. The remote check noticed -- it printed `no hf token`
# and exited 1 -- but the local script ignored the exit status and carried on to bootstrap,
# to the job, and to a final line reading `DONE -- box is running`. A check whose failure
# does not stop anything is not a check, which is the same defect this project has now fixed
# in a training gate, a sweep's threshold capture, and a publish helper.
#
# THE TOKEN TRAP, measured. Three sources on one laptop, and the one huggingface_hub reaches
# for BY DEFAULT is the read-only one:
#     env HF_TOKEN                 write OK
#     ~/.cache/huggingface/token   403 "you must use a write token"
#     ~/.huggingface/token         write OK
# So the token is taken from the ENVIRONMENT and proven to WRITE before the box is touched.
# Publishing is the only thing standing between a finished run and a destroyed disk.
set -euo pipefail

ID=${INSTANCE_ID:?set INSTANCE_ID}
JOB=${JOB:?set JOB}
CFG=${CFG:-}
CKPT=${CKPT:-}
BRANCH=${BRANCH:-merge/main-20260805}
JOB_TIMEOUT=${JOB_TIMEOUT:-14400}
HARD_DEADLINE=${HARD_DEADLINE:-18000}
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"

fail() { echo "[prov] *** FATAL: $* ***" >&2; exit 1; }

# --- Prove the credentials LOCALLY, before an instance is touched --------------------------
[ -n "${LAMBDA_API_KEY:-}" ] || fail "LAMBDA_API_KEY is not set"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is not set in the environment"
# `uv run python`, not `python3`: the system interpreter has no huggingface_hub, so the probe
# fails to IMPORT and the guard reports "cannot write" for a token that writes fine. A gate
# that cannot tell a missing library from a missing permission fails safe but reads wrong.
(cd "$(dirname "$0")/../.." && uv run python -) <<'PROBE' || fail "HF_TOKEN cannot WRITE to the Hub -- a run would train and then fail to publish"
import io, os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"].strip())
repo = "whr778/gliner2-run-logs"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
# An upload, because create_repo(exist_ok=True) on an existing repo is a no-op that
# returns cleanly for a read-only token.
api.upload_file(path_or_fileobj=io.BytesIO(b"provision write probe\n"),
                path_in_repo="_write_probe.txt", repo_id=repo, repo_type="dataset")
api.delete_file("_write_probe.txt", repo, repo_type="dataset")
print(f"[prov] HF_TOKEN writes OK as {api.whoami().get('name')}")
PROBE

# --- Wait for the instance ---------------------------------------------------------------
IP=""
for i in $(seq 1 90); do
  IP=$(curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances | python3 -c "
import sys, json
for x in json.load(sys.stdin)['data']:
    if x['id'] == '$ID' and x['status'] == 'active' and x.get('ip'):
        print(x['ip'])")
  [ -n "$IP" ] && break
  sleep 20
done
[ -n "$IP" ] || fail "instance $ID never became active"
echo "[prov] active at $IP"
for i in $(seq 1 40); do $SSH ubuntu@$IP true 2>/dev/null && break; sleep 15; done
$SSH ubuntu@$IP true 2>/dev/null || fail "ssh never came up at $IP"

# --- ECC pre-flight: a faulty A10 cost a relaunch on 2026-09-16 ---------------------------
ECC=$($SSH ubuntu@$IP 'nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader' 2>/dev/null | tr -d ' ')
echo "[prov] ECC uncorrected: ${ECC:-unknown}"
case "$ECC" in
  0|"[N/A]"|"") : ;;
  *) curl -s -u "$LAMBDA_API_KEY:" -X POST \
       https://cloud.lambda.ai/api/v1/instance-operations/terminate \
       -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$ID\"]}" >/dev/null
     fail "$ECC ECC errors -- terminated rather than train on it";;
esac

# --- Credentials, written as files, and VERIFIED ON THE BOX -------------------------------
# The login greeting on this account goes to STDOUT and has corrupted a .lambda_key before,
# billing two boxes idle. Every secret is printf'd into a file, never echoed through a pipe,
# and every one is checked for shape afterwards. `set -e` here plus no `||` at the call site
# means a failure stops the provision instead of decorating it.
$SSH ubuntu@$IP "bash -s" <<CREDS || fail "credentials did not land on the box"
set -euo pipefail
printf '%s' '$LAMBDA_API_KEY' > ~/.lambda_key && chmod 600 ~/.lambda_key
printf '%s' '$ID'             > ~/.instance_id
printf '%s' '$HF_TOKEN'       > ~/.hf_token   && chmod 600 ~/.hf_token
grep -qE '^[0-9a-f]{32}\$' ~/.instance_id || { echo "instance id corrupt"; exit 1; }
grep -qE '^hf_' ~/.hf_token               || { echo "hf token missing or malformed"; exit 1; }
[ -s ~/.lambda_key ]                      || { echo "lambda key empty"; exit 1; }
echo "[prov] credentials landed clean"
CREDS

$SSH ubuntu@$IP "bash -s" <<SETUP || fail "repo checkout failed"
set -euo pipefail
cd ~ && rm -rf gliner2
git clone -q https://github.com/whr778/GLiNER2.git gliner2
cd gliner2 && git checkout -q $BRANCH
echo "[prov] repo at \$(git log --oneline -1)"
SETUP

echo "[prov] bootstrap $(date -u)"
$SSH ubuntu@$IP "bash -lc 'cd ~/gliner2 && CFG=$CFG CKPT=$CKPT bash tools/lambda/bootstrap_box.sh'" \
  2>&1 | grep -v "^Restored session:" | tail -30
# PIPESTATUS, not $?, or the pipeline's tail masks the bootstrap's own failure -- which is
# precisely how a FATAL data restore was followed by "box is running".
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "bootstrap failed -- not starting the job"

echo "[prov] starting job $(date -u)"
$SSH ubuntu@$IP "bash -lc 'cd ~/gliner2 && JOB=\"$JOB\" JOB_TIMEOUT=$JOB_TIMEOUT HARD_DEADLINE=$HARD_DEADLINE nohup bash tools/lambda/box_run.sh > ~/box.log 2>&1 < /dev/null &'" \
  || fail "could not start the job"
echo "[prov] job started on $IP -- box_run.sh owns termination from here"
