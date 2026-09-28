#!/bin/bash
# Run the test suite on a real CUDA box, then terminate it. No training, no HF publish.
#
# WHY THIS IS NOT provision_box.sh: that script is built for training jobs and demands a
# write-capable $HF_TOKEN plus a config and checkpoint, because its jobs publish models.
# A test run publishes nothing, so requiring a token would make the strongest credential
# on the laptop a prerequisite for the cheapest job we run.
#
# WHY A CUDA RUN AT ALL: `_resolve_device` picks cuda -> mps -> cpu, so on a laptop the
# suite NEVER takes the cuda branch. Two known device-specific facts make that gap real --
# torch.gather on int64 corrupts above 2^24 on MPS but is exact on CUDA, and the DDP test's
# skip guard cannot fire on Darwin because gloo is available there. Neither is visible
# locally, in either direction.
#
# THREE INDEPENDENT STOPS, because idle time is billed:
#   1. `timeout` on the pytest run itself
#   2. a detached hard-deadline watchdog on the box
#   3. terminate on the normal path, in a trap that runs on every exit
#
#   BRANCH=merge/upstream-20260928 bash tools/lambda/run_tests_cuda.sh
set -uo pipefail
NAME=${NAME:-cuda-tests}
BRANCH=${BRANCH:?branch required}
TYPES=${TYPES:-"gpu_1x_a10 gpu_1x_a100_sxm4"}
SSH_KEY_NAME=${SSH_KEY_NAME:-gliner2-mac}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
REPO=${REPO:-https://github.com/whr778/GLiNER2.git}
TEST_TIMEOUT=${TEST_TIMEOUT:-3600}
HARD_DEADLINE=${HARD_DEADLINE:-5400}
: "${LAMBDA_API_KEY:?not set}"

API=https://cloud.lambda.ai/api/v1
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
ID=""

# The trap is the stop that matters: it fires on success, on failure, and on Ctrl-C.
terminate() {
  [ -n "$ID" ] || return 0
  echo "[cuda] terminating $ID"
  curl -s -u "$LAMBDA_API_KEY:" -H 'Content-Type: application/json' \
    -d "{\"instance_ids\":[\"$ID\"]}" "$API/instance-operations/terminate" >/dev/null
}
trap terminate EXIT INT TERM

for t in $TYPES; do
  REGION=$(curl -s -u "$LAMBDA_API_KEY:" "$API/instance-types" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data'].get('$t') or {}
caps = [c['name'] for c in (d.get('regions_with_capacity_available') or [])]
print(next((c for c in caps if c.startswith('us-')), caps[0] if caps else ''))")
  [ -n "$REGION" ] || { echo "[cuda] no capacity for $t"; continue; }
  ID=$(curl -s -u "$LAMBDA_API_KEY:" -H 'Content-Type: application/json' \
    -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$t\",\"ssh_key_names\":[\"$SSH_KEY_NAME\"],\"name\":\"$NAME\"}" \
    "$API/instance-operations/launch" | python3 -c "
import sys, json; d = json.load(sys.stdin)
print((d.get('data') or {}).get('instance_ids', [''])[0])")
  [ -n "$ID" ] && { echo "[cuda] launched $t in $REGION as $ID"; break; }
done
[ -n "$ID" ] || { echo "[cuda] *** could not launch any of: $TYPES ***"; exit 1; }

echo "[cuda] waiting for the box to come up"
for _ in $(seq 1 60); do
  IP=$(curl -s -u "$LAMBDA_API_KEY:" "$API/instances/$ID" | python3 -c "
import sys, json
try: print((json.load(sys.stdin).get('data') or {}).get('ip') or '')
except Exception: print('')")
  [ -n "$IP" ] && $SSH "ubuntu@$IP" true 2>/dev/null && break
  sleep 20
done
[ -n "${IP:-}" ] || { echo "[cuda] *** box never got an IP ***"; exit 1; }
echo "[cuda] box is up at $IP"

# ECC ON A FRESH BOX HAS COME UP PENDING BEFORE, which makes every number suspect.
$SSH "ubuntu@$IP" 'nvidia-smi --query-gpu=name,ecc.mode.current --format=csv,noheader'

# Stop 2: a DETACHED watchdog, so a hung run cannot outlive the deadline even if this
# laptop loses the network. setsid keeps it alive when the ssh session dies.
$SSH "ubuntu@$IP" "setsid nohup bash -c 'sleep $HARD_DEADLINE; sudo poweroff' \
  >/dev/null 2>&1 < /dev/null &" || true

$SSH "ubuntu@$IP" "bash -s" <<REMOTE
set -uo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="\$HOME/.local/bin:\$PATH"
git clone --depth 1 --branch "$BRANCH" "$REPO" repo 2>&1 | tail -2
cd repo
# PYTHON IS PINNED, like bootstrap_box.sh. Left to itself uv picked 3.14 here, which is not
# the interpreter any GPU run of this project has ever used.
uv venv --python 3.12 2>&1 | tail -2
uv sync --group dev 2>&1 | tail -3

# THE cu128 TRAP, and it is the reason this block exists rather than a plain uv sync.
# Lambda boxes run a 12.8 driver; uv resolves a torch wheel built for CUDA 13, and the
# first torch.cuda call dies with "The NVIDIA driver on your system is too old (found
# version 12080)". Measured on an A10 on 2026-09-28: the whole suite ran with CUDA
# unavailable, so every cuda-gated test SKIPPED and the run looked like a pass.
#
# `--reinstall-package torch` IS THE LOAD-BEARING FLAG. Both wheels call themselves
# 2.11.0 -- the CUDA 13 one is just +cu130 -- so a plain pin is already satisfied, uv
# changes nothing, and torch.cuda.is_available() stays False. Same trap documented in
# bootstrap_box.sh; this is a deliberate duplicate of those three lines, and the two must
# be changed together.
uv pip install --reinstall-package torch "torch==2.11.0" \
  --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2

# PROVE IT, AND STOP IF IT DID NOT TAKE. The A10 run continued past a failed swap and
# produced a 2,814-passed report that said nothing whatever about CUDA. A test run whose
# entire purpose is the cuda branch must refuse to score anything without it.
./.venv/bin/python - <<'PROOF' || { echo "[cuda] *** REFUSING: CUDA unavailable after the cu128 swap; a run now would score the CPU path and report it as CUDA ***"; exit 1; }
import sys, torch
print(f"[cuda] torch {torch.__version__} | available {torch.cuda.is_available()} | built for {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit(1)
print("[cuda] device:", torch.cuda.get_device_name(0))
PROOF

# Stop 1: the timeout on the run itself.
timeout $TEST_TIMEOUT ./.venv/bin/python -m pytest tests/ -q -rf --deselect tests/training/test_matching.py 2>&1 | tail -40
echo "[cuda] pytest exit: \${PIPESTATUS[0]}"
REMOTE

echo "[cuda] done; the trap terminates the box on the way out"
