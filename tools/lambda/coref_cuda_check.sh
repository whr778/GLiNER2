#!/bin/bash
# Entity-coref link on CUDA: the full val sweep (tools/train/coref_threshold.py --device cuda) with GPU and
# process memory logged every 20 s, then the per-doc link logits copied back to compare with the CPU run.
# Questions: does memory stay flat on CUDA over 2,852 docs (MPS grows per input shape and was SIGKILLed),
# and do CUDA logits pick the same cut-off as CPU (+0.20)? Nothing is published; results come back by scp
# to $OUT. Setup (uv, Python 3.12, the cu128 torch swap + its proof) is run_tests_cuda.sh's, deliberately
# duplicated -- change the two together.
#
# THREE INDEPENDENT STOPS: `timeout` on the sweep, a detached hard-deadline watchdog, terminate in a trap.
#
#   bash tools/lambda/coref_cuda_check.sh
set -uo pipefail
NAME=${NAME:-coref-cuda-check}
BRANCH=$(git rev-parse --abbrev-ref HEAD)
EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
TYPES=${TYPES:-"gpu_1x_a10 gpu_1x_a100_sxm4"}
SSH_KEY_NAME=${SSH_KEY_NAME:-gliner2-mac}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
REPO=${REPO:-https://github.com/whr778/GLiNER2.git}
MODEL=${MODEL:-whr778/gliner2-corefC-treatment}
JOB_TIMEOUT=${JOB_TIMEOUT:-2400}
HARD_DEADLINE=${HARD_DEADLINE:-3600}
OUT=${OUT:-/Volumes/Development/tmp/coref_cuda_check}
: "${LAMBDA_API_KEY:?not set}"
: "${HF_TOKEN:?not set -- the model and val files are private}"

git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[coref-cuda] *** REFUSING -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }

API=https://cloud.lambda.ai/api/v1
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
SCP="scp -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
ID=""

terminate() {
  [ -n "$ID" ] || return 0
  echo "[coref-cuda] terminating $ID"
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
  [ -n "$REGION" ] || { echo "[coref-cuda] no capacity for $t"; continue; }
  ID=$(curl -s -u "$LAMBDA_API_KEY:" -H 'Content-Type: application/json' \
    -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$t\",\"ssh_key_names\":[\"$SSH_KEY_NAME\"],\"name\":\"$NAME\"}" \
    "$API/instance-operations/launch" | python3 -c "
import sys, json; d = json.load(sys.stdin)
print((d.get('data') or {}).get('instance_ids', [''])[0])")
  [ -n "$ID" ] && { echo "[coref-cuda] launched $t in $REGION as $ID"; break; }
done
[ -n "$ID" ] || { echo "[coref-cuda] *** could not launch any of: $TYPES ***"; exit 1; }

echo "[coref-cuda] waiting for the box to come up"
for _ in $(seq 1 60); do
  IP=$(curl -s -u "$LAMBDA_API_KEY:" "$API/instances/$ID" | python3 -c "
import sys, json
try: print((json.load(sys.stdin).get('data') or {}).get('ip') or '')
except Exception: print('')")
  [ -n "$IP" ] && $SSH "ubuntu@$IP" true 2>/dev/null && break
  sleep 20
done
[ -n "${IP:-}" ] || { echo "[coref-cuda] *** box never got an IP ***"; exit 1; }
echo "[coref-cuda] box is up at $IP"

$SSH "ubuntu@$IP" 'nvidia-smi --query-gpu=name,ecc.mode.current,ecc.errors.uncorrected.volatile.total --format=csv,noheader'
$SSH "ubuntu@$IP" "setsid nohup bash -c 'sleep $HARD_DEADLINE; sudo poweroff' >/dev/null 2>&1 < /dev/null &" || true

$SSH "ubuntu@$IP" "bash -s" <<REMOTE
set -uo pipefail
export HF_TOKEN='$HF_TOKEN'
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="\$HOME/.local/bin:\$PATH"
git clone --depth 1 --branch "$BRANCH" "$REPO" repo 2>&1 | tail -2
cd repo
[ "\$(git rev-parse --short=12 HEAD)" = "$EXPECT_COMMIT" ] \
  || { echo "[coref-cuda] *** REFUSING: clone is \$(git rev-parse --short=12 HEAD), expected $EXPECT_COMMIT ***"; exit 1; }
uv venv --python 3.12 2>&1 | tail -2
uv sync 2>&1 | tail -3
uv pip install --reinstall-package torch "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
./.venv/bin/python - <<'PROOF' || { echo "[coref-cuda] *** REFUSING: CUDA unavailable after the cu128 swap ***"; exit 1; }
import sys, torch
print(f"[coref-cuda] torch {torch.__version__} | available {torch.cuda.is_available()} | built for {torch.version.cuda}")
sys.exit(0 if torch.cuda.is_available() else 1)
PROOF

mkdir -p ~/out
( while true; do
    echo "\$(date -u +%H:%M:%S) gpu_used_mib \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) rss_kib \$(ps -o rss= -p \$(pgrep -n -f coref_threshold.py) 2>/dev/null)"
    sleep 20
  done ) > ~/out/mem.log 2>&1 &
MON=\$!
START=\$(date +%s)
COREF_CACHE=\$HOME/out/cache timeout $JOB_TIMEOUT ./.venv/bin/python tools/train/coref_threshold.py \
  --model $MODEL --split val --device cuda --out \$HOME/out/val_cuda.json > \$HOME/out/sweep.log 2>&1
echo "[coref-cuda] sweep exit \$? after \$(( \$(date +%s) - START )) s"
kill \$MON
grep -E "string|PICKED|threshold \+0\.[0-4]0" \$HOME/out/sweep.log
echo "[coref-cuda] memory every 2 min:"; awk 'NR%6==1' ~/out/mem.log
REMOTE

mkdir -p "$OUT"
$SCP -r "ubuntu@$IP:out/*" "$OUT/" && ls -la "$OUT" "$OUT/cache"
echo "[coref-cuda] done; the trap terminates the box on the way out"
