#!/bin/bash
# Full-menu inference timing on CUDA: infer.py with one shared --schema-json (corefC's 810-entity menu),
# global decode, coreferent-trigger link merge, argument links, and entity coref on corefC-treatment.
# Question: seconds per doc on a GPU, to price the full blind test (2,956 docs x 3 models; ~130 s/doc on
# CPU). Inputs ($INPUTS: timing20.jsonl, timing5.jsonl, corefC_model_schema.json) go up by scp; predictions
# and logs come back to $OUT. Nothing is published. Setup is run_tests_cuda.sh's -- change them together.
#
# THREE INDEPENDENT STOPS: `timeout` on the sweep, a detached hard-deadline watchdog, terminate in a trap.
#
#   bash tools/lambda/full_menu_timing.sh
set -uo pipefail
NAME=${NAME:-full-menu-timing}
BRANCH=$(git rev-parse --abbrev-ref HEAD)
EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
TYPES=${TYPES:-"gpu_1x_a10 gpu_1x_a100_sxm4"}
SSH_KEY_NAME=${SSH_KEY_NAME:-gliner2-mac}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
REPO=${REPO:-https://github.com/whr778/GLiNER2.git}
JOB_TIMEOUT=${JOB_TIMEOUT:-1500}
HARD_DEADLINE=${HARD_DEADLINE:-5400}
OUT=${OUT:-/Volumes/Development/tmp/full_menu_timing}
INPUTS=${INPUTS:-/Volumes/Development/tmp/blind_full}
: "${LAMBDA_API_KEY:?not set}"
: "${HF_TOKEN:?not set -- the model and val files are private}"

git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[timing] *** REFUSING -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }

API=https://cloud.lambda.ai/api/v1
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
SCP="scp -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
ID=""

terminate() {
  [ -n "$ID" ] || return 0
  echo "[timing] terminating $ID"
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
  [ -n "$REGION" ] || { echo "[timing] no capacity for $t"; continue; }
  ID=$(curl -s -u "$LAMBDA_API_KEY:" -H 'Content-Type: application/json' \
    -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$t\",\"ssh_key_names\":[\"$SSH_KEY_NAME\"],\"name\":\"$NAME\"}" \
    "$API/instance-operations/launch" | python3 -c "
import sys, json; d = json.load(sys.stdin)
print((d.get('data') or {}).get('instance_ids', [''])[0])")
  [ -n "$ID" ] && { echo "[timing] launched $t in $REGION as $ID"; break; }
done
[ -n "$ID" ] || { echo "[timing] *** could not launch any of: $TYPES ***"; exit 1; }

echo "[timing] waiting for the box to come up"
for _ in $(seq 1 60); do
  IP=$(curl -s -u "$LAMBDA_API_KEY:" "$API/instances/$ID" | python3 -c "
import sys, json
try: print((json.load(sys.stdin).get('data') or {}).get('ip') or '')
except Exception: print('')")
  [ -n "$IP" ] && $SSH "ubuntu@$IP" true 2>/dev/null && break
  sleep 20
done
[ -n "${IP:-}" ] || { echo "[timing] *** box never got an IP ***"; exit 1; }
echo "[timing] box is up at $IP"

$SSH "ubuntu@$IP" 'nvidia-smi --query-gpu=name,ecc.mode.current,ecc.errors.uncorrected.volatile.total --format=csv,noheader'
$SSH "ubuntu@$IP" "setsid nohup bash -c 'sleep $HARD_DEADLINE; sudo poweroff' >/dev/null 2>&1 < /dev/null &" || true

$SCP "$INPUTS/timing20.jsonl" "$INPUTS/timing5.jsonl" "$INPUTS/corefC_model_schema.json" "ubuntu@$IP:" \
  || { echo "[timing] *** could not copy inputs ***"; exit 1; }

$SSH "ubuntu@$IP" "bash -s" <<REMOTE
set -uo pipefail
export HF_TOKEN='$HF_TOKEN'
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="\$HOME/.local/bin:\$PATH"
git clone --depth 1 --branch "$BRANCH" "$REPO" repo 2>&1 | tail -2
cd repo
[ "\$(git rev-parse --short=12 HEAD)" = "$EXPECT_COMMIT" ] \
  || { echo "[timing] *** REFUSING: clone is \$(git rev-parse --short=12 HEAD), expected $EXPECT_COMMIT ***"; exit 1; }
uv venv --python 3.12 2>&1 | tail -2
uv sync 2>&1 | tail -3
uv pip install --reinstall-package torch "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
./.venv/bin/python - <<'PROOF' || { echo "[timing] *** REFUSING: CUDA unavailable after the cu128 swap ***"; exit 1; }
import sys, torch
print(f"[timing] torch {torch.__version__} | available {torch.cuda.is_available()} | built for {torch.version.cuda}")
sys.exit(0 if torch.cuda.is_available() else 1)
PROOF

mkdir -p ~/out
( while true; do
    echo "\$(date -u +%H:%M:%S) gpu_used_mib \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) rss_kib \$(ps -o rss= -p \$(pgrep -n -f tools/infer.py) 2>/dev/null)"
    sleep 20
  done ) > ~/out/mem.log 2>&1 &
MON=\$!
COMMON="--schema-json \$HOME/corefC_model_schema.json --global-decode --merge-coreferent link --include-spans --link-arguments --labels-file tools/train/config/labels/unified-full-v3.yaml --device cuda"
run() {  # name model input extra...
  local name=\$1 model=\$2 input=\$3; shift 3
  local start=\$(date +%s)
  timeout $JOB_TIMEOUT ./.venv/bin/python tools/infer.py --model \$model --input \$HOME/\$input \$COMMON "\$@" \
    --output \$HOME/out/\$name.pred.jsonl > \$HOME/out/\$name.log 2>&1
  echo "[timing] \$name exit \$? | \$(wc -l < \$HOME/out/\$name.pred.jsonl) docs in \$(( \$(date +%s) - start )) s (incl. model load)"
  grep "^\[infer\]" \$HOME/out/\$name.log | grep -E "device|merge|schema" 
}
run treatment20 whr778/gliner2-corefC-treatment timing20.jsonl --entity-coref 0.2
run treatment5 whr778/gliner2-corefC-treatment timing5.jsonl --entity-coref 0.2
run eb21_5 whr778/gliner2-eb21 timing5.jsonl
kill \$MON
echo "[timing] peak gpu_used_mib \$(awk '{print \$3}' ~/out/mem.log | sort -n | tail -1)"
REMOTE

mkdir -p "$OUT"
$SCP -r "ubuntu@$IP:out/*" "$OUT/" && ls -la "$OUT"
echo "[timing] done; the trap terminates the box on the way out"
