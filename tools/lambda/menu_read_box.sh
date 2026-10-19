#!/bin/bash
# A VAL menu read on CUDA: sweep_menu_threshold.py --val-only for each checkpoint in $CKPTS under $MENU with
# $CFG, threshold grid $GRID. The sweep decodes the whole split once per threshold, which takes hours on a
# laptop CPU; a GPU makes it minutes. Data and models come from the Hub ($HF_TOKEN); the JSONs come back to
# $OUT. No test split is read and nothing is published. Setup is full_menu_timing.sh's -- change them together.
#
# THREE INDEPENDENT STOPS: `timeout` on each sweep, a detached hard-deadline watchdog, terminate in a trap.
#
#   CKPTS="whr778/gliner2-relneg-control whr778/gliner2-relneg-treatment" CFG=tools/train/config/ab/relneg-read.yaml \
#     MENU=corpus_full GRID="0.05 0.1 0.2 0.3 0.4 0.5" OUT=/Volumes/Development/tmp/relneg_read_gpu bash tools/lambda/menu_read_box.sh
set -uo pipefail
NAME=${NAME:-menu-read}
BRANCH=$(git rev-parse --abbrev-ref HEAD)
EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
TYPES=${TYPES:-"gpu_1x_a10 gpu_1x_a100_sxm4"}
SSH_KEY_NAME=${SSH_KEY_NAME:-gliner2-mac}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
REPO=${REPO:-https://github.com/whr778/GLiNER2.git}
JOB_TIMEOUT=${JOB_TIMEOUT:-3600}
HARD_DEADLINE=${HARD_DEADLINE:-9000}
OUT=${OUT:?OUT=<local dir for the JSONs>}
CKPTS=${CKPTS:?CKPTS="<hub id> ..."}
CFG=${CFG:?CFG=<config path in the repo>}
MENU=${MENU:?MENU=corpus_full | app:<name> | widened:K | gold}
GRID=${GRID:-"0.1 0.2 0.3 0.4 0.5"}
: "${LAMBDA_API_KEY:?not set}"
: "${HF_TOKEN:?not set -- the model and val files are private}"

git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[read] *** REFUSING -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }

API=https://cloud.lambda.ai/api/v1
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
SCP="scp -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
ID=""

terminate() {
  [ -n "$ID" ] || return 0
  echo "[read] terminating $ID"
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
  [ -n "$REGION" ] || { echo "[read] no capacity for $t"; continue; }
  ID=$(curl -s -u "$LAMBDA_API_KEY:" -H 'Content-Type: application/json' \
    -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$t\",\"ssh_key_names\":[\"$SSH_KEY_NAME\"],\"name\":\"$NAME\"}" \
    "$API/instance-operations/launch" | python3 -c "
import sys, json; d = json.load(sys.stdin)
print((d.get('data') or {}).get('instance_ids', [''])[0])")
  [ -n "$ID" ] && { echo "[read] launched $t in $REGION as $ID"; break; }
done
[ -n "$ID" ] || { echo "[read] *** could not launch any of: $TYPES ***"; exit 1; }

echo "[read] waiting for the box to come up"
for _ in $(seq 1 60); do
  IP=$(curl -s -u "$LAMBDA_API_KEY:" "$API/instances/$ID" | python3 -c "
import sys, json
try: print((json.load(sys.stdin).get('data') or {}).get('ip') or '')
except Exception: print('')")
  [ -n "$IP" ] && $SSH "ubuntu@$IP" true 2>/dev/null && break
  sleep 20
done
[ -n "${IP:-}" ] || { echo "[read] *** box never got an IP ***"; exit 1; }
echo "[read] box is up at $IP"

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
  || { echo "[read] *** REFUSING: clone is \$(git rev-parse --short=12 HEAD), expected $EXPECT_COMMIT ***"; exit 1; }
uv venv --python 3.12 2>&1 | tail -2
uv sync 2>&1 | tail -3
uv pip install --reinstall-package torch "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
./.venv/bin/python - <<'PROOF' || { echo "[read] *** REFUSING: CUDA unavailable after the cu128 swap ***"; exit 1; }
import sys, torch
print(f"[read] torch {torch.__version__} | available {torch.cuda.is_available()} | built for {torch.version.cuda}")
sys.exit(0 if torch.cuda.is_available() else 1)
PROOF
mkdir -p ~/out
for ck in $CKPTS; do
  name=\$(basename \$ck)
  start=\$(date +%s)
  timeout $JOB_TIMEOUT ./.venv/bin/python tools/train/sweep_menu_threshold.py --config $CFG --checkpoint \$ck --menu $MENU \
    --val-only --device cuda --grid $GRID --out \$HOME/out/\$name.json > \$HOME/out/\$name.log 2>&1
  echo "[read] \$name exit \$? in \$(( \$(date +%s) - start )) s | \$(grep -a -E 'device|chose' \$HOME/out/\$name.log | tr '\n' ' ')"
done
REMOTE

mkdir -p "$OUT"
$SCP -r "ubuntu@$IP:out/*" "$OUT/" && ls -la "$OUT"
echo "[read] done; the trap terminates the box on the way out"
