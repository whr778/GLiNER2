#!/bin/bash
# Relation-head OVERFIT trace on CUDA: tools/train/overfit_relations.py for each corpus in $CORPORA
# ($RECORDS train records, $STEPS steps) from $CKPT with $CFG. Can the relation head fit a few
# training records? JSONs and logs come back to $OUT. Nothing is published. Setup is menu_read_box.sh's.
#
# THREE INDEPENDENT STOPS: `timeout` on each run, a detached hard-deadline watchdog, terminate in a trap.
#
#   CORPORA="sentence_rex professorbob_re biored" OUT=/Volumes/Development/tmp/overfit_gpu bash tools/lambda/overfit_box.sh
set -uo pipefail
NAME=${NAME:-overfit-relations}
BRANCH=$(git rev-parse --abbrev-ref HEAD)
EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
TYPES=${TYPES:-"gpu_1x_a10 gpu_1x_a100_sxm4"}
SSH_KEY_NAME=${SSH_KEY_NAME:-gliner2-mac}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
REPO=${REPO:-https://github.com/whr778/GLiNER2.git}
JOB_TIMEOUT=${JOB_TIMEOUT:-3600}
HARD_DEADLINE=${HARD_DEADLINE:-9000}
OUT=${OUT:?OUT=<local dir for the JSONs>}
CORPORA=${CORPORA:?CORPORA="<corpus> ..."}
CKPT=${CKPT:-whr778/gliner2-eb21}
CFG=${CFG:-tools/train/config/base/eb21.yaml}
RECORDS=${RECORDS:-64}
STEPS=${STEPS:-400}
EXTRA=${EXTRA:-}
: "${LAMBDA_API_KEY:?not set}"
: "${HF_TOKEN:?not set -- the model and val files are private}"

git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[overfit-box] *** REFUSING -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }

API=https://cloud.lambda.ai/api/v1
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
SCP="scp -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
ID=""

terminate() {
  [ -n "$ID" ] || return 0
  echo "[overfit-box] terminating $ID"
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
  [ -n "$REGION" ] || { echo "[overfit-box] no capacity for $t"; continue; }
  ID=$(curl -s -u "$LAMBDA_API_KEY:" -H 'Content-Type: application/json' \
    -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$t\",\"ssh_key_names\":[\"$SSH_KEY_NAME\"],\"name\":\"$NAME\"}" \
    "$API/instance-operations/launch" | python3 -c "
import sys, json; d = json.load(sys.stdin)
print((d.get('data') or {}).get('instance_ids', [''])[0])")
  [ -n "$ID" ] && { echo "[overfit-box] launched $t in $REGION as $ID"; break; }
done
[ -n "$ID" ] || { echo "[overfit-box] *** could not launch any of: $TYPES ***"; exit 1; }

echo "[overfit-box] waiting for the box to come up"
for _ in $(seq 1 60); do
  IP=$(curl -s -u "$LAMBDA_API_KEY:" "$API/instances/$ID" | python3 -c "
import sys, json
try: print((json.load(sys.stdin).get('data') or {}).get('ip') or '')
except Exception: print('')")
  [ -n "$IP" ] && $SSH "ubuntu@$IP" true 2>/dev/null && break
  sleep 20
done
[ -n "${IP:-}" ] || { echo "[overfit-box] *** box never got an IP ***"; exit 1; }
echo "[overfit-box] box is up at $IP"

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
  || { echo "[overfit-box] *** REFUSING: clone is \$(git rev-parse --short=12 HEAD), expected $EXPECT_COMMIT ***"; exit 1; }
uv venv --python 3.12 2>&1 | tail -2
uv sync 2>&1 | tail -3
uv pip install --reinstall-package torch "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
./.venv/bin/python - <<'PROOF' || { echo "[overfit-box] *** REFUSING: CUDA unavailable after the cu128 swap ***"; exit 1; }
import sys, torch
print(f"[overfit-box] torch {torch.__version__} | available {torch.cuda.is_available()} | built for {torch.version.cuda}")
sys.exit(0 if torch.cuda.is_available() else 1)
PROOF
mkdir -p ~/out
for c in $CORPORA; do
  start=\$(date +%s)
  PYTHONUNBUFFERED=1 timeout $JOB_TIMEOUT ./.venv/bin/python tools/train/overfit_relations.py --config $CFG --checkpoint $CKPT \
    --corpus \$c --records $RECORDS --steps $STEPS --device cuda $EXTRA --out \$HOME/out/\$c.json > \$HOME/out/\$c.log 2>&1
  echo "[overfit-box] \$c exit \$? in \$(( \$(date +%s) - start )) s"
  grep -a "^\[overfit\] {" \$HOME/out/\$c.log | tail -1 | cut -c1-400
done
REMOTE

mkdir -p "$OUT"
$SCP -r "ubuntu@$IP:out/*" "$OUT/" && ls -la "$OUT"
echo "[overfit-box] done; the trap terminates the box on the way out"
