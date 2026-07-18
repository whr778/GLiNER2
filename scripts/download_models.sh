#!/usr/bin/env bash
#
# Download trained GLiNER2 checkpoints from the AWS training box to this laptop,
# into out/fastino/ (git-ignored). Once local, they're auto-discovered by the
# viewer's model registry (out/**/best).
#
# Usage:
#   bash scripts/download_models.sh            # download whatever checkpoints exist now
#   bash scripts/download_models.sh --wait     # wait until the batch finishes, then download
#
# The box IP defaults to the current instance; override with GLINER2_AWS_IP.
# Pulls each config's best/ (+ final/ + last/ if present) and every
# *_metrics.json, plus PAPER.md and the training logs.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

KEY="$HOME/.ssh/id_ed25519"
IP="${GLINER2_AWS_IP:-100.53.18.106}"
REMOTE="ubuntu@$IP"
SSH=(ssh -i "$KEY" -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)

remote() { "${SSH[@]}" "$REMOTE" "$@"; }

wait_for_batch() {
  echo "Waiting for the batch to finish on $IP (polling every 5 min)..."
  while true; do
    if remote 'grep -q "^Summary:" ~/GLiNER2/out/train_logs/batch.log 2>/dev/null'; then
      echo "Batch reports done:"
      remote 'grep "^Summary:" ~/GLiNER2/out/train_logs/batch.log | tail -1'
      break
    fi
    printf '.'
    sleep 300
  done
}

[ "${1:-}" = "--wait" ] && wait_for_batch

echo "Discovering checkpoints on $IP ..."
DIRS="$(remote 'cd ~/GLiNER2 && find out/fastino -maxdepth 2 -type d \( -name best -o -name final -o -name last \) 2>/dev/null | sort')"
if [ -z "$DIRS" ]; then
  echo "No checkpoints found under out/fastino/ on the box yet."
  exit 1
fi
echo "$DIRS" | sed 's/^/  /'

# Pull each checkpoint dir into its matching local parent.
for d in $DIRS; do
  parent="$(dirname "$d")"
  mkdir -p "$parent"
  echo "-> $d"
  rsync -az --info=progress2 -e "ssh -i $KEY" "$REMOTE:GLiNER2/$d" "$parent/"
done

echo "Pulling *_metrics.json (all configs) ..."
rsync -az -m -e "ssh -i $KEY" \
  --include='*/' --include='*_metrics.json' --exclude='*' \
  "$REMOTE:GLiNER2/out/fastino/" out/fastino/

echo "Pulling PAPER.md + training logs ..."
rsync -az -e "ssh -i $KEY" "$REMOTE:GLiNER2/tools/train/PAPER.md" tools/train/PAPER.md 2>/dev/null || true
rsync -az -e "ssh -i $KEY" "$REMOTE:GLiNER2/out/train_logs/" out/train_logs/ 2>/dev/null || true

echo ""
echo "Done. Local checkpoints:"
find out/fastino -maxdepth 2 -type d -name best 2>/dev/null | sed 's/^/  /'
