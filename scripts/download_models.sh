#!/usr/bin/env bash
#
# Download trained GLiNER2 checkpoints from the AWS training box to this laptop,
# into out/fastino/ (git-ignored). Once local, they're auto-discovered by the
# viewer's model registry (out/**/best).
#
# Usage:
#   bash scripts/download_models.sh             # download whatever checkpoints exist now
#   bash scripts/download_models.sh --wait      # wait until the batch finishes, then download
#   bash scripts/download_models.sh --parallel  # fan out N concurrent streams (default 8)
#
# --parallel is far faster over a per-flow-limited or high-latency path (e.g. a
# VPN tunnel): a single rsync-over-SSH stream is capped by the bandwidth-delay
# product and collapses under loss, but aggregate throughput scales with the
# number of concurrent streams. Set the count with GLINER2_JOBS. Flags combine
# (e.g. --wait --parallel). The box IP defaults to the current instance; override
# with GLINER2_AWS_IP. Pulls each config's best/ (+ final/ + last/ if present)
# and every *_metrics.json, plus PAPER.md and the training logs.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

KEY="$HOME/.ssh/id_ed25519"
IP="${GLINER2_AWS_IP:-100.53.18.106}"
REMOTE="ubuntu@$IP"
JOBS="${GLINER2_JOBS:-8}"
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

# Pull every checkpoint dir concurrently. Aggregate throughput over a per-flow-
# throttled / lossy path scales with concurrency. Drop -z: 8 gzip streams would
# contend for the box's few vCPUs and safetensors barely compress, so
# compression only slows a parallel run.
download_parallel() {
  echo "Downloading with $JOBS parallel streams (compression off) ..."
  export GL_KEY="$KEY" GL_REMOTE="$REMOTE"
  echo "$DIRS" | xargs -P "$JOBS" -I{} sh -c '
    parent=$(dirname "{}"); mkdir -p "$parent"
    if rsync -a -e "ssh -i $GL_KEY" "$GL_REMOTE:GLiNER2/{}" "$parent/"; then
      echo "  ok:   {}"
    else
      echo "  FAIL: {}"
    fi
  '
}

WAIT=0
PARALLEL=0
for arg in "$@"; do
  case "$arg" in
    --wait) WAIT=1 ;;
    --parallel) PARALLEL=1 ;;
    *) echo "unknown option: $arg (use --wait and/or --parallel)" >&2; exit 1 ;;
  esac
done
[ "$WAIT" = 1 ] && wait_for_batch

echo "Discovering checkpoints on $IP ..."
DIRS="$(remote 'cd ~/GLiNER2 && find out/fastino -maxdepth 2 -type d \( -name best -o -name final -o -name last \) 2>/dev/null | sort')"
if [ -z "$DIRS" ]; then
  echo "No checkpoints found under out/fastino/ on the box yet."
  exit 1
fi
echo "$DIRS" | sed 's/^/  /'

# Pull each checkpoint dir into its matching local parent.
if [ "$PARALLEL" = 1 ]; then
  download_parallel
else
  for d in $DIRS; do
    parent="$(dirname "$d")"
    mkdir -p "$parent"
    echo "-> $d"
    rsync -az --progress -e "ssh -i $KEY" "$REMOTE:GLiNER2/$d" "$parent/"
  done
fi

echo "Pulling *_metrics.json (all configs) ..."
rsync -az --prune-empty-dirs -e "ssh -i $KEY" \
  --include='*/' --include='*_metrics.json' --exclude='*' \
  "$REMOTE:GLiNER2/out/fastino/" out/fastino/

echo "Pulling PAPER.md + training logs ..."
rsync -az -e "ssh -i $KEY" "$REMOTE:GLiNER2/tools/train/PAPER.md" tools/train/PAPER.md 2>/dev/null || true
rsync -az -e "ssh -i $KEY" "$REMOTE:GLiNER2/out/train_logs/" out/train_logs/ 2>/dev/null || true

echo ""
echo "Done. Local checkpoints:"
find out/fastino -maxdepth 2 -type d -name best 2>/dev/null | sed 's/^/  /'
