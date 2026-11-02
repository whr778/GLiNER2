#!/bin/bash
# WikiEvents dev at threshold 0.1 and test at 0.5, on eb22e best.
# Same splits and thresholds as life-mdeberta-wikievents. The window is 4096.
# Weights only. This job does not train.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1
PY=./.venv/bin/python
CFG=tools/train/config/trace/eb22e-wikievents-eval.yaml
DEST=eb22e-wikievents-eval
CKPT=$HOME/ckpt/best
source tools/lambda/_publish.sh
mkdir -p "$HOME/ckpt" "$HOME/eb22e-wikievents-eval"

fail() {
  echo "[wikievents-eval] *** FATAL: $* ***"
  printf '%s\n' "[wikievents-eval] *** FATAL: $* ***" >> "$HOME/eb22e-wikievents-eval/fatal.txt"
  publish "$DEST" "$HOME/eb22e-wikievents-eval/fatal.txt" || echo "[wikievents-eval] the failure note did not reach the hub"
  exit 1
}

echo "[wikievents-eval] ===== START $(date -u) | $(git rev-parse --short HEAD) | whr778/gliner2-eb22e-epoch:best ====="

$PY - "$CKPT" <<'PY' || fail "download failed"
import os, sys
from huggingface_hub import snapshot_download
out = os.path.dirname(sys.argv[1])
snapshot_download(
    "whr778/gliner2-eb22e-epoch",
    allow_patterns=["best/*"],
    ignore_patterns=["best/training_state.pt"],
    local_dir=out,
    token=os.environ["HF_TOKEN"],
)
PY
[ -s "$CKPT/model.safetensors" ] || fail "$CKPT has no weights"
echo "[wikievents-eval] downloaded $(date -u) | $(ls "$CKPT" | tr '\n' ' ')"

score() {
  local split=$1 threshold=$2
  echo "[wikievents-eval] $split threshold $threshold $(date -u)"
  $PY -u tools/train/eval.py --config "$CFG" --checkpoint "$CKPT" --split "$split" \
    --threshold "$threshold" --batch-size 2 \
    > "$HOME/eb22e-wikievents-eval/${split}.log" 2>&1
  local rc=$?
  cat "$HOME/eb22e-wikievents-eval/${split}.log"
  [ "$rc" -eq 0 ] || fail "$split eval failed"
  [ -s "out/eb22e-wikievents-eval/${split}_metrics.json" ] || fail "$split metrics were not written"
}

score val 0.1
score test 0.5

publish "$DEST" \
  "$HOME/eb22e-wikievents-eval/val.log" \
  "$HOME/eb22e-wikievents-eval/test.log" \
  out/eb22e-wikievents-eval/val_metrics.json \
  out/eb22e-wikievents-eval/test_metrics.json \
  || fail "publish failed"
echo "[wikievents-eval] ===== DONE $(date -u) ====="
