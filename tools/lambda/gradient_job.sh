#!/bin/bash
# GRADIENT CONFLICT JOB: tools/train/gradient_conflict.py on CUDA (bf16 autocast, batch 4 -- the training recipe)
# over the eb22c autopsy checkpoints, then publish the log and JSON to run-logs DEST/ (verified on the Hub).
# On CPU one training forward at batch 4 needs ~44 GB in fp32 (the H100 run logged peak_alloc 22 GB in bf16), which
# swapped a 32 GB laptop on 2026-10-21; that is why this runs on a GPU.
#
#   JOB="CFG=tools/train/config/base/eb22c-ddp-trace.yaml DEST=gradient-conflict bash tools/lambda/gradient_job.sh"
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
PY=./.venv/bin/python
CFG=${CFG:?CFG}; DEST=${DEST:?DEST}
R=${CKPT_REPO:-whr778/gliner2-eb22c-collapse-trace}
STEPS=${STEPS:-3500,4000,4500,22000,24000}   # comma-separated: JOB travels inside bash -lc '...' quotes
OUT=$HOME/conflict.json
LOG=$HOME/conflict.log
source tools/lambda/_publish.sh
echo "[gradient] ===== START $(date -u) | $(git rev-parse --short HEAD) | $CFG | checkpoints $STEPS ====="
CKPTS=""
for s in ${STEPS//,/ }; do CKPTS="$CKPTS ${R}:checkpoint-$s"; done
PYTHONUNBUFFERED=1 $PY tools/train/gradient_conflict.py --config "$CFG" --checkpoints $CKPTS \
  --records 600 --batches 48 --batch-size 4 --device cuda --verify-link 3 --out "$OUT" > "$LOG" 2>&1
RC=$?
echo "[gradient] gradient_conflict exit $RC $(date -u)"
grep -E "^\[conflict\] (/|.*: step|verify-link)" "$LOG" | head -30
publish "$DEST" "$LOG" $([ -s "$OUT" ] && echo "$OUT") || { echo "[gradient] *** FATAL: PUBLISH FAILED ***"; exit 1; }
[ "$RC" -eq 0 ] || { echo "[gradient] *** FATAL: gradient_conflict FAILED (log published) ***"; exit "$RC"; }
echo "[gradient] ===== DONE $(date -u) ====="
