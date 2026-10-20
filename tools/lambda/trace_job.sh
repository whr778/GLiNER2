#!/bin/bash
# TRACE JOB: plain single-GPU train.py on CFG, stopped right after the eval at STOP_STEP, publishing the log and
# eval_history.jsonl to run-logs DEST/ every 20 minutes and at the end.
#
# Why a step stop and not max_steps: max_steps can resize the LR schedule, and the schedule is what the eb21 traces
# test (eb22c's events collapse at steps ~4-8k on one GPU, 2026-10-20). The run keeps its real num_epochs; this job
# only decides when to stop watching.
#
#   JOB="CFG=tools/train/config/base/eb21-trace.yaml DEST=eb21-trace STOP_STEP=12000 bash tools/lambda/trace_job.sh"
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1
PY=./.venv/bin/python
CFG=${CFG:?CFG}; DEST=${DEST:?DEST}; STOP_STEP=${STOP_STEP:?STOP_STEP}
OUTDIR=$($PY -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['output_dir'])")
HIST=$OUTDIR/eval_history.jsonl
LOG=$HOME/trace.log
source tools/lambda/_publish.sh
echo "[trace] ===== START $(date -u) | $(git rev-parse --short HEAD) | $CFG | stop after eval at step $STOP_STEP ====="

PYTHONUNBUFFERED=1 $PY -u tools/train/train.py --config "$CFG" > "$LOG" 2>&1 &
TRAIN=$!
last_pub=$(date +%s)
last_step() { [ -s "$HIST" ] && $PY -c "import json; print(json.loads(open('$HIST').read().splitlines()[-1])['eval_step'])" || echo 0; }
while kill -0 $TRAIN 2>/dev/null; do
  sleep 60
  step=$(last_step)
  if [ "$step" -ge "$STOP_STEP" ]; then
    echo "[trace] eval at step $step >= $STOP_STEP -- stopping training $(date -u)"
    kill -9 $TRAIN 2>/dev/null
    break
  fi
  if [ $(( $(date +%s) - last_pub )) -ge 1200 ]; then
    publish "$DEST" "$LOG" $([ -s "$HIST" ] && echo "$HIST") || echo "[trace] periodic publish failed (continuing)"
    last_pub=$(date +%s)
  fi
done
wait $TRAIN 2>/dev/null
echo "[trace] training ended at eval step $(last_step) $(date -u)"
publish "$DEST" "$LOG" "$HIST" || { echo "[trace] *** FINAL PUBLISH FAILED ***"; exit 1; }
echo "[trace] ===== DONE $(date -u) ====="
