#!/bin/bash
# RESCORE JOB: diagnose_checkpoint.py (VAL, per-source, threshold grid, precision kept) over the eb22c A/B checkpoints
# on the Hub, one checkpoint at a time on the box's GPU, publishing each result to run-logs DEST/ as it lands.
# On the laptop a checkpoint takes ~5 min on CPU (fp32, sdpa); on a GPU the checkpoint loads with flash attention
# under bf16 autocast -- the SAME instrument as the in-run eval -- in ~1 min. Every checkpoint is scored with that one
# instrument so the comparison across runs and steps is like for like.
#
#   JOB="CFG=tools/train/config/base/eb22c-ddp-trace.yaml DEST=rescore bash tools/lambda/rescore_job.sh"
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1      # flash attention must really be in use, or the load fails loudly
export CKPT_CACHE=$HOME/ckpts
PY=./.venv/bin/python
CFG=${CFG:?CFG}; DEST=${DEST:?DEST}
STEPS_FULL=${STEPS_FULL:-4000,5000,6000,7000,8000,10000,12000,20000,24000,28000,30000}
STEPS_LR=${STEPS_LR:-4000,5000,6000,7000,8000,10000,12000}
OUT=$HOME/rescore
mkdir -p "$OUT"
source tools/lambda/_publish.sh
echo "[rescore] ===== START $(date -u) | $(git rev-parse --short HEAD) | $CFG | full ${STEPS_FULL} | task_lr ${STEPS_LR} ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
failed=0; done_n=0
score_one() {  # repo step tag
  local repo=$1 step=$2 tag=$3 out=$OUT/$tag-$step.json log=$OUT/$tag-$step.log
  local dir; dir=$($PY -c "import sys; sys.path.insert(0,'tools/train'); import probe_event_collapse as P; print(P.local_checkpoint('$repo:checkpoint-$step'))" 2>/dev/null | tail -1)
  if [ ! -d "$dir" ]; then echo "[rescore] $tag $step: checkpoint NOT downloaded ($dir)"; failed=$((failed+1)); return; fi
  PYTHONUNBUFFERED=1 $PY tools/train/diagnose_checkpoint.py --config "$CFG" --checkpoint "$dir" --only maven cmnee duee casie \
     --records 40 --grid 0.01 0.05 0.1 0.2 0.3 0.5 --device cuda --out "$out" > "$log" 2>&1
  local rc=$?
  echo "[rescore] $tag $step: exit $rc | $(grep -c '^\[diag\] [a-z]* (' "$log") sources scored | $(date -u +%H:%M:%S)"
  [ "$rc" -eq 0 ] && [ -s "$out" ] && done_n=$((done_n+1)) || { failed=$((failed+1)); grep -aE "Error|Traceback" "$log" | tail -2; }
  publish "$DEST" "$log" $([ -s "$out" ] && echo "$out") || echo "[rescore] publish of $tag-$step failed (continuing; final publish retries)"
  rm -rf "$dir"
}
for s in ${STEPS_FULL//,/ }; do
  score_one whr778/gliner2-eb22c-collapse-trace "$s" control
  score_one whr778/gliner2-eb22c-seed43-trace "$s" seed43
  score_one whr778/gliner2-eb22c-sep-trace "$s" sep
done
for s in ${STEPS_LR//,/ }; do score_one whr778/gliner2-eb22c-tasklr4e5-trace "$s" tasklr; done
echo "[rescore] scored $done_n, failed $failed $(date -u)"
publish "$DEST" "$OUT"/*.json "$OUT"/*.log || { echo "[rescore] *** FATAL: FINAL PUBLISH FAILED ***"; exit 1; }
[ "$failed" -eq 0 ] || { echo "[rescore] *** FATAL: $failed checkpoint(s) failed (results published) ***"; exit 2; }
echo "[rescore] ===== DONE $(date -u) ====="
