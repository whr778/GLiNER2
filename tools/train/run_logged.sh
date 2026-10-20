#!/bin/bash
# Run train.py under torchrun with EVERYTHING it prints captured to a timestamped log, detached from the terminal.
#
# The trainer logs to the terminal only; a run launched by hand left no record of its losses or evals (eb22c at the
# office, 2026-10-20). The log opens with the commit, uncommitted changes, GPUs and command, and a copy of the config
# is written into it, so the log says exactly what ran.
#
#   bash tools/train/run_logged.sh tools/train/config/base/eb22c.yaml 2      # config, number of GPUs
set -uo pipefail
CFG=${1:?usage: run_logged.sh <config> [num_gpus]}
NGPU=${2:-1}
NAME=$(basename "$CFG" .yaml)
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
LOG=logs/$NAME-$STAMP.log
# ONE GPU = plain train.py, exactly as eb21 trained. torchrun --nproc_per_node=1 sets LOCAL_RANK, which turns on the
# trainer's DDP path (world size 1: gradients measured bit-identical, but a different sampler and code path).
if [ "$NGPU" -eq 1 ]; then
  CMD="uv run tools/train/train.py --config $CFG"
else
  CMD="uv run torchrun --standalone --nproc_per_node=$NGPU tools/train/train.py --config $CFG"
fi
{
  echo "[run] $(date) | commit $(git rev-parse --short HEAD) | config $CFG | gpus $NGPU"
  echo "[run] command: $CMD"
  git status --short | sed 's/^/[run] uncommitted: /'
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[run] gpu: /'
} > "$LOG"
{ echo "[run] ---- config $CFG ----"; cat "$CFG"; echo "[run] ---- end config ----"; } >> "$LOG"
PYTHONUNBUFFERED=1 nohup $CMD >> "$LOG" 2>&1 &
echo "running as pid $! -> $LOG   (watch: tail -f $LOG)"
