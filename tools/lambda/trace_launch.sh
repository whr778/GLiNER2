#!/bin/bash
# ONE single-GPU trace run on Lambda (trace_job.sh): gates, then launch_when_available.
#   NAME=eb22c-dose3-trace CFG=tools/train/config/base/eb22c-dose3-trace.yaml STOP_STEP=10000 bash tools/lambda/trace_launch.sh
#   multi-GPU + checkpoints on the Hub: NGPU=2 CKPT_REPO=whr778/<repo> TYPES=gpu_2x_h100_sxm5 ... (same command)
# H100 SXM5 preferred (two PCIe boxes in us-west-3 never left "booting", 2026-10-20), PCIe fallback.
# Stops: the job stops after the eval at STOP_STEP; JOB_TIMEOUT 6 h, HARD_DEADLINE 7 h, terminate on the normal path.
set -uo pipefail
NAME=${NAME:?NAME}; CFG=${CFG:?CFG}; STOP_STEP=${STOP_STEP:?STOP_STEP}
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }
# SKIP_START_CHECK=<reason>: eb22c's head (~120 queries x 768 training candidates) passes 24 GB on the 32 GB laptop by
# step 2 at ANY window size (measured 2026-10-21), so the laptop check cannot finish; the reason is printed and the box's
# own first logging steps carry the proof instead. Default: run it.
if [ -n "${SKIP_START_CHECK:-}" ]; then
  echo "[$NAME] local start check SKIPPED: $SKIP_START_CHECK"
else
  bash tools/lambda/local_start_check.sh "$CFG" \
    || { echo "[$NAME] *** REFUSING TO LAUNCH -- the config cannot start train.py locally ***"; exit 7; }
fi
export JOB_TIMEOUT=${JOB_TIMEOUT:-21600}
export HARD_DEADLINE=${HARD_DEADLINE:-25200}
echo "[$NAME] commit $EXPECT_COMMIT | stop after eval at step $STOP_STEP | logs -> $NAME/"
exec env NAME="$NAME" JOB="CFG=$CFG DEST=$NAME STOP_STEP=$STOP_STEP STOP_EPOCH=${STOP_EPOCH:-} NGPU=${NGPU:-1} CKPT_REPO=${CKPT_REPO:-} bash tools/lambda/trace_job.sh" \
     TYPES="${TYPES:-gpu_1x_h100_sxm5 gpu_1x_h100_pcie}" bash tools/lambda/launch_when_available.sh
