#!/bin/bash
# ONE single-GPU trace run on Lambda (trace_job.sh): gates, then launch_when_available.
#   NAME=eb22c-dose3-trace CFG=tools/train/config/base/eb22c-dose3-trace.yaml STOP_STEP=10000 bash tools/lambda/trace_launch.sh
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
bash tools/lambda/local_start_check.sh "$CFG" \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- the config cannot start train.py locally ***"; exit 7; }
export JOB_TIMEOUT=${JOB_TIMEOUT:-21600}
export HARD_DEADLINE=${HARD_DEADLINE:-25200}
echo "[$NAME] commit $EXPECT_COMMIT | stop after eval at step $STOP_STEP | logs -> $NAME/"
exec env NAME="$NAME" JOB="CFG=$CFG DEST=$NAME STOP_STEP=$STOP_STEP bash tools/lambda/trace_job.sh" \
     TYPES="${TYPES:-gpu_1x_h100_sxm5 gpu_1x_h100_pcie}" bash tools/lambda/launch_when_available.sh
