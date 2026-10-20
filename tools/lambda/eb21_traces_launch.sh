#!/bin/bash
# eb21-trace + eb21-5ep-trace on Lambda, one single-GPU box each, in parallel (trace_job.sh): is eb21's events
# collapse-free at 1000-step resolution, and does eb22c's schedule (num_epochs 5) alone make it collapse?
# H100 PCIe $3.29/h preferred, SXM5 $4.29/h fallback; ~5 h + ~4 h, ~$30. Stops: JOB_TIMEOUT 7 h, HARD_DEADLINE 8 h,
# and each job stops itself right after the eval at its STOP_STEP.
#
#   bash tools/lambda/eb21_traces_launch.sh
set -uo pipefail
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[eb21-traces] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
for c in eb21-trace eb21-5ep-trace; do
  uv run python tools/train/check_corpora_fetchable.py --config tools/train/config/base/$c.yaml --offline \
    || { echo "[eb21-traces] *** REFUSING TO LAUNCH -- a corpus of $c is unfetchable ***"; exit 3; }
  bash tools/lambda/local_start_check.sh tools/train/config/base/$c.yaml \
    || { echo "[eb21-traces] *** REFUSING TO LAUNCH -- $c cannot start train.py locally ***"; exit 7; }
done
mkdir -p /Volumes/Development/tmp/launch_eb21
export JOB_TIMEOUT=${JOB_TIMEOUT:-25200}
export HARD_DEADLINE=${HARD_DEADLINE:-28800}
TYPES=${TYPES:-"gpu_1x_h100_pcie gpu_1x_h100_sxm5"}
echo "[eb21-traces] commit $EXPECT_COMMIT | logs -> eb21-trace/ and eb21-5ep-trace/"
env NAME="eb21-trace" JOB="CFG=tools/train/config/base/eb21-trace.yaml DEST=eb21-trace STOP_STEP=12000 bash tools/lambda/trace_job.sh" \
    TYPES="$TYPES" bash tools/lambda/launch_when_available.sh > /Volumes/Development/tmp/launch_eb21/eb21-trace.log 2>&1 &
env NAME="eb21-5ep-trace" JOB="CFG=tools/train/config/base/eb21-5ep-trace.yaml DEST=eb21-5ep-trace STOP_STEP=9000 bash tools/lambda/trace_job.sh" \
    TYPES="$TYPES" bash tools/lambda/launch_when_available.sh > /Volumes/Development/tmp/launch_eb21/eb21-5ep-trace.log 2>&1 &
wait
