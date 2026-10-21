#!/bin/bash
# ONE gradient-conflict job on Lambda (gradient_job.sh): gates, then launch_when_available.
#   NAME=gradient-conflict CFG=tools/train/config/base/eb22c-ddp-trace.yaml bash tools/lambda/gradient_launch.sh
# Stops: JOB_TIMEOUT 1.5 h, HARD_DEADLINE 2 h, terminate on the normal path. Expected ~25 min on an H100.
set -uo pipefail
NAME=${NAME:?NAME}; CFG=${CFG:?CFG}
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }
export JOB_TIMEOUT=${JOB_TIMEOUT:-5400}
export HARD_DEADLINE=${HARD_DEADLINE:-7200}
echo "[$NAME] commit $EXPECT_COMMIT | logs -> $NAME/"
exec env NAME="$NAME" JOB="CFG=$CFG DEST=$NAME STEPS='${STEPS:-3500 4000 4500 22000 24000}' bash tools/lambda/gradient_job.sh" \
     TYPES="${TYPES:-gpu_1x_h100_sxm5 gpu_1x_h100_pcie gpu_1x_a100_sxm4}" bash tools/lambda/launch_when_available.sh
