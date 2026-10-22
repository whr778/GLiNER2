#!/bin/bash
# ONE threshold-robust re-score job on Lambda (rescore_job.sh): gates, then launch_when_available on an A10.
#   NAME=rescore CFG=tools/train/config/base/eb22c-ddp-trace.yaml bash tools/lambda/rescore_launch.sh
# Stops: JOB_TIMEOUT 2 h, HARD_DEADLINE 2.5 h, terminate on the normal path. Expected ~45 min, ~$1-2 on an A10.
set -uo pipefail
NAME=${NAME:?NAME}; CFG=${CFG:?CFG}
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }
export JOB_TIMEOUT=${JOB_TIMEOUT:-7200}
export HARD_DEADLINE=${HARD_DEADLINE:-9000}
echo "[$NAME] commit $EXPECT_COMMIT | logs -> $NAME/"
exec env NAME="$NAME" JOB="CFG=$CFG DEST=$NAME bash tools/lambda/rescore_job.sh" \
     TYPES="${TYPES:-gpu_1x_a10}" bash tools/lambda/launch_when_available.sh
