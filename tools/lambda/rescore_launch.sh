#!/bin/bash
# ONE threshold-robust re-score job on Lambda (rescore_job.sh): gates, then launch_when_available on an A10.
#   NAME=rescore CFG=tools/train/config/base/eb22c-ddp-trace.yaml bash tools/lambda/rescore_launch.sh
# Stops: JOB_TIMEOUT 3.5 h, HARD_DEADLINE 4 h, terminate on the normal path. MEASURED 2026-10-22: 257 s per checkpoint on an
# A10 (4 sources x 40 records x 6 thresholds), so the 40 checkpoints take ~2.9 h, ~$4. STEPS_FULL / STEPS_LR (comma lists)
# pass through, so a cut-off tail can be scored by a second launch without redoing the rest.
set -uo pipefail
NAME=${NAME:?NAME}; CFG=${CFG:?CFG}
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }
export JOB_TIMEOUT=${JOB_TIMEOUT:-12600}
export HARD_DEADLINE=${HARD_DEADLINE:-14400}
echo "[$NAME] commit $EXPECT_COMMIT | logs -> $NAME/"
exec env NAME="$NAME" JOB="CFG=$CFG DEST=$NAME${STEPS_FULL:+ STEPS_FULL=$STEPS_FULL}${STEPS_LR:+ STEPS_LR=$STEPS_LR} bash tools/lambda/rescore_job.sh" \
     TYPES="${TYPES:-gpu_1x_a10}" bash tools/lambda/launch_when_available.sh
