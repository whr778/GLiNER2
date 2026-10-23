#!/bin/bash
# ONE relation-overfit matrix job on Lambda (overfit_job.sh): gates, then launch_when_available on an A10.
#   NAME=overfit-matrix CFG=tools/train/config/base/eb22c-ddp-trace.yaml bash tools/lambda/overfit_launch.sh
# ARMS / CORPORA / CKPT / RECORDS / STEPS pass through to the job. Stops: RUN_TIMEOUT 30 min per run inside the job,
# JOB_TIMEOUT 3 h, HARD_DEADLINE 3.5 h, terminate on the normal path. Laptop reference 2026-10-23: biored 7.6 s/step on
# CPU; the 2026-10-19 A10 did a 64 x 400 corpus in ~25 min on eb21.
set -uo pipefail
NAME=${NAME:?NAME}; CFG=${CFG:?CFG}
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$NAME] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }
export JOB_TIMEOUT=${JOB_TIMEOUT:-10800}
export HARD_DEADLINE=${HARD_DEADLINE:-12600}
echo "[$NAME] commit $EXPECT_COMMIT | logs -> $NAME/"
exec env NAME="$NAME" JOB="CFG=$CFG DEST=$NAME${ARMS:+ ARMS='$ARMS'}${CORPORA:+ CORPORA='$CORPORA'}${CKPT:+ CKPT=$CKPT}${RECORDS:+ RECORDS=$RECORDS}${STEPS:+ STEPS=$STEPS} bash tools/lambda/overfit_job.sh" \
     TYPES="${TYPES:-gpu_1x_a10}" bash tools/lambda/launch_when_available.sh
