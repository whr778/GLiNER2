#!/bin/bash
# Launch the MENU THRESHOLD job (menu_threshold_job.sh) on one A100. Commit pinned; three stops: JOB_TIMEOUT,
# the hard-deadline watchdog, and terminate on the normal path.
#   ARMS=menuref-full,menuref-goldfree bash tools/lambda/menu_threshold_launch.sh
set -uo pipefail
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[mt] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
# Measured 2026-10-07: dose-20 arm ~40 min; the over-firing dose-1 control >80 min (it decodes thousands of
# spans per doc under the 180-label menu). Size the timeout for the slowest arm you pass in ARMS.
export JOB_TIMEOUT=${JOB_TIMEOUT:-14400}     # whole job: 4 h
export HARD_DEADLINE=${HARD_DEADLINE:-16200} # whole box: 4.5 h
ARMS=${ARMS:-menudose-treatment,menudose-control}
case "$ARMS" in *[!a-z0-9,-]*) echo "[mt] ARMS must be comma-separated names, no spaces: $ARMS"; exit 7;; esac
exec env NAME="menu-threshold" JOB="ARMS=$ARMS bash tools/lambda/menu_threshold_job.sh" TYPES="${TYPES:-gpu_1x_a100_sxm4}" \
     bash tools/lambda/launch_when_available.sh
