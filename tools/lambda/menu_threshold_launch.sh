#!/bin/bash
# Launch the MENU THRESHOLD job (menu_threshold_job.sh) on one A100. Commit pinned; three stops: JOB_TIMEOUT,
# the hard-deadline watchdog, and terminate on the normal path.
#   bash tools/lambda/menu_threshold_launch.sh
set -uo pipefail
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[mt] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
export JOB_TIMEOUT=${JOB_TIMEOUT:-7200}      # whole job: 2 h (estimate ~1 h)
export HARD_DEADLINE=${HARD_DEADLINE:-10800} # whole box: 3 h
exec env NAME="menu-threshold" JOB="bash tools/lambda/menu_threshold_job.sh" TYPES="${TYPES:-gpu_1x_a100_sxm4}" \
     bash tools/lambda/launch_when_available.sh
