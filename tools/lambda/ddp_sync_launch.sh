#!/bin/bash
# DDP weight-sync probe: one 2x H100 box, ddp_sync_probe.sh (DDP tests, then arms A static_graph / B dynamic graph).
# ~30-40 min at $8.38/h. Stops: JOB_TIMEOUT 1.5 h on the job, HARD_DEADLINE 2 h on the box, terminate on the normal path.
#
#   bash tools/lambda/ddp_sync_launch.sh
set -uo pipefail
EXP=ddp-probe
CFG=tools/train/config/base/eb22c.yaml
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }
bash tools/lambda/local_start_check.sh "$CFG" \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- the config cannot start train.py locally ***"; exit 7; }
export JOB_TIMEOUT=${JOB_TIMEOUT:-5400}
export HARD_DEADLINE=${HARD_DEADLINE:-7200}
echo "[$EXP] commit $EXPECT_COMMIT | logs -> ddp-probe/"
exec env NAME="ddp-probe" JOB="bash tools/lambda/ddp_sync_probe.sh" TYPES="${TYPES:-gpu_2x_h100_sxm5}" \
     bash tools/lambda/launch_when_available.sh
