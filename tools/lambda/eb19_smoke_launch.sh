#!/bin/bash
# eb19 SMOKE launcher -- one A100, 1500 steps of eb19.yaml FROM SCRATCH (tools/lambda/eb19_smoke.sh).
#
#   bash tools/lambda/eb19_smoke_launch.sh
#
# ~24k samples at eb18's 13.7 samples/s = ~30 min training + ~15 min provisioning, ~$1.50.
# Stops: the smoke's own `timeout 5400` on train.py, JOB_TIMEOUT 2 h, HARD_DEADLINE 3 h, terminate
# on the normal path. Commit pinned: the box must run exactly this commit.
set -uo pipefail
EXP=eb19_smoke
CFG=tools/train/config/base/eb19.yaml

export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }

uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }

export JOB_TIMEOUT=${JOB_TIMEOUT:-7200}
export HARD_DEADLINE=${HARD_DEADLINE:-10800}

echo "[$EXP] commit $EXPECT_COMMIT  logs -> whr778/gliner2-run-logs:eb19_smoke/"
exec env NAME="eb19-smoke" JOB="STEPS=${STEPS:-1500} bash tools/lambda/eb19_smoke.sh" \
     TYPES="${TYPES:-gpu_1x_a100_sxm4}" bash tools/lambda/launch_when_available.sh
