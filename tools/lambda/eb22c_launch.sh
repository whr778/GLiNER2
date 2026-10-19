#!/bin/bash
# eb22c FULL RUN -- one 80 GB H100: the scaling check first (scale_then_train.sh phase A, ~45 min), then 5 epochs
# from scratch on the fastest per-step batch that fits (phase B -> event_base_run.sh: model first, then metrics).
#
#   bash tools/lambda/eb22c_launch.sh
#
# Lambda has no single-GPU A100-80GB; the 80 GB card is the H100 (PCIe $3.29/h preferred, SXM5 $4.29/h fallback).
# Stops: JOB_TIMEOUT 46 h on the job, HARD_DEADLINE 50 h on the box, terminate on the normal path. 5 epochs at the
# A100's speed is ~40 h, so the timeout covers the slow case; expect far less if the H100 scales (~$90-130).
# Epoch checkpoints are pushed (PUSH_EPOCHS=1) so the run can be hand-stopped on any epoch.
set -uo pipefail
EXP=eb22c
CFG=tools/train/config/base/eb22c.yaml
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
[ -f "$CFG" ] || { echo "[$EXP] *** REFUSING TO LAUNCH -- $CFG missing ***"; exit 6; }
uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }
uv run python tools/train/check_label_menus.py --config "$CFG" \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- classification menus disagree ***"; exit 4; }
export JOB_TIMEOUT=${JOB_TIMEOUT:-165600}     # whole job: 46 h
export HARD_DEADLINE=${HARD_DEADLINE:-180000} # whole box incl. provisioning: 50 h
JOB="CFG=$CFG \
OUTDIR=./out/eb22c \
REPO=whr778/gliner2-eb22c \
DEST=eb22c \
PROBE=0 COREF_READOUT=1 PUSH_EPOCHS=1 \
bash tools/lambda/scale_then_train.sh"
echo "[$EXP] commit $EXPECT_COMMIT | model -> whr778/gliner2-eb22c | logs -> eb22c/ (scaling -> eb22c/scaling/)"
exec env NAME="eb22c" JOB="$JOB" TYPES="${TYPES:-gpu_1x_h100_pcie gpu_1x_h100_sxm5}" \
     bash tools/lambda/launch_when_available.sh
