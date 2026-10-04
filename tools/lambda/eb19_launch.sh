#!/bin/bash
# eb19 BASE RUN -- one box, full schedule (tools/train/config/base/eb19.yaml).
#
#   bash tools/lambda/eb19_launch.sh
#
# eb18's recipe + record_role_hard_negatives 8 + proposal_gold identity, 9 epochs (see the config header).
# Measured on eb18 (A100): ~4.2 h per epoch incl. validation -> ~38 h train + ~1.2 h sweep/blind test +
# per-epoch pushes. Stops: JOB_TIMEOUT 50 h (whole job), HARD_DEADLINE 54 h, terminate on the normal path.
# Model first, then metrics, then (non-fatal) every checkpoint-epoch-N to whr778/gliner2-eb19-checkpoint-epoch-N
# and the trigger-miss probe on best/. Commit pinned: the box must run exactly this commit.
set -uo pipefail
EXP=eb19
CFG=tools/train/config/base/eb19.yaml

# THE BOX MUST RUN THIS COMMIT. provision_box.sh defaults BRANCH to merge/main-20260805; the
# first p2fast launch inherited that, cloned a branch without these configs, and all four
# runners died on FileNotFoundError. Pass the branch AND the commit; the provisioner refuses
# a clone that does not match. Refuse here if the commit is not on origin yet.
export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
[ -f "$CFG" ] || { echo "[$EXP] *** REFUSING TO LAUNCH -- $CFG missing ***"; exit 6; }

uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }
uv run python tools/train/check_label_menus.py --config "$CFG" \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- classification menus disagree ***"; exit 4; }

export JOB_TIMEOUT=${JOB_TIMEOUT:-180000}     # whole job: 50 h
export HARD_DEADLINE=${HARD_DEADLINE:-194400} # whole box incl. provisioning: 54 h

PROBE_ENV="PROBE=1 PUSH_EPOCHS=1"

JOB="CFG=$CFG \
OUTDIR=./out/eb19 \
REPO=whr778/gliner2-eb19 \
DEST=eb19 \
$PROBE_ENV \
bash tools/lambda/event_base_run.sh"

echo "[$EXP] model -> whr778/gliner2-eb19  logs -> eb19/  epochs -> whr778/gliner2-eb19-checkpoint-epoch-N"
# PIN THE CARD: arms on different silicon are not a matched A/B.
exec env NAME="eb19" JOB="$JOB" TYPES="${TYPES:-gpu_1x_a100_sxm4}" \
     bash tools/lambda/launch_when_available.sh
