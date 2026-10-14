#!/bin/bash
# eb21 BASE RUN -- one box, full schedule (tools/train/config/base/eb21.yaml).
#
#   bash tools/lambda/eb21_launch.sh
#
# eb20 + unified-full-v3 labels (Title_Snake) + label negatives dose 20/20 + the entity-order
# shuffle, on the fastino v2.1.0 merge (merge/upstream-20261014), 9 epochs (see the config header).
# Timing: the eb21 smoke ran 9.5 samples/s (eb19 15.3) on 237,901 train records -> ~7.2 h per epoch
# incl. validation, so 9 epochs (~65 h) do NOT fit the stops. DECIDED 2026-10-14 (option B): keep
# JOB_TIMEOUT 50 h / HARD_DEADLINE 54 h and HAND-STOP when the selection metric flattens (the eb19
# recipe, memory eb19-smoke-clean). If the timeout fires first (~epoch 6-7), the runner pushes best/
# and every epoch, finds no test_metrics.json and HOLDS until the deadline: ~4 h to run the sweep +
# blind test on the box, or run them later from the Hub checkpoint. ~$100 at $1.99/h.
# Model first, then metrics, then (non-fatal) every checkpoint-epoch-N to whr778/gliner2-eb21-checkpoint-epoch-N
# and the trigger-miss probe on best/. Commit pinned: the box must run exactly this commit.
set -uo pipefail
EXP=eb21
CFG=tools/train/config/base/eb21.yaml

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
OUTDIR=./out/eb21 \
REPO=whr778/gliner2-eb21 \
DEST=eb21 \
$PROBE_ENV \
bash tools/lambda/event_base_run.sh"

echo "[$EXP] model -> whr778/gliner2-eb21  logs -> eb21/  epochs -> whr778/gliner2-eb21-checkpoint-epoch-N"
# PIN THE CARD: arms on different silicon are not a matched A/B.
exec env NAME="eb21" JOB="$JOB" TYPES="${TYPES:-gpu_1x_a100_sxm4}" \
     bash tools/lambda/launch_when_available.sh
