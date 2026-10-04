#!/bin/bash
# FAST A/B -- launch one arm of any fast A/B family (configs tools/train/config/ab/<EXP>-<ARM>.yaml).
#
#   EXP=p3arg ARM=control bash tools/lambda/fast_ab_launch.sh    # control also probes eb18 (the baseline)
#   EXP=p3arg ARM=hard    bash tools/lambda/fast_ab_launch.sh    # one box per arm, in parallel
#
# Each family's QUESTION, arms and proof lines live in its config headers. Shared design: one
# random draw of eb18's mix for every arm (max_samples_seed), one pass, capped validation, a FIXED
# threshold. After training each box runs the trigger-miss probe, non-fatal, beside the metrics.
#
# Each box is autonomous once launched (event_base_run.sh): model pushed first, metrics and logs
# published with retries and verified against the Hub, then terminate. Three stops: JOB_TIMEOUT
# on train.py, the detached hard-deadline watchdog, and terminate on the normal path.
set -uo pipefail
EXP=${EXP:?EXP=<family>, e.g. p3arg}
ARM=${ARM:?ARM=<arm>, e.g. control}
CFG=tools/train/config/ab/$EXP-$ARM.yaml

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

# Measured on eb18 (A100): 16.7 chunks/s, so 30k chunks ~30 min; capped val ~2 min; blind test
# ~30-40 min. JOB_TIMEOUT bounds the WHOLE job in box_run.sh (train, publish, then the probe:
# up to 2 x (30 + 90) min on the control box, which also probes eb18) -- event_base_run.sh reuses
# it for train.py alone. The probe runs AFTER the metrics publish, so only it can be cut short.
export JOB_TIMEOUT=${JOB_TIMEOUT:-21600}     # whole job: 6h
export HARD_DEADLINE=${HARD_DEADLINE:-25200} # whole box incl. provisioning: 7h

PROBE_ENV="PROBE=1"
[ "$ARM" = "control" ] && PROBE_ENV="PROBE=1 PROBE_BASE=1 PROBE_BASE_CKPT=whr778/gliner2-eb18-balanced"

JOB="CFG=$CFG \
OUTDIR=./out/$EXP-$ARM \
REPO=whr778/gliner2-$EXP-$ARM \
DEST=${EXP}_$ARM \
$PROBE_ENV \
bash tools/lambda/event_base_run.sh"

echo "[$EXP] arm=$ARM  model -> whr778/gliner2-$EXP-$ARM  logs -> ${EXP}_$ARM/"
# PIN THE CARD: arms on different silicon are not a matched A/B.
exec env NAME="$EXP-$ARM" JOB="$JOB" TYPES="${TYPES:-gpu_1x_a100_sxm4}" \
     bash tools/lambda/launch_when_available.sh
