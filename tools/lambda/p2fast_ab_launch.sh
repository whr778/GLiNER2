#!/bin/bash
# PHASE 2 FAST A/B -- launch one arm. Run four times, once per arm (four boxes in parallel).
#
#   ARM=control  bash tools/lambda/p2fast_ab_launch.sh     # also probes eb18 (the baseline)
#   ARM=control2 bash tools/lambda/p2fast_ab_launch.sh
#   ARM=identity bash tools/lambda/p2fast_ab_launch.sh
#   ARM=idsep    bash tools/lambda/p2fast_ab_launch.sh
#
# THE QUESTION: does labelling proposal-loss gold by span identity (`proposal_gold: identity`)
# change trigger recall/precision? control vs control2 is the noise floor; idsep adds
# `absent_reduction: separate` to see whether the two stack. Configs: tools/train/config/ab/
# p2fast-*.yaml, one 30k-record RANDOM draw of eb18's mix shared by every arm, one pass, fixed
# threshold 0.3. Paper: tools/events_working_papers/LIFE_OF_A_SAMPLE.md section 2.7.
#
# PROOF PER ARM, from inside the run: the `proposal_gold=<mode>` and `boundary absent_reduction=
# <mode>` log lines. After training each box runs the trigger-miss probe (never proposed / below
# the gate / wrong type), non-fatal, published beside the metrics.
#
# Each box is autonomous once launched (event_base_run.sh): model pushed first, metrics and logs
# published with retries and verified against the Hub, then terminate. Three stops: JOB_TIMEOUT
# on train.py, the detached hard-deadline watchdog, and terminate on the normal path.
set -uo pipefail
ARM=${ARM:?ARM=control, control2, identity or idsep}
case "$ARM" in control|control2|identity|idsep) ;;
  *) echo "ARM must be control, control2, identity or idsep"; exit 2;; esac
EXP=p2fast
CFG=tools/train/config/ab/$EXP-$ARM.yaml

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
