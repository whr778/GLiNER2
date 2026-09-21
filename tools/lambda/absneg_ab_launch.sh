#!/bin/bash
# ABSENT-NEGATIVES A/B -- launch one arm. Run twice, once per arm.
#
#   ARM=control   bash tools/lambda/absneg_ab_launch.sh
#   ARM=treatment bash tools/lambda/absneg_ab_launch.sh
#
# THE ONE VARIABLE is `boundary_head.absent_negatives_in_denominator`. Both arms inject label
# negatives identically, so the delta is attributable to the RANKING CHANNEL rather than to
# the negatives themselves -- "negatives help" was already measured and is not the question.
#
# WHAT IT TESTS. `proposal_listwise_loss` is multiple-negatives ranking and it SKIPS any query
# with no gold. An injected label negative IS a label mapped to an empty list, so negatives
# contributed EXACTLY ZERO to both listwise losses -- 0.6 of combined weight. That predicts
# the signature the negatives arm measured: precision up, recall down, F1 flat.
#
# THE RUN PROVES THE TREATMENT APPLIED. `absent_negatives_used` is logged every step. Zero on
# the treatment arm means absent labels never reached the denominator and the result is void.
#
# SURVIVING THIS END GOING AWAY. Each box is autonomous once launched: clone, train, push the
# model, publish metrics and logs, terminate. The laptop can reboot, lose the network, or be
# swapped for a phone hotspot with no effect. On the box, `publish` retries 6x over ~30 min
# and verifies against the Hub's file list; START.txt with the CODE COMMIT goes up at minute
# zero; heartbeat and a log tail every 30 min. Three stops: job timeout, detached watchdog,
# terminate on the normal path.
set -uo pipefail
ARM=${ARM:?ARM=control or ARM=treatment}
# `roles` is absneg4's single arm: absent pooling scoped to event ROLES, compared against
# BOTH absneg2 arms, which are reusable because the operating point and test set match.
case "$ARM" in control|treatment|roles) ;;
  *) echo "ARM must be control, treatment or roles"; exit 2;; esac
# EXP selects the config family: `absneg` is the first pair (confounded by checkpoint
# selection), `absneg2` the re-run that selects on a task metric. Parameterised rather than
# copied -- three bugs in one day came from a second copy drifting from the first.
EXP=${EXP:-absneg}

uv run python tools/train/check_corpora_fetchable.py \
    --config "tools/train/config/ab/$EXP-$ARM.yaml" --offline \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }

export JOB_TIMEOUT=${JOB_TIMEOUT:-82800}
export HARD_DEADLINE=${HARD_DEADLINE:-90000}

JOB="CFG=tools/train/config/ab/$EXP-$ARM.yaml \
OUTDIR=./out/$EXP-$ARM \
REPO=whr778/gliner2-$EXP-$ARM \
DEST=${EXP}_$ARM \
bash tools/lambda/event_base_run.sh"

echo "[$EXP] arm=$ARM  model -> whr778/gliner2-$EXP-$ARM  logs -> ${EXP}_$ARM/"
# PIN THE CARD. launch_when_available.sh falls back to gpu_1x_a10 when the A100 pool is
# empty, and on 2026-09-19 that put the control on an A100 and the treatment on an A10 --
# two arms on different silicon is not a matched A/B, whatever the loss does. Wait for the
# right card instead of silently accepting a different one.
exec env NAME="$EXP-$ARM" JOB="$JOB" TYPES="${TYPES:-gpu_1x_a100_sxm4}" \
     bash tools/lambda/launch_when_available.sh
