#!/bin/bash
# eb23a BASE RUN -- full 5-epoch schedule (tools/train/config/base/eb23a.yaml).
#
#   bash tools/lambda/eb23a_launch.sh
#
# Auto-scales batch_size and gradient_accumulation_steps in event_base_run.sh to preserve effective batch = 16
# (82,930 steps, warmup 4,146):
#   - 1x H100 80 GB: batch_size=8, grad_accum=2, eval_batch_size=4
#   - 1x A100 40 GB: batch_size=4, grad_accum=4, eval_batch_size=2
#   - 2x H100 80 GB: batch_size=8, grad_accum=1, eval_batch_size=4
#   - 4x H100 80 GB: batch_size=4, grad_accum=1, eval_batch_size=4
# Stops: JOB_TIMEOUT 54 h on train.py, HARD_DEADLINE 58 h on the box, terminate on the normal path.
# Publishes metrics before the model, uploads each completed checkpoint-epoch-N (with training_state.pt)
# during training and at the end, and pushes best/ to whr778/gliner2-eb23a.
set -uo pipefail
EXP=eb23a
CFG=tools/train/config/base/eb23a.yaml

export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
[ -f "$CFG" ] || { echo "[$EXP] *** REFUSING TO LAUNCH -- $CFG missing ***"; exit 6; }
grep -q "pair_split_reduction: true" "$CFG" \
  && grep -q "hard_negative_keep_all_when_absent: false" "$CFG" \
  && grep -q "remove_events_prob: 0.0" "$CFG" \
  && grep -q "record_drop_anchor_field_loss: true" "$CFG" \
  && grep -q "strip_bare_space_subword: true" "$CFG" \
  && grep -q "record_field_max_tokens: 48" "$CFG" \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- $CFG is missing eb23a recipe keys ***"; exit 8; }

uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }
uv run python tools/train/check_label_menus.py --config "$CFG" \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- classification menus disagree ***"; exit 4; }
if [ -n "${SKIP_START_CHECK:-}" ]; then
  echo "[$EXP] local start check SKIPPED: $SKIP_START_CHECK"
else
  bash tools/lambda/local_start_check.sh "$CFG" \
    || { echo "[$EXP] *** REFUSING TO LAUNCH -- the config cannot start train.py locally ***"; exit 7; }
fi

export JOB_TIMEOUT=${JOB_TIMEOUT:-194400}     # train.py: 54 h
export HARD_DEADLINE=${HARD_DEADLINE:-208800} # whole box incl. provisioning: 58 h

JOB="CFG=$CFG \
OUTDIR=./out/eb23a \
REPO=whr778/gliner2-eb23a \
DEST=eb23a \
PROBE=0 COREF_READOUT=1 PUSH_EPOCHS=1 \
bash tools/lambda/event_base_run.sh"

echo "[$EXP] commit $EXPECT_COMMIT | model -> whr778/gliner2-eb23a | logs -> eb23a/ | epochs -> whr778/gliner2-eb23a/checkpoint-epoch-N"
exec env NAME="eb23a" JOB="$JOB" TYPES="${TYPES:-gpu_1x_h100_sxm5 gpu_1x_h100_pcie gpu_1x_a100_sxm4}" \
     bash tools/lambda/launch_when_available.sh
