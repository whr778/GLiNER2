#!/bin/bash
# eb22d BASE RUN -- one A100 SXM4 40 GB, the full 5-epoch schedule (tools/train/config/base/eb22d.yaml).
#
#   bash tools/lambda/eb22d_launch.sh
#
# eb22d = eb22c + the tier-1 fixes (seed 42, DDP averaging, by-name optimizer restore, epoch checkpoints, per-epoch
# metric_sweep) + boundary_marginal_loss bce + coref_link_distance mlp 64 (both DECIDED by the user 2026-10-24). Its
# model block equals base/eb22c-ddp-trace.yaml's except the coref MLP, so every eb22c control and arm compares at
# matched steps -- which is why this is NOT eb22c_launch.sh: that runner's H100 scaling phase picks a new per-GPU
# batch and would change the 82,930-step schedule.
# Card and speed are the measured ones: the eb22c trace arms ran on this card at ~1.6 s/step with evals (peak 25.7 GB);
# eb21's full-val sweep + blind test took 1.6 h here. Estimate ~48 h (~38 h training + ~6 h of per-epoch 5-point
# metric sweeps over the 19,286-record val + ~3 h end of run), ~$96 at $1.99/h.
# Stops: JOB_TIMEOUT 54 h on train.py, HARD_DEADLINE 58 h on the box (cap ~$115), terminate on the normal path.
# If the timeout fires the runner still pushes best/ and every epoch, then holds for the deadline (event_base_run.sh).
# Model first, then metrics, then (non-fatal) every checkpoint-epoch-N to whr778/gliner2-eb22d-checkpoint-epoch-N
# and the entity-coref readout on best/.
set -uo pipefail
EXP=eb22d
CFG=tools/train/config/base/eb22d.yaml

export BRANCH=$(git rev-parse --abbrev-ref HEAD)
export EXPECT_COMMIT=$(git rev-parse --short=12 HEAD)
git fetch -q origin "$BRANCH" && [ "$(git rev-parse --short=12 "origin/$BRANCH")" = "$EXPECT_COMMIT" ] \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- $EXPECT_COMMIT is not origin/$BRANCH; push first ***"; exit 5; }
[ -f "$CFG" ] || { echo "[$EXP] *** REFUSING TO LAUNCH -- $CFG missing ***"; exit 6; }
grep -q "boundary_marginal_loss: bce" "$CFG" && grep -q "coref_link_distance: mlp" "$CFG" \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- $CFG is not the decided recipe (bce + coref mlp) ***"; exit 8; }

uv run python tools/train/check_corpora_fetchable.py --config "$CFG" --offline \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }
uv run python tools/train/check_label_menus.py --config "$CFG" \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- classification menus disagree ***"; exit 4; }
bash tools/lambda/local_start_check.sh "$CFG" \
  || { echo "[$EXP] *** REFUSING TO LAUNCH -- the config cannot start train.py locally ***"; exit 7; }

export JOB_TIMEOUT=${JOB_TIMEOUT:-194400}     # train.py: 54 h
export HARD_DEADLINE=${HARD_DEADLINE:-208800} # whole box incl. provisioning: 58 h

JOB="CFG=$CFG \
OUTDIR=./out/eb22d \
REPO=whr778/gliner2-eb22d \
DEST=eb22d \
PROBE=0 COREF_READOUT=1 PUSH_EPOCHS=1 \
bash tools/lambda/event_base_run.sh"

echo "[$EXP] commit $EXPECT_COMMIT | model -> whr778/gliner2-eb22d | logs -> eb22d/ | epochs -> whr778/gliner2-eb22d-checkpoint-epoch-N"
exec env NAME="eb22d" JOB="$JOB" TYPES="${TYPES:-gpu_1x_a100_sxm4}" \
     bash tools/lambda/launch_when_available.sh
