#!/bin/bash
# THE EVENT-CAPABLE BASE: eb16's data with events on the RECORD head.
# ~14h on one A100, ~$28. Nothing here depends on the operator's laptop.
#
# SUPERSEDED 2026-09-15: these figures are of uncertain provenance and are NOT at
# threshold 0.5 -- the model card says `Decision threshold: 0.3`. Measured at the
# val-selected 0.2: event_argument strict 0.0991 / relaxed 0.5884, event_trigger
# 0.5984, event_type 0.8650. See EVENT_ARGUMENT_DIAGNOSIS.md section 4c.
# WHAT THIS RUN IS FOR. `event_argument` reads 0.1178 STRICT against 0.5783 RELAXED on the
# same predictions -- the model finds the arguments and cannot BIND them, because the
# mention path compiles ONE instance per event type and 64.2% of gold event instances share
# their type with another instance in the same document. `event_records: true` routes events
# through the record head, which is multi-instance by construction.
# (EVENT_ARGUMENT_DIAGNOSIS.md.)
#
# SURVIVAL PROPERTIES, in the order they matter for a 14-hour run:
#
#   1. THE JOB IS NOT ATTACHED TO THE LAPTOP. It runs under tmux, started by box_run.sh.
#      ssh dropping, the network going, or the laptop rebooting are invisible to it.
#   2. THE MODEL IS PUSHED FIRST, before metrics. For a base train the CHECKPOINT is the
#      irreplaceable artefact -- metrics can be recomputed from it, and it cannot be
#      recomputed from them. This inverts the order used for the cheap A/B runs, and the
#      inversion is deliberate.
#   3. THE MODEL IS PUSHED EVEN IF THE BLIND TEST DIES. train.py trains and then evaluates;
#      a crash in eval must not cost 14 hours of GPU. The push is keyed on `best/` existing,
#      not on the exit code.
#   4. EVERY PUSH RETRIES 6x over ~30 min and VERIFIES against the Hub's own file list.
#      A clean return is not proof: upload_folder has returned successfully having written
#      nothing, and that cost ~15h of A100 once.
#   5. A FAILED PUSH HOLDS THE BOX for the watchdog window instead of terminating into a
#      loss. Losing a $28 model to save $2 of idle is bad arithmetic.
#   6. THE BOX TERMINATES ITSELF on the normal path, and a detached hard-deadline watchdog
#      terminates it regardless. Three independent stops, per box_run.sh.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1   # sdpa on bf16 mmBERT is a CORRECTNESS failure, not a slowdown

PY=./.venv/bin/python
# Parameterised so the negatives rebuild reuses this runner rather than forking it --
# forking the publish block once already shipped a run with no publishing at all.
CFG=${CFG:-tools/train/config/base/eb16-eventrecords-tr.yaml}
OUTDIR=${OUTDIR:-./out/eb16-eventrecords-tr}
REPO=${REPO:-whr778/gliner2-eb16-eventrecords-tr}
DEST=${DEST:-event_base}
LOG=$HOME/event_base.log
source tools/lambda/_publish.sh
RESCUE=0

echo "[base] ===== START $(date -u) ====="
echo "[base] config $CFG -> $REPO"

# PROVENANCE, PUBLISHED AT MINUTE ZERO. Establishing whether a finished model predated the
# bf16 consistency-loss fix had to be done by arithmetic on push timestamps, because nothing
# recorded what code built it. Write it down before the run can fail.
{
  echo "arm_dest      $DEST"
  echo "config        $CFG"
  echo "repo          $REPO"
  echo "commit        $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "branch        $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "started_utc   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "instance      $(cat ~/.instance_id 2>/dev/null || echo unknown)"
  echo "gpu           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
  echo "torch         $($PY -c 'import torch;print(torch.__version__)' 2>/dev/null)"
} > "$HOME/START.txt"
publish "$DEST" "$HOME/START.txt" "$CFG" || echo "[base] start marker did not publish (continuing)"

# A heartbeat so a 14-hour run is legible from outside without attaching to tmux.
( while sleep 900; do
    echo "[hb] $(date -u) gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' ') $(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -aoE '[0-9]+/[0-9]+ \[[^]]*\]' | tail -1)"
  done ) >> "$HOME/heartbeat.log" 2>&1 &
disown

# PERIODIC PUBLISH, so a 16-hour run is not a 16-hour hole. Nothing used to reach the Hub
# until the very end, which means a network outage or a dead box at hour 15 took the whole
# log with it -- exactly what happened on 2026-09-17. Every 30 minutes, ship the heartbeat
# and the tail of the training log. Failures here are IGNORED on purpose: this is a
# convenience stream, and it must never be able to fail the run it is reporting on.
( while sleep 1800; do
    tail -c 200000 "$LOG" > "$HOME/progress.log" 2>/dev/null
    publish "$DEST" "$HOME/heartbeat.log" "$HOME/progress.log" >/dev/null 2>&1 || true
  done ) >> "$HOME/publisher.log" 2>&1 &
disown

timeout 72000 $PY -u tools/train/train.py --config "$CFG" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[base] train.py rc=$rc  $(date -u)"

# 1. THE MODEL, FIRST AND REGARDLESS OF rc.
if [ -d "$OUTDIR/best" ]; then
  echo "[base] pushing checkpoint (this is the irreplaceable artefact)"
  n=0
  until $PY -u tools/train/push_to_hub.py --checkpoint "$OUTDIR/best" \
        --repo-id "$REPO" --private 2>&1 | tail -4; do
    n=$((n+1)); echo "[base] model push FAILED (attempt $n/6)"
    [ "$n" -ge 6 ] && { echo "[base] *** MODEL PUSH FAILED AFTER 6 ATTEMPTS ***"; RESCUE=1; break; }
    sleep $((n * 120))
  done
else
  echo "[base] *** NO CHECKPOINT AT $OUTDIR/best -- nothing to push ***"
  RESCUE=1
fi

# 1b. THE FINAL-EPOCH CHECKPOINT, to a SEPARATE repo.
#
# `best/` is whichever epoch won on `metric_for_best`, and two arms of one A/B can win on
# DIFFERENT epochs: roles2 shipped its control at epoch 2 against its treatment at epoch 5,
# so every head rose and the run was void. The trainer already writes `final/` and
# `checkpoint-epoch-N/`; the box then terminates and destroys them, which is why that run
# could not be rescued by re-scoring and needs the control retrained (~$20).
#
# Pushing `final/` makes a like-for-like comparison possible AFTER THE FACT, whatever
# selection did. Non-fatal by design: this is the diagnostic copy, and it must never cost
# the run its irreplaceable `best/`.
if [ -d "$OUTDIR/final" ]; then
  echo "[base] pushing the final-epoch checkpoint (diagnostic copy, non-fatal)"
  $PY -u tools/train/push_to_hub.py --checkpoint "$OUTDIR/final" \
      --repo-id "$REPO-final" --private \
      --commit-message "final-epoch checkpoint, for like-for-like comparison" 2>&1 | tail -3 \
    || echo "[base] final-epoch push failed; continuing (best/ is what matters)"
else
  echo "[base] no $OUTDIR/final to push"
fi

# 2. METRICS AND LOGS, whatever happened to the model.
cp "$OUTDIR/test_metrics.json" "$HOME/test_metrics.json" 2>/dev/null || echo "[base] no test_metrics.json"
# eval_metrics.json is OPTIONAL -- train.py writes test_metrics.json and val_metrics.json,
# never this name. Copying it with `|| true` and then NAMING IT IN THE PUBLISH LIST made the
# publish correctly report a never-written artefact and hold the box, after a 16-hour run
# whose model and metrics had both already landed. An optional file must not appear in a
# required list: only pass it when it exists.
EXTRA=""
cp "$OUTDIR/eval_metrics.json" "$HOME/eval_metrics.json" 2>/dev/null \
  && EXTRA="$HOME/eval_metrics.json"
publish "$DEST" "$HOME/test_metrics.json" $EXTRA "$LOG" \
        "$HOME/heartbeat.log" "$CFG" || RESCUE=1

echo "[base] ===== DONE $(date -u) rc=$rc ====="
if [ "$RESCUE" -ne 0 ]; then
  echo "[base] *** SOMETHING DID NOT PUBLISH -- holding the box for rescue ***"
  echo "[base] checkpoint: $OUTDIR/best   logs: $LOG"
  echo "[base] the hard-deadline watchdog still terminates this instance."
  sleep infinity
fi
