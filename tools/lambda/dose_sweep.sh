#!/bin/bash
# DOSE SWEEP: the smallest negatives dose that keeps the rejection win.
#
# Full dose bought +0.0969 argument precision over base for +0.0031 recall; the control bought
# +0.0327 / +0.0257 (LABEL_NEGATIVES_PLAN 5d). Both the CONTROL and the BASE are already
# measured on this exact split, so these arms need neither -- three trainings, not five.
#
#   lo    entities 1                              -- is one entity negative enough?
#   half  entities 1, events 1                    -- add the event dimension back
#   wlo   full dose, abstention_loss_weight 0.1   -- same dose, half the gate pressure
#
# Each arm is trained, blind-tested, and run through the absent-type firing probe on cmnee
# (n=300, documents that FIT -- the A/B's own attempt ran on four because it was pointed at
# casie against a 900-character filter). Checkpoints are pushed: the last A/B's died with the
# box and cost a retrain to ask a follow-up question.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1

PY=./.venv/bin/python
OUT=$HOME/dose
DEST=${DEST:-negatives_dose}
ARMS=${ARMS:-"lo half wlo"}
mkdir -p "$OUT"
source tools/lambda/_publish.sh
RESCUE=0
trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

for arm in $ARMS; do
  echo "[dose] ===== $arm  $(date -u) ====="
  $PY -u tools/train/train.py --config "tools/train/config/ab/dose-$arm.yaml" 2>&1 \
    | tee "$OUT/$arm.log" | grep -aE "schema_dropout|label negatives ON|negative queries|composition\]   negatives|Blind test" | tail -12
  echo "[dose] $arm rc=${PIPESTATUS[0]}"
  cp "out/dose-$arm/test_metrics.json" "$OUT/$arm.json" 2>/dev/null || echo "[dose] NO METRICS $arm"
  publish "$DEST" "$OUT/$arm.json" "$OUT/$arm.log" || RESCUE=1
  CK=out/dose-$arm/best
  if [ -d "$CK" ]; then
    $PY -u tools/train/push_to_hub.py --checkpoint "$CK" \
        --repo-id "whr778/gliner2-dose-$arm" --private 2>&1 | tail -2 || RESCUE=1
  fi
done

echo "[dose] ===== absent-type firing, all arms, cmnee n=300  $(date -u) ====="
SPEC=""
for arm in $ARMS; do
  [ -d "out/dose-$arm/best" ] && SPEC="$SPEC $arm=out/dose-$arm/best"
done
if [ -n "$SPEC" ]; then
  $PY -u tools/train/probe_event_type_fp.py --checkpoints $SPEC \
      --test data/cmnee.test.jsonl --train data/cmnee.train.jsonl \
      --n 300 --threshold 0.3 --device cuda 2>&1 | tee "$OUT/firing.txt" | tail -8
  publish "$DEST" "$OUT/firing.txt" || RESCUE=1
fi

echo "[dose] ===== DONE $(date -u) ====="
HAVE=$(find "$OUT" -name '*.json' -o -name 'firing.txt' 2>/dev/null | wc -l)
if [ "$RESCUE" -ne 0 ] && [ "$HAVE" -gt 0 ]; then
  echo "[dose] *** publish failed with $HAVE artefact(s) -- holding for rescue ***"
  sleep infinity
fi
