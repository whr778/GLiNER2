#!/bin/bash
# CANDIDATE-POOL A/B, second attempt. Runs ONLY the two `shared` arms.
#
# WHY THE CONTROL IS NOT RE-RUN. Attempt one (2026-09-17) produced a complete `perquery`
# arm and two treatment arms that died in under a minute, refused by a startup guard that
# wrongly listed `candidate_pool` as structural. The control is sound and is reused; the
# commit that freed the flag touched the joint-decode branch, a probe, tests and this
# gate, and these configs decode GREEDY, so the control's code path is unchanged. Stated
# here because a comparison across two code states has to be auditable, not assumed.
#
# WHY IT IS BEING ASKED AT ALL: the entity-typed-argument sweep found that merely ADDING an
# entity menu costs 30-35% of argument recall, which is what `per_query` predicts when
# entity and event queries compete for candidate budget. `shared` builds one document pool
# instead, and is the direct remedy.
#
# THE GATE IS ENFORCED HERE, NOT LEFT FOR A HUMAN TO NOTICE. `shared_pool_builder` exists in
# every checkpoint and receives NO GRADIENT under per_query, so an arm that failed to switch
# looks exactly like an arm that switched and did nothing. Each arm must print
#   [pool] candidate_pool=shared  shared-pool grad norm <non-zero>
# and this script ABORTS THE RUN if it does not -- including before spending on arm two.
# Measured locally: per_query gives exactly 0.000e+00, shared gives 8.890e+00.
#
# READING ORDER, decided in advance so it cannot be chosen to suit the result: `shared` is
# matched-step and starts from an untrained pool, so a NULL there is uninformative and only
# a POSITIVE counts. `shared-long` doubles the samples and is the arm that can produce a
# readable null.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1   # FA2 is a CORRECTNESS requirement on mmBERT in bf16

PY=./.venv/bin/python
OUT=$HOME/pool_ab
DEST=${DEST:-pool_ab}
ARMS=${ARMS:-"shared shared-long"}
RESCUE=0
mkdir -p "$OUT"
source tools/lambda/_publish.sh

# The orchestration log on EVERY exit path, not just the happy one.
trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

for arm in $ARMS; do
  echo "[pool] ===== $arm  $(date -u) ====="
  $PY -u tools/train/train.py --config "tools/train/config/ab/pool-$arm.yaml" \
      2>&1 | tee "$OUT/$arm.log"
  echo "[pool] $arm rc=${PIPESTATUS[0]}"

  # THE GATE. Read it from the arm's own log, before anything else is believed.
  # `[0-9.e+-]*` DID NOT MATCH `nan`, so the capture returned the literal word `norm` and
  # the zero-check below could not fire. Measured on the first run: the gate printed
  # `grad norm = norm` while the real value was nan. Capture whatever the field holds.
  GRAD=$(grep -o "shared-pool grad norm [^ ]*" "$OUT/$arm.log" | tail -1 | awk '{print $NF}')
  echo "[pool] GATE $arm: shared-pool grad norm = ${GRAD:-<ABSENT>}"
  cp "out/pool-$arm/test_metrics.json" "$OUT/$arm.json" 2>/dev/null \
    || echo "[pool] NO METRICS for $arm"
  publish "$DEST" "$OUT/$arm.json" "$OUT/$arm.log" || RESCUE=1

  # `0`, `0.000e+00` and an absent line all mean the same thing: the treatment did not
  # apply. Refuse to spend on the next arm rather than produce a second unreadable one.
  case "${GRAD:-0}" in
    ""|0|0.0|0.000e+00|0.000000e+00)
      echo "[pool] *** THE TREATMENT DID NOT APPLY ($arm) -- shared pool carries no"
      echo "[pool] *** gradient. This arm is a duplicate of the control and the A/B is"
      echo "[pool] *** void. Not starting any further arm."
      RESCUE=1
      break;;
    *nan*|*inf*|*NaN*|*Inf*)
      # A NON-FINITE norm is not a null and not a small effect -- it is a broken arm, and
      # it is what BOTH arms produced on 2026-09-17. Stop for the same reason as zero.
      echo "[pool] *** NON-FINITE SHARED-POOL GRADIENT ($arm): $GRAD"
      echo "[pool] *** The shared path is diverging, not underperforming. Not starting"
      echo "[pool] *** any further arm -- there is nothing to compare."
      RESCUE=1
      break;;
  esac
  echo "[pool] ===== $arm done $(date -u) ====="
done

if [ "$RESCUE" -ne 0 ]; then
  echo "[pool] *** HOLDING THE BOX -- a publish failed or an arm was void ***"
  echo "[pool] artefacts: $OUT ; checkpoints: ~/gliner2/out/pool-*/best"
  echo "[pool] the hard-deadline watchdog still terminates this instance."
  sleep infinity
fi
echo "[pool] ALL ARMS DONE $(date -u)"
