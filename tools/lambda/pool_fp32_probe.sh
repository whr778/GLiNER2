#!/bin/bash
# Is the shared-pool NaN a PRECISION fault or a PATH fault? Then, only if precision, run the
# A/B that has now failed to produce a number twice.
#
# ESTABLISHED, so this does not re-measure it: with bf16 on, `candidate_pool: shared` gives a
# finite loss at step 0 and a NaN BACKWARD at step 1 -- grad norm nan, 225 parameters poisoned
# including the entire encoder. `per_query` is clean at every step. Four steps on CPU stay
# finite, so the variable is CUDA plus bf16 autocast, not step count.
#
# STAGE 1 IS A GATE ON STAGE 2, and that is the whole design. Running the training arms first
# would burn an hour rediscovering a NaN that takes ninety seconds to provoke. So: probe in
# both precisions, and only train if fp32 is clean.
#
#   fp32 clean  -> autocast in the shared pool's backward. Train the fp32 arms; a positive
#                  result then justifies a targeted autocast(enabled=False) and a bf16 rerun.
#   fp32 also NaN -> the shared pool has a real backward bug independent of precision. STOP.
#                  There is nothing to A/B, and the finding is the bug.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1

PY=./.venv/bin/python
OUT=$HOME/fp32_probe
DEST=${DEST:-pool_fp32_probe}
mkdir -p "$OUT"
source tools/lambda/_publish.sh
RESCUE=0
trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

# --- STAGE 1: the precision question, ~2 minutes --------------------------------------
echo "[fp32] ===== PROBE $(date -u) ====="
$PY -u tools/train/debug_shared_pool_nan.py --pool shared --no-bf16 --steps 5 \
    2>&1 | tee "$OUT/probe-fp32.log"
$PY -u tools/train/debug_shared_pool_nan.py --pool shared --bf16 --steps 5 \
    2>&1 | tee "$OUT/probe-bf16.log"
publish "$DEST" "$OUT/probe-fp32.log" "$OUT/probe-bf16.log" || RESCUE=1

# A clean fp32 probe is: no parameter ever reports a non-finite gradient. Grep for the
# opposite, because "0 parameter(s) with non-finite grad" contains the words either way --
# the count is what matters, and only a NON-zero count is a failure.
BAD=$(grep -c "with non-finite grad" "$OUT/probe-fp32.log" 2>/dev/null || echo 0)
CLEAN=$(grep -c "0 parameter(s) with non-finite grad" "$OUT/probe-fp32.log" 2>/dev/null || echo 0)
echo "[fp32] probe lines: $BAD backward report(s), $CLEAN of them clean"

if [ "$BAD" -eq 0 ] || [ "$BAD" -ne "$CLEAN" ]; then
  echo "[fp32] *** fp32 IS NOT CLEAN (or the probe never ran) -- the shared pool has a"
  echo "[fp32] *** backward fault independent of precision. NOT training: there is nothing"
  echo "[fp32] *** to A/B, and the bug is the finding."
  grep -E "=== step|non-finite grad|MODULE" "$OUT/probe-fp32.log" | head -20
  exit 0
fi

echo "[fp32] fp32 is CLEAN across all steps -- the fault is bf16 autocast. Training the arms."

# --- STAGE 2: the A/B, in fp32 --------------------------------------------------------
for arm in shared-fp32 shared-long-fp32; do
  echo "[fp32] ===== $arm  $(date -u) ====="
  $PY -u tools/train/train.py --config "tools/train/config/ab/pool-$arm.yaml" \
      2>&1 | tee "$OUT/$arm.log"
  echo "[fp32] $arm rc=${PIPESTATUS[0]}"
  GRAD=$(grep -o "shared-pool grad norm [^ ]*" "$OUT/$arm.log" | tail -1 | awk '{print $NF}')
  echo "[fp32] GATE $arm: shared-pool grad norm = ${GRAD:-<ABSENT>}"
  cp "out/pool-$arm/test_metrics.json" "$OUT/$arm.json" 2>/dev/null \
    || echo "[fp32] NO METRICS for $arm"
  publish "$DEST" "$OUT/$arm.json" "$OUT/$arm.log" || RESCUE=1
  case "${GRAD:-0}" in
    ""|0|0.0|0.000e+00|0.000000e+00)
      echo "[fp32] *** treatment did not apply ($arm) -- not starting another arm"; RESCUE=1; break;;
    *nan*|*inf*|*NaN*|*Inf*)
      echo "[fp32] *** non-finite gradient in fp32 too ($arm): $GRAD"; RESCUE=1; break;;
  esac
done

if [ "$RESCUE" -ne 0 ]; then
  echo "[fp32] *** holding for rescue; the watchdog still terminates this instance ***"
  sleep infinity
fi
echo "[fp32] DONE $(date -u)"
