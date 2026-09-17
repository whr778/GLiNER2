#!/bin/bash
# Localize the CUDA-side non-finite that kills `candidate_pool: shared` at micro-batch 1.
#
# ~15 minutes of A10. Runs BOTH arms: `per_query` is the control that proves the harness is
# clean, so a finding in `shared` cannot be blamed on the instrument. Every CPU-reachable
# variable was already ruled out and stayed finite (see EVENT_ARGUMENT_DIAGNOSIS.md option 1),
# which is why this needs a GPU at all.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1

PY=./.venv/bin/python
OUT=$HOME/nan_debug
DEST=${DEST:-pool_nan_debug}
mkdir -p "$OUT"
source tools/lambda/_publish.sh
RESCUE=0
trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

for pool in per_query shared; do
  echo "[nan] ===== $pool  $(date -u) ====="
  $PY -u tools/train/debug_shared_pool_nan.py --pool "$pool" --steps 3 --batch 4 \
      2>&1 | tee "$OUT/$pool.log"
  echo "[nan] $pool rc=${PIPESTATUS[0]}"
  publish "$DEST" "$OUT/$pool.log" || RESCUE=1
done

echo "[nan] ===== SUMMARY ====="
grep -h "MODULE\|non-finite\|loss=" "$OUT"/*.log | head -40

if [ "$RESCUE" -ne 0 ]; then
  echo "[nan] *** a publish failed -- holding for rescue; the watchdog still terminates ***"
  sleep infinity
fi
echo "[nan] DONE $(date -u)"
