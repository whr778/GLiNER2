#!/bin/bash
# GPU end-to-end test of tools/derive (stages 1-6) on CASIE val against three bases that
# differ structurally: eb18 (boundary + event_records + label_map), gliner2.5-multi-v1
# (boundary, no label inventory, mDeBERTa) and gliner2-base-v1 (legacy SPAN: stages 4 and
# the record sweep must be skipped, not crash). Every base publishes its calibration JSON
# and log whatever the exit code; a base with no JSON is a FAILED base and says so.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1

PY=./.venv/bin/python
DEST=${DEST:-derive_gpu_test}
OUT=$HOME/derived
mkdir -p "$OUT"
source tools/lambda/_publish.sh
FAILED=0

for model in whr778/gliner2-eb18-balanced fastino/gliner2.5-multi-v1 fastino/gliner2-base-v1; do
  name=casie__${model##*/}
  echo "[derive-test] ===== $model  $(date -u) ====="
  timeout 3000 $PY -u tools/derive/derive_config.py --corpus data/casie --model "$model" \
      --out "$OUT" --gpu --device cuda 2>&1 | tee "$OUT/$name.log" | grep -aE "^\[derive\]|Traceback|Error" | tail -8
  if [ -f "$OUT/$name.calibration.json" ]; then
    publish "$DEST" "$OUT/$name.calibration.json" "$OUT/$name.labels_review.md" "$OUT/$name.log" || FAILED=1
  else
    echo "[derive-test] *** $model wrote NO calibration -- FAILED ***"
    publish "$DEST" "$OUT/$name.log" || true
    FAILED=1
  fi
done
echo "[derive-test] ===== DONE failed=$FAILED $(date -u) ====="
exit $FAILED
