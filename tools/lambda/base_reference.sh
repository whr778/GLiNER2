#!/bin/bash
# THE MISSING REFERENCE: the warm-start BASE, scored on the A/B's OWN split.
#
# Every number in the negatives A/B is a fine-tuned model measured against another fine-tuned
# model. Neither was ever compared to the point they BOTH started from on the same data -- the
# base's published figures are on ITS test set (20,602 records), the arms' on cmnee+casie.
# Different denominators, so "control 0.3517" could be above or below where it began and
# nothing on file says which.
#
# Without this, a decline read as "the treatment costs recall" may be "the fine-tune costs
# recall and the treatment costs a little more", which is a different decision. It is one eval
# pass, and it makes section 5c interpretable.
#
# Scored at BOTH thresholds the A/B used, and with --full-menu, so it lines up with every
# figure already recorded.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1

PY=./.venv/bin/python
OUT=$HOME/baseref
DEST=${DEST:-negatives_base_reference}
CFG=tools/train/config/ab/negatives2-control.yaml   # the A/B's own data + eval settings
CK=$HOME/ckpt/eb16-eventrecords-tr
mkdir -p "$OUT"
source tools/lambda/_publish.sh
RESCUE=0
trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

[ -f "$CK/model.safetensors" ] || $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('whr778/gliner2-eb16-eventrecords-tr', local_dir='$CK')" >/dev/null 2>&1

for split in val test; do
  echo "[baseref] ===== BASE on the A/B $split split  $(date -u) ====="
  $PY -u tools/train/eval.py --config "$CFG" --checkpoint "$CK" \
      --split "$split" --threshold 0.3 --full-menu 2>&1 | tee "$OUT/base-$split.log" | tail -12
  cp "out/negatives2-control/${split}_metrics.json" "$OUT/base-$split.json" 2>/dev/null \
    || echo "[baseref] NO metrics for $split"
  publish "$DEST" "$OUT/base-$split.json" "$OUT/base-$split.log" || RESCUE=1
done

echo "[baseref] ===== DONE $(date -u) ====="
HAVE=$(find "$OUT" -name '*.json' 2>/dev/null | wc -l)
if [ "$RESCUE" -ne 0 ] && [ "$HAVE" -gt 0 ]; then
  echo "[baseref] *** publish failed with $HAVE artefact(s) -- holding for rescue ***"
  sleep infinity
fi
