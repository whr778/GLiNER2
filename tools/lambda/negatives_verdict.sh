#!/bin/bash
# The two questions the A/B left open, on one box that terminates itself.
#
# 1. DOES THE TREATMENT ACTUALLY REJECT? The A/B's acceptance metric ran on FOUR documents --
#    it was pointed at casie, whose median document is 2,323 characters, against a probe
#    filtering at 900. 103 of 107 were silently dropped and the result (12-14 predictions,
#    differences of one) was noise. cmnee's median is 319 and ALL 2,724 event documents fit.
#
# 2. IS THE RECALL LOSS CALIBRATION RATHER THAN TRAINING? The treatment now has a TRAINED
#    abstention gate and `abstention_threshold` ships at 0.5. A gate that learned to fire may
#    simply be at the wrong operating point -- the same shape as the stage-0 gate that ran its
#    whole life at 0.5 and needed 0.998. If a threshold recovers the recall while keeping the
#    precision gain, the trade is a knob rather than a cost.
#
# Neither needs retraining: both arms are on the Hub.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1

PY=./.venv/bin/python
OUT=$HOME/verdict
DEST=${DEST:-negatives_verdict}
mkdir -p "$OUT"
source tools/lambda/_publish.sh
RESCUE=0
trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

for m in eb16-eventrecords-tr negatives2-control negatives2-treatment; do
  CK=$HOME/ckpt/$m
  [ -f "$CK/model.safetensors" ] || $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('whr778/gliner2-$m', local_dir='$CK')" >/dev/null 2>&1
done

# --- 1. REJECTION, on a corpus whose documents actually fit -----------------------------
echo "[verdict] ===== absent-type firing, cmnee, n=300  $(date -u) ====="
$PY -u tools/train/probe_event_type_fp.py \
    --checkpoints "base=$HOME/ckpt/eb16-eventrecords-tr" \
                  "control=$HOME/ckpt/negatives2-control" \
                  "treatment=$HOME/ckpt/negatives2-treatment" \
    --test data/cmnee.test.jsonl --train data/cmnee.train.jsonl \
    --n 300 --threshold 0.3 --device cuda 2>&1 | tee "$OUT/firing.txt" | tail -8
publish "$DEST" "$OUT/firing.txt" || RESCUE=1

# --- 2. IS IT CALIBRATION? sweep the abstention gate on the TREATMENT --------------------
# Pick on VALIDATION, score test ONCE -- sweeping on test and quoting the best is fitting it,
# and this programme has retracted a finding for that shape.
CFG=tools/train/config/ab/negatives2-treatment.yaml
for t in 0.3 0.5 0.7 0.9; do
  echo "[verdict] === VAL abstention_threshold=$t  $(date -u) ==="
  $PY -u tools/train/eval.py --config "$CFG" --checkpoint "$HOME/ckpt/negatives2-treatment" \
      --split val --threshold 0.3 --abstention-threshold "$t" \
      2>&1 | tee "$OUT/abst-$t.log" | tail -2
  cp out/negatives2-treatment/val_metrics.json "$OUT/abst-$t.json" 2>/dev/null \
    || echo "[verdict] NO metrics at $t"
  rm -f out/negatives2-treatment/val_metrics.json
  publish "$DEST" "$OUT/abst-$t.json" "$OUT/abst-$t.log" || RESCUE=1
done

echo "[verdict] ===== DONE $(date -u) ====="
HAVE=$(find "$OUT" -name '*.json' -o -name 'firing.txt' 2>/dev/null | wc -l)
if [ "$RESCUE" -ne 0 ] && [ "$HAVE" -gt 0 ]; then
  echo "[verdict] *** publish failed with $HAVE artefact(s) -- holding for rescue ***"
  sleep infinity
fi
