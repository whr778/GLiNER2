#!/bin/bash
# LABEL NEGATIVES A/B + the full-menu re-baseline, on ONE box that terminates itself.
#
# WHAT THIS ANSWERS. A model trained only on menus built from its own gold is never shown a
# label it must reject. Measured: the incumbent fires on 63 of 100 documents given a schema of
# ONLY absent event types, and its real event_type precision is 0.5521 where the blind test
# reports 1.0000 by construction. The treatment arm offers absent labels during training; the
# control is identical but for `negative_labels_per_dim`.
#
# ACCEPTANCE IS REJECTION, NOT F1. Absent-type firing must fall and full-menu precision must
# rise, WITHOUT gold-menu recall collapsing -- an overcorrecting abstention gate is the
# failure mode on the other side, and a gate that admits nothing has a perfect FP rate.
#
# GATES, read BEFORE any metric. Both are in-band and both CAN FAIL:
#   [composition]   negatives: ...            control NONE, treatment populated
#   negative queries: N absent available...   an arm printing available=0 did NOT apply it
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1

PY=./.venv/bin/python
OUT=$HOME/negab
DEST=${DEST:-negatives_ab}
mkdir -p "$OUT"
source tools/lambda/_publish.sh
RESCUE=0
trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

# --- 1. RE-BASELINE the two existing models under the full-menu mode --------------------
# This has to exist BEFORE any negatives-trained model does, or there is nothing to compare
# the treatment against except a number produced by a menu that cannot be wrong.
for m in eb16-rebuild-tr:base/eb16-rebuild-tr eb16-eventrecords-tr:ab/eventrecords-ep1-eval; do
  name=${m%%:*}; cfgp=${m##*:}
  echo "[negab] ===== full-menu re-baseline $name  $(date -u) ====="
  CK=$HOME/ckpt/$name
  [ -f "$CK/model.safetensors" ] || $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('whr778/gliner2-$name', local_dir='$CK')" || { echo "[negab] download FAILED"; continue; }
  $PY -u tools/train/eval.py --config "tools/train/config/$cfgp.yaml" \
      --checkpoint "$CK" --split test --threshold 0.3 --full-menu 2>&1 | tee "$OUT/rebase-$name.log" | tail -3
  for f in out/*/test_metrics.json; do [ -f "$f" ] && cp "$f" "$OUT/rebase-$name.json"; done
  publish "$DEST" "$OUT/rebase-$name.json" "$OUT/rebase-$name.log" || RESCUE=1
done

# --- 2. THE A/B ------------------------------------------------------------------------
for arm in control treatment; do
  echo "[negab] ===== $arm  $(date -u) ====="
  $PY -u tools/train/train.py --config "tools/train/config/ab/negatives-$arm.yaml" \
      2>&1 | tee "$OUT/$arm.log" | grep -aE "composition|negative queries|Blind test|eval_" | tail -20
  echo "[negab] $arm rc=${PIPESTATUS[0]}"
  cp "out/negatives-$arm/test_metrics.json" "$OUT/$arm.json" 2>/dev/null || echo "[negab] NO METRICS $arm"
  publish "$DEST" "$OUT/$arm.json" "$OUT/$arm.log" || RESCUE=1
done

# --- 3. THE ACCEPTANCE METRIC: does the treatment REJECT? --------------------------------
for arm in control treatment; do
  CK=out/negatives-$arm/best
  [ -d "$CK" ] || continue
  echo "[negab] ===== absent-type firing, $arm ====="
  $PY -u tools/train/probe_event_type_fp.py --checkpoints "$arm=$CK" \
      --n 150 --threshold 0.3 --device cuda 2>&1 | tail -4 | tee -a "$OUT/firing.txt"
done
publish "$DEST" "$OUT/firing.txt" || RESCUE=1

echo "[negab] ===== DONE $(date -u) ====="
if [ "$RESCUE" -ne 0 ]; then
  echo "[negab] *** A PUBLISH FAILED -- holding the box; artefacts in $OUT ***"
  sleep infinity
fi
