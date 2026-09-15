#!/bin/bash
# EVENT THRESHOLD SWEEP on the incumbent, `whr778/gliner2-eb16-rebuild-tr`.
#
# WHY. Every event number this project quotes is a SINGLE-POINT reading at threshold 0.5 --
# event_argument 0.1178 strict / 0.5783 relaxed, event_trigger 0.6051, event_type 0.7545.
# `evaluate_config` calls `_run_blind_test` directly and does not re-sweep, and
# `sweep_record_thresholds.py` calibrates the RECORD head for structures, not these.
#
# THE SIGNATURE THAT MOTIVATES IT: event_type precision is EXACTLY 1.000 at recall 0.606.
# A head that never emits a wrong type while missing 39% of them is sitting far above its
# optimal operating point. And argument recall (0.492) IS trigger recall (0.493) -- the
# argument head inherits the cascade rather than failing independently, so a threshold that
# moves triggers moves arguments with it.
#
# The most expensive lesson on file has this exact shape: the stage-0 gate ran its whole
# life at 0.5, needed 0.998, and moving it bought more than two GPU fine-tuning runs had.
#
# METHOD, AND IT IS THE POINT: PICK ON VALIDATION, SCORE THE BLIND TEST ONCE. Sweeping on
# test and quoting the best is fitting the test set, and this programme has retracted a
# finding for that shape. The test pass runs ONE threshold -- the val winner -- and that
# number is the honest re-baseline of the incumbent.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1

PY=./.venv/bin/python
CFG=tools/train/config/base/eb16-rebuild-tr.yaml
CKPT=$HOME/ckpt/eb16-rebuild-tr
OUT=$HOME/sweep
DEST=${DEST:-event_threshold_sweep}
GRID=${GRID:-"0.1 0.2 0.3 0.4 0.5"}
mkdir -p "$OUT"
source tools/lambda/_publish.sh
RESCUE=0

[ -f "$CKPT/model.safetensors" ] || $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('whr778/gliner2-eb16-rebuild-tr', local_dir='$CKPT')"

# --- VALIDATION PASS: the only place the grid is allowed to run -------------------------
for t in $GRID; do
  echo "[sweep] === VAL threshold=$t  $(date -u) ==="
  $PY -u tools/train/eval.py --config "$CFG" --checkpoint "$CKPT" \
      --split val --threshold "$t" 2>&1 | tee "$OUT/val-$t.log" | tail -2
  cp out/eb16-rebuild-tr/val_metrics.json "$OUT/val-$t.json" 2>/dev/null \
    || echo "[sweep] NO val metrics at $t"
  rm -f out/eb16-rebuild-tr/val_metrics.json   # every pass writes this path
  publish "$DEST" "$OUT/val-$t.json" "$OUT/val-$t.log" || RESCUE=1
done

# --- PICK on validation -----------------------------------------------------------------
BEST=$($PY - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
best, bt = None, None
print("[sweep] validation grid (event_argument relaxed / strict, trigger strict):")
for f in sorted(glob.glob(os.path.join(out, "val-*.json"))):
    t = os.path.basename(f)[4:-5]
    m = json.load(open(f))
    rel = m.get("eval_event_argument_relaxed_micro_f1", 0.0)
    st  = m.get("eval_event_argument_strict_micro_f1", 0.0)
    tr  = m.get("eval_event_trigger_strict_micro_f1", 0.0)
    rr  = m.get("eval_event_argument_relaxed_micro_recall", 0.0)
    print(f"    t={t}  arg relaxed {rel:.4f} (R {rr:.4f})  arg strict {st:.4f}  trigger {tr:.4f}")
    # SELECT ON RELAXED: this sweep is about RECALL / the ceiling, not about binding.
    if best is None or rel > best:
        best, bt = rel, t
print(f"[sweep] validation winner: threshold={bt} (event_argument relaxed F1 {best:.4f})",
      file=sys.stderr)
print(bt)
PY
)
echo "[sweep] chosen on validation: $BEST"

# --- TEST: ONE threshold, ONE pass ------------------------------------------------------
echo "[sweep] === TEST threshold=$BEST (single pass)  $(date -u) ==="
$PY -u tools/train/eval.py --config "$CFG" --checkpoint "$CKPT" \
    --split test --threshold "$BEST" 2>&1 | tee "$OUT/test-$BEST.log" | tail -2
cp out/eb16-rebuild-tr/test_metrics.json "$OUT/test-$BEST.json" 2>/dev/null \
  || echo "[sweep] NO test metrics"
publish "$DEST" "$OUT/test-$BEST.json" "$OUT/test-$BEST.log" || RESCUE=1

echo "[sweep] ===== DONE $(date -u)  chosen threshold $BEST ====="
if [ "$RESCUE" -ne 0 ]; then
  echo "[sweep] *** A PUBLISH FAILED -- holding the box; artefacts in $OUT ***"
  sleep infinity
fi
