#!/bin/bash
# eb17 vs eb18 on ONE operating point: the first like-for-like delta between the two bases.
#
# eb18's blind test ran at its swept threshold 0.5 with global_decode on; eb17's recorded
# 0.1752 strict argument F1 was scored at 0.3 with global_decode off. Neither number can be
# subtracted from the other. Both test splits hold the SAME 19,879 documents (verified), and
# each model is scored through its OWN config, so each sees gold in its own label map.
#
# FIVE PASSES, two of them gates that must reproduce a recorded number:
#   0  eb17  0.3  global_decode off (its own config)  GATE: event_argument strict == 0.1752
#   1  eb17  0.3  global_decode on
#   2  eb17  0.5  global_decode on
#   3  eb18  0.3  global_decode on
#   4  eb18  0.5  global_decode on                     GATE: event_argument strict == 0.1077
# Every pass publishes its metrics JSON and log as soon as it ends; a pass without metrics
# is a FAILED pass and says so.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1

PY=./.venv/bin/python
DEST=${DEST:-rescore_eb17_eb18}
OUT=$HOME/rescore
mkdir -p "$OUT"
source tools/lambda/_publish.sh
FAILED=0

fetch() {  # repo -> local dir
  local dir=$HOME/ckpt/${1##*/}
  [ -f "$dir/model.safetensors" ] || $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$1', local_dir='$dir')" >&2 || return 1
  echo "$dir"
}

run_pass() {  # name repo config outdir threshold gd-flag
  local name=$1 repo=$2 cfg=$3 outdir=$4 thr=$5 gd=$6
  # PASSES="p3-... p4-..." re-runs only the named passes (published ones are not re-bought).
  if [ -n "${PASSES:-}" ] && [[ " $PASSES " != *" $name "* ]]; then return; fi
  local ckpt; ckpt=$(fetch "$repo") || { echo "[rescore] *** fetch failed: $repo ***"; FAILED=1; return; }
  rm -f "$outdir/test_metrics.json"
  echo "[rescore] ===== $name  $repo  threshold=$thr  $gd  $(date -u) ====="
  $PY -u tools/train/eval.py --config "$cfg" --checkpoint "$ckpt" --split test \
      --threshold "$thr" $gd 2>&1 | tee "$OUT/$name.log" | tail -4
  if [ -f "$outdir/test_metrics.json" ]; then
    cp "$outdir/test_metrics.json" "$OUT/$name.json"
    publish "$DEST" "$OUT/$name.json" "$OUT/$name.log" || FAILED=1
  else
    echo "[rescore] *** $name wrote NO metrics -- FAILED PASS ***"
    publish "$DEST" "$OUT/$name.log" || true
    FAILED=1
  fi
}

E17=tools/train/config/base/eb17-best.yaml
E18=tools/train/config/base/eb18-balanced.yaml
# The box is bootstrapped for ONE config (eb18); eb17 reads files eb18 does not (e.g. rams).
$PY tools/data/restore_from_hf.py --config "$E17" 2>&1 | tail -2
run_pass p0-eb17-t03-native whr778/gliner2-eb17-best      $E17 ./out/eb17-best     0.3 "--no-global-decode"
run_pass p1-eb17-t03-gd     whr778/gliner2-eb17-best      $E17 ./out/eb17-best     0.3 "--global-decode"
run_pass p2-eb17-t05-gd     whr778/gliner2-eb17-best      $E17 ./out/eb17-best     0.5 "--global-decode"
run_pass p3-eb18-t03-gd     whr778/gliner2-eb18-balanced  $E18 ./out/eb18-balanced 0.3 "--global-decode"
run_pass p4-eb18-t05-gd     whr778/gliner2-eb18-balanced  $E18 ./out/eb18-balanced 0.5 "--global-decode"

echo "[rescore] ===== DONE failed=$FAILED $(date -u) ====="
exit $FAILED
