#!/bin/bash
# SCORE THE BLIND TEST ONCE AT THE RECORD-GATE OPERATING POINT THE SWEEP PICKED.
#
# THE SWEEP PICKED ON VALIDATION. `record_anchor_threshold` 0.5 -> 0.1 moved
# event_argument strict F1 0.0342 -> 0.0966 on 4,000 VAL records. That is a pick, not a
# result: this job scores the blind test, once, at that point.
#
# IT RUNS ITS OWN CONTROL, and that is the reason this is not just `eval.py`. eb17-best's
# published test_metrics.json was written by the training run under PRE-MERGE code; the
# code has since moved (OneIE metrics, the record-metadata repair, the boundary merge), so
# a 0.1 number scored today against that file would be confounded by the code, not just by
# the threshold. Both arms are scored HERE, same commit, same split, same menu, and the
# only difference between them is the flag.
#
# THE FLAG IS LOAD-BEARING. Without `record_anchor_threshold_wins` the record gate IS the
# span gate and `record_anchor_threshold` changes NOTHING -- traced 2026-09-29, the setting
# moved 0.5/0.1/0.01 while the decode used 0.3 each time and the output never changed.
# eval.py's --record-anchor-threshold sets the flag and clamps the proposal threshold for
# you; if the CONTROL and TREATMENT rows below come back identical, that wiring broke and
# this job measured nothing.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1
# The allocator's own advice from the OOM above: fragmentation was 506MB reserved-but-
# unallocated at the point it died.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=./.venv/bin/python
CFG=${CFG:-tools/train/config/base/eb17-best.yaml}
OUTDIR=${OUTDIR:-out/eb17-best}
HFCKPT=${HFCKPT:-whr778/gliner2-eb17-best}
NAME=${NAME:-eb17-best}
CKPT=${CKPT:-$HOME/ckpt/$NAME}
THRESH=${THRESH:-0.3}
# BATCH IS CARD-DEPENDENT AND THIS JOB IS NOT. The 2026-09-29 launch inherited
# batch-size 8 from a sweep that ran on a 40GB A100 and put it on a 22GB A10: the
# control arm died with "tried to allocate 2.91 GiB ... 2.50 GiB is free" ~11 minutes
# in, and the treatment arm walked into the same wall. Eval is GPU-insensitive on this
# workload, so the cheap card is right and the BATCH has to come down to match it.
BATCH=${BATCH:-2}
ANCHOR=${ANCHOR:-0.1}
DEST=${DEST:-eb17_blind_record_anchor}
OUT=$HOME/blindtest
mkdir -p "$OUT"
source tools/lambda/_publish.sh

trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

echo "[blind] ===== START $(date -u) ====="
$PY - <<'PROV' | tee "$OUT/provenance.txt"
import subprocess
print("git_commit", subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip())
import torch, transformers
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} built_for={torch.version.cuda}")
print(f"transformers {transformers.__version__}")
PROV

if [ ! -f "$CKPT/model.safetensors" ]; then
  echo "[blind] fetching $HFCKPT"
  $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$HFCKPT', local_dir='$CKPT')" || exit 2
fi

# ---- the two arms --------------------------------------------------------------------
run_arm() {                       # $1 = arm name, $2... = extra eval.py args
  local arm=$1; shift
  echo "[blind] --- arm $arm  $(date -u) ---"
  rm -f "$OUTDIR/test_metrics.json"
  timeout 7200 $PY -u tools/train/eval.py --config "$CFG" --checkpoint "$CKPT" \
      --split test --threshold "$THRESH" --batch-size "$BATCH" "$@" 2>&1 \
      | tee "$OUT/$arm.log" | tail -20
  local rc=${PIPESTATUS[0]}
  if [ -f "$OUTDIR/test_metrics.json" ]; then
    cp "$OUTDIR/test_metrics.json" "$OUT/$arm.json"
    publish "$DEST" "$OUT/$arm.json" "$OUT/$arm.log" || true
  else
    echo "[blind] *** arm $arm wrote NO metrics (eval exited $rc) -- FAILED ARM ***"
  fi
}

run_arm control
run_arm anchor$ANCHOR --record-anchor-threshold "$ANCHOR"

# ---- the comparison, and the gate ----------------------------------------------------
$PY - "$OUT/control.json" "$OUT/anchor$ANCHOR.json" <<'CMP' | tee "$OUT/compare.txt"
import json, sys
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
if not (a.is_file() and b.is_file()):
    print(f"[blind] *** MISSING ARM: control={a.is_file()} treatment={b.is_file()} -- "
          "no comparison is possible and this job FAILED ***")
    raise SystemExit(2)
c, t = json.loads(a.read_text()), json.loads(b.read_text())
keys = sorted(k for k in c if k in t and isinstance(c[k], (int, float)))
print(f"{'metric':44} {'control':>10} {'anchor':>10} {'delta':>10}")
moved = 0
for k in keys:
    d = t[k] - c[k]
    if abs(d) > 1e-9:
        moved += 1
    print(f"{k:44} {c[k]:>10.4f} {t[k]:>10.4f} {d:>+10.4f}")
print()
if moved == 0:
    print("[blind] *** THE ARMS ARE IDENTICAL on every metric. The record-anchor override "
          "did not reach the decode, so this job measured NOTHING -- it is not a null "
          "result. Check that record_anchor_threshold_wins survived the eval-time filter. ***")
    raise SystemExit(3)
print(f"[blind] {moved} of {len(keys)} metrics moved -- the treatment reached the decode.")
CMP
CMPRC=$?

publish "$DEST" "$OUT/provenance.txt" "$OUT/compare.txt" || true
echo "[blind] ===== END $(date -u)  compare_rc=$CMPRC ====="
exit $CMPRC
