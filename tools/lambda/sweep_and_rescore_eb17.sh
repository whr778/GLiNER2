#!/bin/bash
# Two jobs on one box, both cheap, both about READING eb17 rather than training anything.
#
# 1. RECORD-GATE SWEEP ON VALIDATION. eb17's blind test was scored at a single threshold
#    that, per EVENT_ARGUMENT_DIAGNOSIS S14, is simultaneously the SPAN gate and the RECORD
#    gate -- `_decode_records` prefers the passed threshold over `record_anchor_threshold`,
#    so the two have never been separated. Measured 2026-09-29: forcing the settings to win
#    moves one record from 219 to 1465 characters. The sweep runs under that interception
#    and PICKS ON VALIDATION; the blind test is not touched here.
#
# 2. RE-SCORE THE BLIND TEST WITH CURRENT CODE. The training box ran c7901e4, which predates
#    Arg-C, Trig-I, Trig-C and Arg-I. eb17's published test_metrics.json therefore cannot
#    contain them. This re-score is the first literature-comparable argument number this
#    programme has, and Arg-I - Arg-C is the role-confusion split that exists nowhere else.
#
# WHY ONE BOX: same checkpoint, same corpora restore. Provisioning twice would double the
# only fixed cost.
#
# Three stops as always: timeout on the job, the detached watchdog in box_run.sh, and
# terminate on the normal path.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)

PY=./.venv/bin/python
CFG=${CFG:-tools/train/config/base/eb17-best.yaml}
CKPT=${CKPT:-whr778/gliner2-eb17-best}
DEST=${DEST:-eb17_sweep_rescore}
OUT=$HOME/out_sweep
mkdir -p "$OUT"
source tools/lambda/_publish.sh

echo "[sweep+rescore] ===== START $(date -u) ====="
echo "[sweep+rescore] checkpoint $CKPT  config $CFG"
$PY - <<'PROV' | tee "$OUT/provenance.txt"
import subprocess
print("git_commit", subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip())
import torch, transformers, kernels
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} built_for={torch.version.cuda}")
print(f"transformers {transformers.__version__} kernels {kernels.__version__}")
PROV

# ---- 1. the sweep, on VALIDATION ----------------------------------------------------
echo "[sweep+rescore] --- record-gate sweep (VALIDATION) $(date -u) ---"
# SUBSAMPLE FOR SHAPE. The sweep is 19 full eval passes (7 anchor + 5 field + 4
# temperature + 3 span). eb17's own per-epoch eval over the full 19,286-record val split
# took ~15 min on an A100, so 19 of them is ~4.75h -- against a JOB_TIMEOUT of 5h, with the
# re-score still to run after it. The first attempt at this job was launched without a cap
# and would have timed out mid-sweep having spent the money and produced neither result.
#
# A seeded 4,000-record subsample answers the SHAPE question -- does the optimum sit near
# 0.1 or near 0.5 -- in ~3 min a pass. The shipped threshold must then be confirmed on the
# FULL split at the chosen value, which is one more pass, not nineteen.
SWEEP_MAX_RECORDS=${SWEEP_MAX_RECORDS:-4000}
timeout 7200 $PY tools/train/sweep_record_anchor_threshold.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --axes record_anchor_threshold,record_field_threshold,record_temperature \
  --span-thresholds 0.3,0.1,0.05 \
  --max-records "$SWEEP_MAX_RECORDS" \
  --batch-size 8 --out "$OUT/record_gate_sweep_val.json" 2>&1 | tee "$OUT/sweep.log"
echo "[sweep+rescore] sweep exit ${PIPESTATUS[0]}"

# ---- 2. the blind-test re-score, with the four OneIE metrics -------------------------
# Scored at the config's own eval threshold, so it is like-for-like with the published
# test_metrics.json and the ONLY difference is the code.
echo "[sweep+rescore] --- blind-test re-score (current code) $(date -u) ---"
timeout 7200 $PY tools/train/eval.py \
  --config "$CFG" --split test --checkpoint "$CKPT" \
  --batch-size 8 2>&1 | tee "$OUT/rescore.log"
echo "[sweep+rescore] rescore exit ${PIPESTATUS[0]}"

for f in out/eb17-best/test_metrics.json test_metrics.json; do
  [ -f "$f" ] && cp "$f" "$OUT/test_metrics_rescored.json" && break
done

# METRICS ARE THE FINDING AND THEY ARE KILOBYTES. Publish them before anything else can
# fail, and verify against the Hub's own file list rather than a clean return.
# `publish <dest> <file>...` -- it takes FILES, not a directory. There is no publish_dir;
# I wrote one and it would have failed at the end of a paid run, after both jobs.
publish "$DEST" \
  "$OUT/provenance.txt" "$OUT/sweep.log" "$OUT/rescore.log" \
  "$OUT/record_gate_sweep_val.json" "$OUT/test_metrics_rescored.json" \
  || echo "[sweep+rescore] *** PUBLISH FAILED ***"
echo "[sweep+rescore] ===== END $(date -u) ====="
