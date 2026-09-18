#!/bin/bash
# RECOVER ONE LOST BLIND TEST, like-for-like against the published re-baseline.
#
# WHY IT EXISTS. On 2026-09-17 a 16-hour run finished and pushed its model, then lost its
# test_metrics.json to a network outage. The model is on the Hub; only the evaluation is
# gone. This regenerates the evaluation -- ONE pass, ~0.5 A100-hour, not a $28 retrain.
#
# THE COMPARISON IS FIXED BY THE CONTROL, NOT BY CHOICE. The control is
# run-logs:negatives_ab/rebase-eb16-eventrecords-tr.json, produced by negatives_ab.sh as
#   eval.py --config ab/eventrecords-ep1-eval.yaml --split test --threshold 0.3 --full-menu
# so the treatment must be scored at THE SAME threshold, on THE SAME config's split, with
# THE SAME menu. The model card's own warning is the reason: "numbers quoted at each model's
# own threshold are read at different operating points and are not directly comparable."
# Do NOT sweep here and quote the winner -- that is a different measurement.
#
# --full-menu IS LOAD-BEARING, not a nicety. Under the gold menu the menu cannot express a
# wrong answer, so the incumbent's event_type precision is 1.0000 BY CONSTRUCTION. Offered
# its own taxonomy (78 events, 843 relations) the same checkpoint scores 0.2228, relation
# 0.5113 -> 0.0223. Negatives are a PRECISION intervention, so the fullmenu_* keys are where
# the treatment is visible at all.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1

PY=./.venv/bin/python
CFG=${CFG:-tools/train/config/ab/eventrecords-ep1-eval.yaml}
OUTDIR=${OUTDIR:-out/eventrecords-ep1-eval}   # MUST match CFG's output_dir or the copy finds nothing
HFCKPT=${HFCKPT:-whr778/gliner2-eb16-eventrecords-neg}
NAME=${NAME:-eb16-eventrecords-neg}
CKPT=${CKPT:-$HOME/ckpt/$NAME}
THRESH=${THRESH:-0.3}
DEST=${DEST:-negatives_ab}
OUT=$HOME/rescore
mkdir -p "$OUT"
source tools/lambda/_publish.sh
RESCUE=0

trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

if [ ! -f "$CKPT/model.safetensors" ]; then
  echo "[rescore] fetching $HFCKPT"
  $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$HFCKPT', local_dir='$CKPT')" || exit 2
fi

# NO PRE-FLIGHT DECODE GATE, deliberately. The obvious one -- call extract_events and
# require n>0 -- was built and DISCARDED here: on the incumbent, a checkpoint that scores
# event_type F1 0.8156 through the eval path, extract_events returns {"Earthquake": []} for
# an unambiguous earthquake sentence at every threshold from 0.5 down to 0.01, and returns
# the same for a negative control. A gate that cannot separate those two is not a gate.
# (The first draft was worse: it summed len() over the OUTER dict and reported "1 event" for
# an empty result -- the same nested-shape misread that made an extract_entities probe print
# a flat 100% "not predicted" across eleven roles.)
# The job is ~30 minutes and publishes its log, so a real failure is visible for free.
# OPEN QUESTION, worth its own look: why does the convenience API decode nothing on a
# checkpoint the eval path scores 0.8156 on? Not chased here -- it does not gate this run.

echo "[rescore] ===== TEST $NAME  threshold=$THRESH  full-menu  $(date -u) ====="
$PY -u tools/train/eval.py --config "$CFG" --checkpoint "$CKPT" \
    --split test --threshold "$THRESH" --full-menu 2>&1 | tee "$OUT/rebase-$NAME.log" | tail -25
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && echo "[rescore] *** eval.py exited $rc ***"

# The metrics are the finding and are kilobytes. Publish them before anything else.
if [ -f "$OUTDIR/test_metrics.json" ]; then
  cp "$OUTDIR/test_metrics.json" "$OUT/rebase-$NAME.json"
  publish "$DEST" "$OUT/rebase-$NAME.json" "$OUT/rebase-$NAME.log" || RESCUE=1
else
  # A named-but-never-written artefact is a FAILED job, not an empty publish. Say so and
  # EXIT: the trap has already published the log, so there is nothing left on this box worth
  # paying to keep alive. Holding here would bill the full hard deadline for a crash.
  echo "[rescore] *** NO metrics at $OUTDIR/test_metrics.json -- FAILED JOB ***"
  publish "$DEST" "$OUT/rebase-$NAME.log" || true
  exit 4
fi

echo "[rescore] ===== DONE $(date -u) ====="
# Hold ONLY when the finding exists but could not be shipped -- that is the one case where
# an SSH rescue can still recover something the box alone cannot.
[ "$RESCUE" -ne 0 ] && { echo "[rescore] *** metrics EXIST but publish FAILED; holding for rescue ***"; sleep infinity; }
exit 0
