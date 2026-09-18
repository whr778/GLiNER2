#!/bin/bash
# RECOVER ONE LOST BLIND TEST, like-for-like against the published re-baseline.
#
# WHY IT EXISTS. On 2026-09-17 a 16-hour run finished and pushed its model, then lost its
# test_metrics.json to a network outage. The model is on the Hub; only the evaluation is
# gone. This regenerates the evaluation -- ~1 pass, not a $28 retrain.
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

# SANITY BEFORE SPEND: the model must DECODE EVENTS, not merely load. A boundary model that
# loads cleanly and emits zero events is a failure already on file (159ef04). Loading is not
# evidence of working, and this run exists to produce a number -- not to discover at the end
# that every event metric is 0.0 for a mechanical reason.
echo "[rescore] sanity: does it decode events at all?"
CKPT="$CKPT" $PY - <<'SANITY' || { echo "[rescore] *** SANITY FAILED -- not spending ***"; exit 3; }
import json, os, torch
from gliner2 import AutoExtractor
m = AutoExtractor.from_pretrained(os.environ["CKPT"], architecture="boundary").eval()
if torch.cuda.is_available():
    m = m.to("cuda")
text = ("The company announced on Tuesday that it had acquired the startup for $2 billion, "
        "and the deal will close in March.")
with torch.no_grad():
    out = m.extract_events(text, {"Acquisition": ["buyer", "target", "price", "date"]})
evs = (out or {}).get("events", out)
n = sum(len(v) for v in evs.values()) if isinstance(evs, dict) else len(evs or [])
print(f"[rescore] sanity decode -> {json.dumps(out, ensure_ascii=False)[:240]}")
print(f"[rescore] sanity event count = {n}")
raise SystemExit(0 if n > 0 else 1)
SANITY

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
  echo "[rescore] *** NO metrics at $OUTDIR/test_metrics.json -- FAILED JOB, not an empty publish ***"
  publish "$DEST" "$OUT/rebase-$NAME.log" || RESCUE=1
  RESCUE=1
fi

echo "[rescore] ===== DONE $(date -u) ====="
[ "$RESCUE" -ne 0 ] && { echo "[rescore] *** publish failed or no metrics; holding for inspection ***"; sleep infinity; }
exit 0
