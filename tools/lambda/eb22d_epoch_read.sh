#!/bin/bash
# Coref-merge calibration, then full val, on the uploaded eb22d epoch checkpoint.
# The training job stops at checkpoint-epoch-1 and never reaches calibration.
#   CFG=tools/train/config/base/eb22d-trace.yaml DEST=eb22d-epoch-read \
#   REPO=whr778/gliner2-eb22d-trace-epoch SUB=checkpoint-epoch-1 \
#   bash tools/lambda/eb22d_epoch_read.sh
# Weights only: training_state.pt is the restart, and this job does not train.
# --write sets record_merge_coreferent from the gates (link, or off if none pass).
# The eval then loads that same directory. Both files go to run-logs and back onto the checkpoint.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1
PY=./.venv/bin/python
CFG=${CFG:?CFG}
DEST=${DEST:?DEST}
REPO=${REPO:?REPO}
SUB=${SUB:-checkpoint-epoch-1}
CKPT=$HOME/ckpt/$SUB
source tools/lambda/_publish.sh
mkdir -p "$HOME/ckpt"
# box_run terminates the disk on any exit. A failure note has to be on the Hub first,
# or the only copy dies with the box (eb22d-epoch-read, 2026-10-27, box 27fab264).
fail() {
  echo "[epoch-read] *** FATAL: $* ***"
  printf '%s\n' "[epoch-read] *** FATAL: $* ***" >> "$HOME/epoch-read-fatal.txt"
  publish "$DEST" "$HOME/epoch-read-fatal.txt" || echo "[epoch-read] the failure note did not reach the hub"
  exit 1
}
echo "[epoch-read] ===== START $(date -u) | $(git rev-parse --short HEAD) | $REPO:$SUB ====="

# Bootstrap restores data only when CFG is set on the provision, and the first read
# launch set it only inside this job. Calibration opens the val files directly and
# does not fetch them, so a bare box dies at FileNotFoundError.
echo "[epoch-read] restoring data $(date -u)"
$PY tools/data/restore_from_hf.py --config "$CFG" > "$HOME/restore.log" 2>&1
rc=$?
cat "$HOME/restore.log"
if [ "$rc" -ne 0 ]; then
  cat "$HOME/restore.log" >> "$HOME/epoch-read-fatal.txt"
  fail "data restore"
fi
for f in data/cc_news_events_sonnet55_v2.val.jsonl data/cmnee.val.jsonl data/maven.val.jsonl; do
  [ -s "$f" ] || fail "missing $f after restore"
done

$PY - "$REPO" "$SUB" "$HOME/ckpt" <<'PY' || fail "download failed"
import os, sys
from huggingface_hub import snapshot_download
repo, sub, out = sys.argv[1:4]
snapshot_download(
    repo, allow_patterns=[f"{sub}/*"], ignore_patterns=[f"{sub}/training_state.pt"],
    local_dir=out, token=os.environ["HF_TOKEN"])
PY
[ -s "$CKPT/model.safetensors" ] || fail "$CKPT has no weights"
echo "[epoch-read] downloaded $(date -u) | $(ls "$CKPT" | tr '\n' ' ')"

echo "[epoch-read] calibrating $(date -u)"
$PY -u tools/train/calibrate_coref_threshold.py --config "$CFG" --checkpoint "$CKPT" --batch-size 2 --write \
  > "$HOME/calibrate.log" 2>&1
rc=$?
cat "$HOME/calibrate.log"
if [ "$rc" -ne 0 ]; then
  cat "$HOME/calibrate.log" >> "$HOME/epoch-read-fatal.txt"
  fail "calibration failed"
fi
$PY - "$CKPT" <<'PY'
import json, sys
from pathlib import Path
ckpt = Path(sys.argv[1])
bh = json.loads((ckpt / "config.json").read_text())["boundary_head"]
sweep = json.loads((ckpt / "coref_threshold_sweep.json").read_text())
print(f"[epoch-read] merge {bh.get('record_merge_coreferent')} threshold {bh.get('record_coref_link_threshold')} -- {sweep.get('why')}", flush=True)
PY

echo "[epoch-read] full val at threshold 0.3 $(date -u)"
$PY -u tools/train/eval.py --config "$CFG" --checkpoint "$CKPT" --split val --threshold 0.3 --batch-size 2 \
  > "$HOME/val.log" 2>&1
rc=$?
cat "$HOME/val.log"
if [ "$rc" -ne 0 ]; then
  cat "$HOME/val.log" >> "$HOME/epoch-read-fatal.txt"
  fail "val eval failed"
fi
[ -s "$CKPT/val_metrics.json" ] || fail "val_metrics.json was not written"
$PY - "$CKPT/val_metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
keys = ("eval_event_trigger_strict_micro_f1", "eval_event_argument_strict_micro_f1",
        "eval_event_cluster_strict_micro_f1", "eval_event_cluster_argument_strict_micro_f1")
print("[epoch-read] val " + " ".join(f"{k}={m.get(k)}" for k in keys), flush=True)
PY

# The metrics are the finding. Land them before the checkpoint config, which is the decision.
publish "$DEST" "$CKPT/coref_threshold_sweep.json" "$CKPT/val_metrics.json" \
  || fail "run-logs publish failed"

$PY - "$REPO" "$SUB" "$CKPT" <<'PY' || fail "checkpoint upload failed"
import os, sys
from huggingface_hub import HfApi
repo, sub, ckpt = sys.argv[1:4]
api = HfApi(token=os.environ["HF_TOKEN"])
names = ("config.json", "coref_threshold_sweep.json", "val_metrics.json")
sent = []
for name in names:
    api.upload_file(path_or_fileobj=f"{ckpt}/{name}", path_in_repo=f"{sub}/{name}", repo_id=repo)
    sent.append(f"{sub}/{name}")
on_hub = set(api.list_repo_files(repo))
missing = [f for f in sent if f not in on_hub]
if missing:
    raise SystemExit(f"[epoch-read] *** NOT SAVED *** {missing}")
print("[epoch-read] checkpoint updated:", ", ".join(sent), flush=True)
PY
echo "[epoch-read] ===== DONE $(date -u) ====="
