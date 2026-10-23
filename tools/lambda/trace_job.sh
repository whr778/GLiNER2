#!/bin/bash
# TRACE JOB: plain single-GPU train.py on CFG, stopped right after the eval at STOP_STEP, publishing the log and
# eval_history.jsonl to run-logs DEST/ every 20 minutes and at the end.
#
# Why a step stop and not max_steps: max_steps can resize the LR schedule, and the schedule is what the eb21 traces
# test (eb22c's events collapse at steps ~4-8k on one GPU, 2026-10-20). The run keeps its real num_epochs; this job
# only decides when to stop watching.
#
#   JOB="CFG=tools/train/config/base/eb21-trace.yaml DEST=eb21-trace STOP_STEP=12000 bash tools/lambda/trace_job.sh"
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1
PY=./.venv/bin/python
CFG=${CFG:?CFG}; DEST=${DEST:?DEST}; STOP_STEP=${STOP_STEP:?STOP_STEP}
STOP_EPOCH=${STOP_EPOCH:-}  # set: stop once checkpoint-epoch-N (with training_state.pt) is fully written -- a clean restart point
NGPU=${NGPU:-1}          # >1: torchrun (plain DDP since 0b21629; [ddp-sync] at every eval proves the ranks agree)
CKPT_REPO=${CKPT_REPO:-} # set: upload each new checkpoint-<step> (weights + configs, NOT training_state.pt) to this repo
RESUME_FROM=${RESUME_FROM:-} # set: REPO:checkpoint-dir on the Hub, fetched (with training_state.pt) into output_dir before
                             # train.py starts; CFG must carry checkpoint_restart, and the trainer must LOG the resume
                             # (gated below) or the run is killed -- a silent fresh start would spend the box for nothing
OUTDIR=$($PY -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['output_dir'])")
HIST=$OUTDIR/eval_history.jsonl
LOG=$HOME/trace.log
source tools/lambda/_publish.sh
echo "[trace] ===== START $(date -u) | $(git rev-parse --short HEAD) | $CFG | stop after eval at step $STOP_STEP ====="
if [ -n "$RESUME_FROM" ]; then
  PY=$PY bash tools/lambda/fetch_resume_checkpoint.sh "$RESUME_FROM" "$OUTDIR" "$CFG" || { echo "[trace] *** FATAL: resume checkpoint not usable ***"; exit 1; }
  echo "${RESUME_FROM#*:}" >> "$HOME/uploaded.txt"   # already on the Hub: do not upload the 3.8 GB again
fi

if [ "$NGPU" -gt 1 ]; then
  PYTHONUNBUFFERED=1 $PY -m torch.distributed.run --standalone --nproc_per_node="$NGPU" tools/train/train.py --config "$CFG" > "$LOG" 2>&1 &
else
  PYTHONUNBUFFERED=1 $PY -u tools/train/train.py --config "$CFG" > "$LOG" 2>&1 &
fi
TRAIN=$!
upload_new_checkpoints() {  # every checkpoint to step 8000, then every 2000; verified on the Hub's file list
  [ -n "$CKPT_REPO" ] || return 0
  $PY - "$OUTDIR" "$CKPT_REPO" "$HOME/uploaded.txt" <<'PYU'
import os, re, sys
from pathlib import Path
from huggingface_hub import HfApi
out, repo, done_file = sys.argv[1], sys.argv[2], Path(sys.argv[3])
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, private=True, exist_ok=True)
done = set(done_file.read_text().split()) if done_file.exists() else set()
for d in sorted(Path(out).glob("checkpoint-*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
    step = int(re.sub(r"\D", "", d.name) or -1)
    if d.name in done or step < 0 or not (step <= 8000 or step % 2000 == 0) or not (d / "model.safetensors").exists():
        continue
    epoch_ckpt = d.name.startswith("checkpoint-epoch-")   # a restart point: keep its optimizer state
    api.upload_folder(folder_path=str(d), path_in_repo=d.name, repo_id=repo,
                      ignore_patterns=[] if epoch_ckpt else ["training_state.pt"], commit_message=f"{d.name}")
    files = set(api.list_repo_files(repo))
    if (f"{d.name}/model.safetensors" in files and f"{d.name}/config.json" in files
            and (not epoch_ckpt or f"{d.name}/training_state.pt" in files)):
        done.add(d.name); done_file.write_text("\n".join(sorted(done)))
        print(f"[trace] uploaded {d.name} -> {repo} (verified)", flush=True)
    else:
        print(f"[trace] *** upload of {d.name} NOT visible on the Hub; will retry ***", flush=True)
PYU
}
last_pub=$(date +%s)
last_step() { [ -s "$HIST" ] && $PY -c "import json; print(json.loads(open('$HIST').read().splitlines()[-1])['eval_step'])" || echo 0; }
while kill -0 $TRAIN 2>/dev/null; do
  sleep 60
  step=$(last_step)
  if [ -n "$RESUME_FROM" ] && [ -z "${resume_ok:-}" ] && grep -aq "\[grads\] step" "$LOG"; then
    # the trainer logs "Resumed at epoch E, global step S" before its first step; a run that is stepping without it
    # started fresh and must not spend the box
    if grep -aq "Resumed at epoch" "$LOG"; then
      resume_ok=1; grep -a "Resuming (checkpoint_restart\|Resumed at epoch" "$LOG" | sed 's/^/[trace] resume proof: /'
    else
      echo "[trace] *** FATAL: training is stepping with no 'Resumed at epoch' line -- a fresh start, not a resume; killing $(date -u) ***"
      pkill -9 -f "tools/train/train.py --config $CFG" 2>/dev/null; kill -9 $TRAIN 2>/dev/null; wait $TRAIN 2>/dev/null
      publish "$DEST" "$LOG"; exit 1
    fi
  fi
  EP=$OUTDIR/checkpoint-epoch-$STOP_EPOCH/training_state.pt
  if [ -n "$STOP_EPOCH" ] && [ -s "$EP" ]; then
    sz=$(stat -c %s "$EP"); sleep 30
    if [ "$(stat -c %s "$EP")" = "$sz" ]; then
      echo "[trace] checkpoint-epoch-$STOP_EPOCH written ($sz bytes of training_state) -- stopping training $(date -u)"
      pkill -9 -f "tools/train/train.py --config $CFG" 2>/dev/null
      kill -9 $TRAIN 2>/dev/null
      break
    fi
  fi
  if [ "$step" -ge "$STOP_STEP" ]; then
    echo "[trace] eval at step $step >= $STOP_STEP -- stopping training $(date -u)"
    pkill -9 -f "tools/train/train.py --config $CFG" 2>/dev/null   # every rank, not only the launcher
    kill -9 $TRAIN 2>/dev/null
    break
  fi
  if [ $(( $(date +%s) - last_pub )) -ge 1200 ]; then
    publish "$DEST" "$LOG" $([ -s "$HIST" ] && echo "$HIST") || echo "[trace] periodic publish failed (continuing)"
    upload_new_checkpoints || echo "[trace] checkpoint upload failed (continuing; retried next round)"
    last_pub=$(date +%s)
  fi
done
wait $TRAIN 2>/dev/null
echo "[trace] training ended at eval step $(last_step) $(date -u)"
publish "$DEST" "$LOG" "$HIST" || { echo "[trace] *** FINAL PUBLISH FAILED ***"; exit 1; }
upload_new_checkpoints || { echo "[trace] *** FINAL CHECKPOINT UPLOAD FAILED ***"; exit 1; }
echo "[trace] ===== DONE $(date -u) ====="
