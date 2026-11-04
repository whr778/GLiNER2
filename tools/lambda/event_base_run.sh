#!/bin/bash
# THE EVENT-CAPABLE BASE: eb16's data with events on the RECORD head.
# ~14h on one A100, ~$28. Nothing here depends on the operator's laptop.
#
# SUPERSEDED 2026-09-15: these figures are of uncertain provenance and are NOT at
# threshold 0.5 -- the model card says `Decision threshold: 0.3`. Measured at the
# val-selected 0.2: event_argument strict 0.0991 / relaxed 0.5884, event_trigger
# 0.5984, event_type 0.8650. See EVENT_ARGUMENT_DIAGNOSIS.md section 4c.
# WHAT THIS RUN IS FOR. `event_argument` reads 0.1178 STRICT against 0.5783 RELAXED on the
# same predictions -- the model finds the arguments and cannot BIND them, because the
# mention path compiles ONE instance per event type and 64.2% of gold event instances share
# their type with another instance in the same document. `event_records: true` routes events
# through the record head, which is multi-instance by construction.
# (EVENT_ARGUMENT_DIAGNOSIS.md.)
#
# SURVIVAL PROPERTIES, in the order they matter for a 14-hour run:
#
#   1. THE JOB IS NOT ATTACHED TO THE LAPTOP. It runs under tmux, started by box_run.sh.
#      ssh dropping, the network going, or the laptop rebooting are invisible to it.
#   2. THE MODEL IS PUSHED FIRST, before metrics. For a base train the CHECKPOINT is the
#      irreplaceable artefact -- metrics can be recomputed from it, and it cannot be
#      recomputed from them. This inverts the order used for the cheap A/B runs, and the
#      inversion is deliberate.
#   3. THE MODEL IS PUSHED EVEN IF THE BLIND TEST DIES. train.py trains and then evaluates;
#      a crash in eval must not cost 14 hours of GPU. The push is keyed on `best/` existing,
#      not on the exit code.
#   4. EVERY PUSH RETRIES 6x over ~30 min and VERIFIES against the Hub's own file list.
#      A clean return is not proof: upload_folder has returned successfully having written
#      nothing, and that cost ~15h of A100 once.
#   5. A FAILED PUSH HOLDS THE BOX for the watchdog window instead of terminating into a
#      loss. Losing a $28 model to save $2 of idle is bad arithmetic.
#   6. THE BOX TERMINATES ITSELF on the normal path, and a detached hard-deadline watchdog
#      terminates it regardless. Three independent stops, per box_run.sh.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1   # sdpa on bf16 mmBERT is a CORRECTNESS failure, not a slowdown

PY=./.venv/bin/python
# Parameterised so the negatives rebuild reuses this runner rather than forking it --
# forking the publish block once already shipped a run with no publishing at all.
CFG=${CFG:-tools/train/config/base/eb16-eventrecords-tr.yaml}
OUTDIR=${OUTDIR:-./out/eb16-eventrecords-tr}
REPO=${REPO:-whr778/gliner2-eb16-eventrecords-tr}
DEST=${DEST:-event_base}
LOG=$HOME/event_base.log
source tools/lambda/_publish.sh
RESCUE=0

echo "[base] ===== START $(date -u) ====="
echo "[base] config $CFG -> $REPO"

NGPU=${NGPU:-auto}
if [ "$NGPU" = "auto" ]; then
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  [ "${NGPU:-0}" -ge 1 ] || NGPU=1
fi
$PY - "$CFG" "$NGPU" <<'PYS'
import sys, torch, yaml
cfg_path, ngpu = sys.argv[1], max(1, int(sys.argv[2]))
if torch.cuda.is_available():
    c = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    tr = c["training"]
    old_bs, old_acc = int(tr["batch_size"]), int(tr["gradient_accumulation_steps"])
    eff = old_bs * old_acc
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    name = torch.cuda.get_device_name(0)
    bs = 8 if vram_gb >= 70 else (4 if vram_gb >= 35 else 2)
    bs = min(bs, max(1, eff // ngpu))
    acc = max(1, eff // (bs * ngpu))
    tr["batch_size"], tr["gradient_accumulation_steps"] = bs, acc
    ev = c.setdefault("eval", {})
    old_ebs = int(ev.get("batch_size", 2))
    ebs = max(old_ebs, 4) if vram_gb >= 70 else old_ebs
    ev["batch_size"] = ebs
    yaml.safe_dump(c, open(cfg_path, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    print(f"[base] GPU scaling: {name} ({vram_gb:.1f} GB x {ngpu}) -> batch_size={bs} grad_accum={acc} eval_batch_size={ebs} (effective {bs * acc * ngpu}, was {old_bs}x{old_acc})", flush=True)
PYS

# PROVENANCE, PUBLISHED AT MINUTE ZERO. Establishing whether a finished model predated the
# bf16 consistency-loss fix had to be done by arithmetic on push timestamps, because nothing
# recorded what code built it. Write it down before the run can fail.
{
  echo "arm_dest      $DEST"
  echo "config        $CFG"
  echo "repo          $REPO"
  echo "commit        $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "branch        $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "started_utc   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "instance      $(cat ~/.instance_id 2>/dev/null || echo unknown)"
  echo "gpu           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
  echo "torch         $($PY -c 'import torch;print(torch.__version__)' 2>/dev/null)"
} > "$HOME/START.txt"
publish "$DEST" "$HOME/START.txt" "$CFG" || echo "[base] start marker did not publish (continuing)"

# A heartbeat so a 14-hour run is legible from outside without attaching to tmux.
# The FIRST line is written now: heartbeat.log is a required publish artefact, and the loop's first
# write is 15 min in, so a job shorter than that (menudose control, 14 min) reported it "never
# written" and held the box for rescue -- an hour of idle A100 on 2026-10-07.
echo "[hb] $(date -u) start" >> "$HOME/heartbeat.log"
( while sleep 900; do
    echo "[hb] $(date -u) gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' ') $(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -aoE '[0-9]+/[0-9]+ \[[^]]*\]' | tail -1)"
  done ) >> "$HOME/heartbeat.log" 2>&1 &
disown

# PERIODIC PUBLISH, so a 16-hour run is not a 16-hour hole. Nothing used to reach the Hub
# until the very end, which means a network outage or a dead box at hour 15 took the whole
# log with it -- exactly what happened on 2026-09-17. Every 30 minutes, ship the heartbeat
# and the tail of the training log. Failures here are IGNORED on purpose: this is a
# convenience stream, and it must never be able to fail the run it is reporting on.
( while sleep 1800; do
    tail -c 200000 "$LOG" > "$HOME/progress.log" 2>/dev/null
    HIST_ARG=""
    [ -s "$OUTDIR/eval_history.jsonl" ] && HIST_ARG="$OUTDIR/eval_history.jsonl"
    publish "$DEST" "$HOME/heartbeat.log" "$HOME/progress.log" $HIST_ARG >/dev/null 2>&1 || true
    if [ "${PUSH_EPOCHS:-0}" = "1" ]; then
      for ck in "$OUTDIR"/checkpoint-epoch-*; do
        [ -d "$ck" ] && [ -s "$ck/training_state.pt" ] && [ -s "$ck/model.safetensors" ] || continue
        bname=$(basename "$ck")
        grep -qxF "$bname" "$HOME/uploaded_epochs.txt" 2>/dev/null && continue
        sz=$(stat -c %s "$ck/training_state.pt" 2>/dev/null || echo 0)
        sleep 20
        [ "$(stat -c %s "$ck/training_state.pt" 2>/dev/null || echo -1)" = "$sz" ] || continue
        $PY - "$ck" "$REPO" "$bname" <<'PYE' && echo "$bname" >> "$HOME/uploaded_epochs.txt" || true
import os, sys
from huggingface_hub import HfApi
ck, repo, bname = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=open(os.path.expanduser("~/.hf_token")).read().strip())
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path=ck, path_in_repo=bname, repo_id=repo, commit_message=bname)
files = set(api.list_repo_files(repo))
if not ({f"{bname}/model.safetensors", f"{bname}/config.json", f"{bname}/training_state.pt"} <= files):
    sys.exit(1)
print(f"[base] uploaded {bname} (with training_state.pt) -> {repo}/{bname}", flush=True)
PYE
      done
    fi
  done ) >> "$HOME/publisher.log" 2>&1 &
disown

# JOB_TIMEOUT stops TRAIN.PY, not the box: the runner then publishes logs and
# terminates on the normal path. A watchdog kill skips the publish, so a short
# diagnostic run wants this knob and not a short deadline.
if [ "${NGPU:-1}" -gt 1 ]; then
  timeout ${JOB_TIMEOUT:-72000} $PY -m torch.distributed.run --standalone --nproc_per_node="$NGPU" tools/train/train.py --config "$CFG" 2>&1 | tee "$LOG"
else
  timeout ${JOB_TIMEOUT:-72000} $PY -u tools/train/train.py --config "$CFG" 2>&1 | tee "$LOG"
fi
rc=${PIPESTATUS[0]}
echo "[base] train.py rc=$rc  $(date -u)"

# 1. METRICS AND LOGS FIRST (kilobytes; published before the 600MB+ model pushes).
cp "$OUTDIR/test_metrics.json" "$HOME/test_metrics.json" 2>/dev/null || echo "[base] no test_metrics.json"
EXTRA=""
for opt_f in eval_metrics.json eval_history.jsonl train_results.json; do
  if [ -s "$OUTDIR/$opt_f" ]; then
    cp "$OUTDIR/$opt_f" "$HOME/$opt_f" 2>/dev/null && EXTRA="$EXTRA $HOME/$opt_f"
  fi
done
if [ -s "$HOME/test_metrics.json" ]; then
  publish "$DEST" "$HOME/test_metrics.json" $EXTRA "$LOG" "$HOME/heartbeat.log" "$CFG" || RESCUE=1
else
  publish "$DEST" $EXTRA "$LOG" "$HOME/heartbeat.log" "$CFG" || RESCUE=1
fi

# 2. THE MODEL, REGARDLESS OF rc.
if [ -d "$OUTDIR/best" ]; then
  echo "[base] pushing checkpoint (this is the irreplaceable artefact)"
  n=0
  until $PY -u tools/train/push_to_hub.py --checkpoint "$OUTDIR/best" \
        --repo-id "$REPO" --private 2>&1 | tail -4; do
    n=$((n+1)); echo "[base] model push FAILED (attempt $n/6)"
    [ "$n" -ge 6 ] && { echo "[base] *** MODEL PUSH FAILED AFTER 6 ATTEMPTS ***"; RESCUE=1; break; }
    sleep $((n * 120))
  done
else
  echo "[base] *** NO CHECKPOINT AT $OUTDIR/best -- nothing to push ***"
  RESCUE=1
fi

# 2b. THE FINAL-EPOCH CHECKPOINT, to a SEPARATE repo.
if [ -d "$OUTDIR/final" ]; then
  echo "[base] pushing the final-epoch checkpoint (diagnostic copy, non-fatal)"
  $PY -u tools/train/push_to_hub.py --checkpoint "$OUTDIR/final" \
      --repo-id "$REPO-final" --private \
      --commit-message "final-epoch checkpoint, for like-for-like comparison" 2>&1 | tail -3 \
    || echo "[base] final-epoch push failed; continuing (best/ is what matters)"
else
  echo "[base] no $OUTDIR/final to push"
fi

# 2c. OPTIONAL PER-EPOCH CHECKPOINTS (PUSH_EPOCHS=1), each to its own private repo. Non-fatal:
# they are the diagnostic curve (e.g. the trigger-miss probe per epoch), never the deliverable.
# Needs save_total_limit >= num_epochs in the config, or the trainer has deleted the early ones.
if [ "${PUSH_EPOCHS:-0}" = "1" ]; then
  for ck in "$OUTDIR"/checkpoint-epoch-*; do
    [ -d "$ck" ] || continue
    bname=$(basename "$ck")
    if ! grep -qxF "$bname" "$HOME/uploaded_epochs.txt" 2>/dev/null; then
      $PY - "$ck" "$REPO" "$bname" <<'PYE' && echo "$bname" >> "$HOME/uploaded_epochs.txt" || true
import os, sys
from huggingface_hub import HfApi
ck, repo, bname = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=open(os.path.expanduser("~/.hf_token")).read().strip())
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path=ck, path_in_repo=bname, repo_id=repo, commit_message=bname)
files = set(api.list_repo_files(repo))
if not ({f"{bname}/model.safetensors", f"{bname}/config.json"} <= files):
    sys.exit(1)
print(f"[base] uploaded {bname} -> {repo}/{bname}", flush=True)
PYE
    fi
    echo "[base] pushing $bname to $REPO-$bname (diagnostic, non-fatal)"
    $PY -u tools/train/push_to_hub.py --checkpoint "$ck" --repo-id "$REPO-$bname" --private \
        --commit-message "per-epoch checkpoint, for the per-epoch curve" 2>&1 | tail -2 \
      || echo "[base] $bname push failed; continuing"
  done
fi

publish "$DEST" "$LOG" "$HOME/heartbeat.log" >/dev/null 2>&1 || true

# 3. OPTIONAL TRIGGER-MISS PROBE (PROBE=1), AFTER the model AND the metrics are published,
# so a timeout during the probe can cost only the probe. Non-fatal by design: a
# probe failure must never cost the run its checkpoint or its metrics. It first TRACES on 2 docs
# per corpus (printed for reading), then runs the full sample. PROBE_BASE=1 also probes the
# warm-start base, so one box of an A/B carries the baseline.
PROBE_FILES=""
if [ "${PROBE:-0}" = "1" ] && [ -d "$OUTDIR/best" ]; then
  for target in ${PROBE_BASE:+base} arm; do
    ck="$OUTDIR/best"; [ "$target" = "base" ] && ck="${PROBE_BASE_CKPT:?PROBE_BASE needs PROBE_BASE_CKPT}"
    echo "[base] probe $target ($ck): trace on 2 docs/corpus"
    timeout 1800 $PY -u tools/train/probe_trigger_misses.py --checkpoint "$ck" --config "$CFG" \
        --device cuda --per-corpus 2 2>&1 | tee -a "$HOME/probe.log" | grep -a "^\[probe\]" | tail -40
    echo "[base] probe $target: full run"
    timeout ${PROBE_TIMEOUT:-5400} $PY -u tools/train/probe_trigger_misses.py --checkpoint "$ck" --config "$CFG" \
        --device cuda --per-corpus ${PROBE_PER_CORPUS:-200} --out "$HOME/trigger_misses_$target.json" \
        2>&1 | tee -a "$HOME/probe.log" | grep -a "^\[probe\]" | tail -60 \
      || echo "[base] probe $target failed (non-fatal)"
    [ -f "$HOME/trigger_misses_$target.json" ] && PROBE_FILES="$PROBE_FILES $HOME/trigger_misses_$target.json"
  done
  PROBE_FILES="$PROBE_FILES $HOME/probe.log"
fi

[ -n "$PROBE_FILES" ] && { publish "$DEST" $PROBE_FILES || echo "[base] probe publish failed (non-fatal)"; }

# 4. OPTIONAL ENTITY-COREF READOUT (COREF_READOUT=1): VAL pair AUC of best/'s mention link beside
# string match and cosine (tools/train/coref_readout.py). Same rules as the probe: after the model
# and metrics are published, its own timeout, non-fatal, published on its own.
if [ "${COREF_READOUT:-0}" = "1" ] && [ -d "$OUTDIR/best" ]; then
  timeout ${READOUT_TIMEOUT:-3600} $PY -u tools/train/coref_readout.py --model "$OUTDIR/best" --device cuda \
      --out "$HOME/coref_readout.json" 2>&1 | tee "$HOME/coref_readout.log" | grep -a -v "Warning" | tail -8 \
    || echo "[base] coref readout failed (non-fatal)"
  # The log publishes WITH OR WITHOUT the json: corefB lost its readout to a FileNotFoundError that
  # died with the box because the log only travelled beside the json.
  publish "$DEST" $([ -f "$HOME/coref_readout.json" ] && echo "$HOME/coref_readout.json") "$HOME/coref_readout.log" \
    || echo "[base] coref readout publish failed (non-fatal)"
fi

echo "[base] ===== DONE $(date -u) rc=$rc ====="
if [ "$RESCUE" -ne 0 ]; then
  echo "[base] *** SOMETHING DID NOT PUBLISH -- holding the box for rescue ***"
  echo "[base] checkpoint: $OUTDIR/best   logs: $LOG"
  # BOUNDED, and none at all without a checkpoint: an unbounded hold billed an idle H100 for 2 h after a startup
  # crash that had nothing to rescue (eb22c, 2026-10-19); the hard deadline alone would have allowed ~50 h.
  if [ -d "$OUTDIR/best" ] || [ -d "$OUTDIR/final" ]; then
    echo "[base] holding ${RESCUE_HOLD:-3600}s, then box_run.sh terminates"
    sleep "${RESCUE_HOLD:-3600}"
  else
    echo "[base] no checkpoint exists -- nothing to rescue, not holding"
  fi
fi
