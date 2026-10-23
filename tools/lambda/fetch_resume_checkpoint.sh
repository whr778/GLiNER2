#!/bin/bash
# Fetch a resumable checkpoint from the Hub into a run's output_dir, and refuse unless the run can actually use it.
#   bash tools/lambda/fetch_resume_checkpoint.sh REPO:SUBDIR OUTDIR CFG
# Downloads SUBDIR/* (model, configs AND training_state.pt) under OUTDIR, then gates: training_state.pt present and
# at least 1 GB (the eb22c optimizer state is 2.5 GB; a weights-only checkpoint has none), and CFG carries
# `checkpoint_restart`, without which the trainer ignores the directory and silently trains from scratch.
# Prints the directory listing so the job log shows what the trainer will find. Exit 1 on any refusal.
set -uo pipefail
REF=${1:?REPO:SUBDIR}; OUTDIR=${2:?OUTDIR}; CFG=${3:?CFG}
REPO=${REF%%:*}; SUB=${REF#*:}
[ "$REPO" != "$SUB" ] || { echo "[resume] *** FATAL: '$REF' is not REPO:SUBDIR ***"; exit 1; }
grep -qE "^\s*checkpoint_restart:" "$CFG" || { echo "[resume] *** FATAL: $CFG has no checkpoint_restart -- the checkpoint would be ignored ***"; exit 1; }
mkdir -p "$OUTDIR"
PY=${PY:-python}
$PY - "$REPO" "$SUB" "$OUTDIR" <<'PYR' || { echo "[resume] *** FATAL: download of $REF failed ***"; exit 1; }
import os, sys
from huggingface_hub import snapshot_download
repo, sub, out = sys.argv[1:4]
snapshot_download(repo, allow_patterns=[f"{sub}/*"], local_dir=out, token=os.environ.get("HF_TOKEN"))
PYR
ST=$OUTDIR/$SUB/training_state.pt
[ -s "$ST" ] || { echo "[resume] *** FATAL: $ST missing -- not a resumable checkpoint ***"; exit 1; }
SZ=$(wc -c < "$ST" | tr -d ' ')
[ "$SZ" -ge 1000000000 ] || { echo "[resume] *** FATAL: $ST is $SZ bytes -- no optimizer state in it ***"; exit 1; }
echo "[resume] $REF -> $OUTDIR/$SUB | training_state.pt $SZ bytes | files: $(ls "$OUTDIR/$SUB" | tr '\n' ' ')"
