#!/bin/bash
# Per-epoch junction DET DURING a base run: wait for each checkpoint-epoch-N on the box, copy the
# model files, verify the SHA-256, run tools/train/measure_junction_det.py, then delete the weights.
#
#   nohup bash tools/lambda/pull_epoch_det.sh <box-ip> <first-epoch> <last-epoch> > log 2>&1 &
#
# Why it exists: the runner pushes epoch checkpoints only AFTER train.py exits, so the overfitting
# curve otherwise waits for the whole run. Read-only on the box. Plain bash ON PURPOSE: under zsh
# an unquoted "$O" holding several ssh options is ONE word, every ssh failed rc=255 and the first
# watcher looped 3.5 h without copying anything. scp over SFTP does not expand a remote {a,b}
# list either, so files are copied one by one.
set -uo pipefail
IP=$1; FIRST=$2; LAST=$3
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR)
BOX=ubuntu@$IP
REMOTE=gliner2/out/eb19
LOCAL=/Volumes/Development/tmp/eb19_epochs
OUT=/Volumes/Development/tmp/junction_det
cd "$(dirname "$0")/../.."

for N in $(seq "$FIRST" "$LAST"); do
  echo "[pull] waiting for checkpoint-epoch-$N  $(date -u)"
  until ssh "${SSH_OPTS[@]}" "$BOX" "tr '\r' '\n' < ~/box.log | grep -aq \"Saved full checkpoint 'checkpoint-epoch-$N'\""; do
    rc=$?
    [ "$rc" -eq 255 ] && ! ssh "${SSH_OPTS[@]}" "$BOX" true 2>/dev/null && { echo "[pull] box unreachable $(date -u) -- run over or terminated; stopping"; exit 0; }
    sleep 300
  done
  D=$LOCAL/checkpoint-epoch-$N
  mkdir -p "$D"
  for f in config.json model.safetensors tokenizer.json tokenizer_config.json encoder_config; do
    scp "${SSH_OPTS[@]}" -r "$BOX:$REMOTE/checkpoint-epoch-$N/$f" "$D/" || { echo "[pull] copy FAILED: $f"; exit 1; }
  done
  want=$(ssh "${SSH_OPTS[@]}" "$BOX" "sha256sum $REMOTE/checkpoint-epoch-$N/model.safetensors" | cut -c1-64)
  got=$(shasum -a 256 "$D/model.safetensors" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "[pull] CHECKSUM MISMATCH epoch $N: box '$want' local '$got'"; exit 1; }
  echo "[pull] epoch $N copied, checksum matches  $(date -u)"
  uv run python tools/train/measure_junction_det.py --checkpoint "$D" --docs 48 \
      --out "$OUT/eb19_epoch$N" > "$OUT/eb19_epoch$N.log" 2>&1 || { echo "[pull] measure FAILED epoch $N"; exit 1; }
  grep -E "^\[pair\]" "$OUT/eb19_epoch$N.log"
  rm -rf "$D"
  echo "[pull] epoch $N measured; weights deleted  $(date -u)"
done
echo "[pull] done through epoch $LAST"
