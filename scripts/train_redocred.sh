#!/usr/bin/env bash
#
# Train the RE-DocRED (document-level NER + relation extraction) configs
# sequentially: base then large. Each run is
# `uv run python tools/train/train.py --config <cfg>`, tee'd to
# out/train_logs/<name>.log; a failing run is recorded so the batch continues.
# A "Summary:" line prints at the end (download_models.sh --wait waits for it).
#
# Unlike train_all_events.sh this does NOT touch the event sweep table in
# PAPER.md (RE-DocRED is a relation corpus, not an event one).
#
#   bash scripts/train_redocred.sh
#
# Launch it detached and capture the batch log so --wait can detect completion:
#   tmux new -d -s train 'bash scripts/train_redocred.sh > out/train_logs/batch.log 2>&1'

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
CONFIG_DIR="tools/train/config"
LOG_DIR="out/train_logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
  gliner2-base-v1-redocred
  gliner2-large-v1-redocred
)

# Refuse to start if a training is already running (two heavy trainings on one
# GPU exhaust memory and both OOM).
existing="$(pgrep -f 'tools/train/train.py' 2>/dev/null || true)"
if [ -n "$existing" ]; then
  echo "ERROR: a training process is already running (PID(s): $existing). Aborting." >&2
  exit 1
fi

OK=()
FAIL=()

for name in "${CONFIGS[@]}"; do
  cfg="$CONFIG_DIR/$name.yaml"
  log="$LOG_DIR/$name.log"
  echo ""
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START  $name"
  echo "  config: $cfg"
  echo "  log:    $log"
  echo "============================================================"

  if [ ! -f "$cfg" ]; then
    echo "[skip] config not found: $cfg"
    FAIL+=("$name (missing config)")
    continue
  fi

  if uv run python tools/train/train.py --config "$cfg" 2>&1 | tee "$log"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE   $name"
    OK+=("$name")
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $name (see $log)"
    FAIL+=("$name")
  fi
done

echo ""
echo "============================================================"
echo "Summary: ${#OK[@]} ok, ${#FAIL[@]} failed"
if [ "${#OK[@]}" -gt 0 ]; then
  printf '  ok:   %s\n' "${OK[@]}"
fi
if [ "${#FAIL[@]}" -gt 0 ]; then
  printf '  fail: %s\n' "${FAIL[@]}"
fi
echo "============================================================"

[ "${#FAIL[@]}" -eq 0 ]
