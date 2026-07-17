#!/usr/bin/env bash
#
# Sequentially train every event-dataset config created this session, in order:
#   1. base  - English per-dataset  (fastino/gliner2-base-v1)
#   2. multi - Chinese per-dataset  (fastino/gliner2-multi-v1)
#   3. combined - joint English / Chinese / all
#   4. large - the large-model event config (fastino/gliner2-large-v1)
#
# Each run is `uv run python tools/train/train.py --config <cfg>`, its output is
# tee'd to out/train_logs/<name>.log, and a failing run is recorded so the batch
# continues to the next config. A summary prints at the end.
#
# Runs are sequential (one model at a time) and long -- the full batch is many
# hours. Comment out lines in CONFIGS to skip, or pass config names as arguments
# to run only those.
#
#   bash scripts/train_all_events.sh                      # all, in order
#   bash scripts/train_all_events.sh gliner2-base-v1-casie gliner2-base-v1-rams

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
CONFIG_DIR="tools/train/config"
LOG_DIR="out/train_logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
  # 1. base -- English per-dataset
  gliner2-base-v1-casie
  gliner2-base-v1-docee
  gliner2-base-v1-rams
  gliner2-base-v1-maven
  # 2. multi -- Chinese per-dataset
  gliner2-multi-v1-cmnee
  gliner2-multi-v1-leven
  # 3. combined
  gliner2-base-v1-events-english
  gliner2-multi-v1-events-chinese
  gliner2-multi-v1-events-all
  # 4. large -- English datasets (large-v1)
  gliner2-large-v1-casie
  gliner2-large-v1-docee
  gliner2-large-v1-rams
  gliner2-large-v1-maven
  gliner2-large-v1-wikievents
  gliner2-large-v1-events-english
)

# If config names are passed as arguments, run only those instead.
if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
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

# Non-zero exit if anything failed, so callers can detect it.
[ "${#FAIL[@]}" -eq 0 ]
