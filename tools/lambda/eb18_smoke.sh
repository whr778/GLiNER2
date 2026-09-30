#!/bin/bash
# eb18 P5 SMOKE: does eb18-balanced train at the 768 gold budget, and at what cost?
#
# eb18 raised max_gold_per_query and training_candidate_budget to 768 (eb17: 256) so that no
# training sample is thrown away. That is the one change with an unmeasured GPU price. This
# run answers three things before the ~day-long base run is bought:
#
#   1. step time      -- train_samples_per_second from the Trainer summary;
#   2. peak memory    -- nvidia-smi sampled every 5s into gpu.csv (the card the full run uses);
#   3. capacity hits  -- how often truncate_with_warning fires at 768 (counted in the log).
#
# A finite loss is checked from the logged `loss` lines. Eval is OFF: this times the train
# loop. The log, the generated config and gpu.csv ship WHATEVER the exit code (_publish.sh).
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1   # an sdpa fallback would make the timing meaningless

PY=./.venv/bin/python
STEPS=${STEPS:-200}
DEST=${DEST:-eb18_smoke}
SRC=tools/train/config/base/eb18-balanced.yaml
OUT=$HOME/smoke
mkdir -p "$OUT"
source tools/lambda/_publish.sh

# The generated config must sit beside SRC: labels_file resolves relative to the config.
cfg=$(dirname "$SRC")/smoke-eb18.yaml
trap 'rm -f "$cfg"' EXIT
$PY - "$SRC" "$cfg" "$STEPS" <<'PY'
import sys, yaml
src, dst, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = yaml.safe_load(open(src))
t = c["training"]
t["max_steps"] = steps
t["eval_strategy"] = "no"
t["save_best"] = False
t["logging_steps"] = 10
t["output_dir"] = "./out/smoke-eb18"
t["experiment_name"] = "smoke_eb18"
bh = c["model"]["boundary_head"]
print(f"[smoke] wrote {dst} max_steps={steps} max_gold_per_query={bh.get('max_gold_per_query')} "
      f"training_candidate_budget={bh.get('training_candidate_budget')}")
yaml.safe_dump(c, open(dst, "w"), sort_keys=False, allow_unicode=True)
PY

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5 > "$OUT/gpu.csv" 2>&1 &
SMI=$!

echo "[smoke] === training eb18, $STEPS steps  $(date -u) ==="
start=$(date +%s)
timeout 5400 $PY -u tools/train/train.py --config "$cfg" 2>&1 | tee "$OUT/eb18.log" | tail -3
rc=${PIPESTATUS[0]}
kill $SMI
echo "[smoke] === done rc=$rc after $(( $(date +%s) - start ))s ==="

RESCUE=0
publish "$DEST" "$OUT/eb18.log" "$OUT/gpu.csv" "$cfg" || RESCUE=1

echo
echo "================ eb18 SMOKE RESULT ================"
echo "  rc=$rc"
echo "  $(grep -aoE "train_samples_per_second[^,}]*" "$OUT/eb18.log" | tail -1)"
echo "  $(grep -aoE "train_runtime[^,}]*" "$OUT/eb18.log" | tail -1)"
echo "  peak memory.used: $(awk -F', ' 'NR>1{gsub(/ MiB/,"",$2); if($2>m)m=$2} END{print m" MiB"}' "$OUT/gpu.csv")"
echo "  batches that truncated gold: $(grep -ac "on_capacity_exceeded='truncate_with_warning'" "$OUT/eb18.log")"
echo "  non-finite loss lines: $(grep -aE "'loss': " "$OUT/eb18.log" | grep -ciE "nan|inf")"
echo "  loss lines: $(grep -acE "'loss': " "$OUT/eb18.log")"
echo "==================================================="

if [ "$RESCUE" -ne 0 ]; then
  echo "[smoke] *** PUBLISH FAILED -- holding the box; the hard-deadline watchdog still terminates it ***"
  sleep infinity
fi
