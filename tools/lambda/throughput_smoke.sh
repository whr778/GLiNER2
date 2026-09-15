#!/bin/bash
# THROUGHPUT SMOKE: what does `event_records: true` cost in samples/s?
#
# WHY THIS RUNS BEFORE THE BASE TRAIN. The full eb16-eventrecords-tr run is 838,760 samples
# (167,752 documents x 5 epochs), and this project has two measured throughputs for the
# same model family on the same H100:
#
#     22.0 samples/s   FA2+bf16, the curve configs
#      4.6 samples/s   the warm-start run carrying json_structures supervision
#                      -- "the record head is ~5x slower to train", unresolved since
#                      2026-08-10, with the idle-GPU / pegged-core signature of
#                      Python-side work
#
# `event_records: true` moves ~100k event records ONTO that record head. Between those two
# rates the base costs $35 or $167. Fifteen minutes here collapses a 5x cost uncertainty.
#
# BOTH ARMS RUN, CONTROL FIRST. `false` is arm 1 and is not a mistake: an absolute rate for
# `true` alone cannot separate "the record head is slow" from "this mixture is slow". The
# answer wanted is the DELTA on identical data, so the flag is flipped on one config rather
# than two configs being compared.
#
# EVERY ARM PUBLISHES, INCLUDING ON FAILURE. The first version of this script had no
# publishing at all: the box self-terminated on its normal path and took the only copy of
# the result with it (~$0.30 for nothing, 2026-09-15). If an arm dies at step 0 its log IS
# the finding, so the log ships whatever the exit code.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1   # an sdpa fallback would make the timing meaningless

PY=./.venv/bin/python
STEPS=${STEPS:-120}
DEST=${DEST:-throughput_smoke}
SRC=tools/train/config/base/eb16-eventrecords-tr.yaml
OUT=$HOME/smoke
mkdir -p "$OUT"
source tools/lambda/_publish.sh
RESCUE=0

for flag in false true; do
  cfg=$OUT/smoke-$flag.yaml
  echo "[smoke] === PHASE: writing config, event_records=$flag  $(date -u) ==="
  $PY - "$SRC" "$cfg" "$flag" "$STEPS" <<'PY'
import sys, yaml
src, dst, flag, steps = sys.argv[1], sys.argv[2], sys.argv[3] == "true", int(sys.argv[4])
c = yaml.safe_load(open(src))
c["model"]["boundary_head"]["event_records"] = flag
t = c["training"]
t["max_steps"] = steps           # HF Trainer: overrides num_epochs
t["eval_strategy"] = "no"        # timing the TRAIN loop, not eval
t["save_best"] = False
t["logging_steps"] = 10
t["output_dir"] = f"./out/smoke-{'on' if flag else 'off'}"
t["experiment_name"] = f"smoke_{'on' if flag else 'off'}"
yaml.safe_dump(c, open(dst, "w"), sort_keys=False, allow_unicode=True)
print(f"[smoke] wrote {dst} event_records={flag} max_steps={steps}")
PY

  echo "[smoke] === PHASE: loading data + training, event_records=$flag  $(date -u) ==="
  start=$(date +%s)
  timeout 3600 $PY -u tools/train/train.py --config "$cfg" 2>&1 | tee "$OUT/$flag.log" | tail -3
  rc=${PIPESTATUS[0]}
  echo "[smoke] === PHASE: done, event_records=$flag rc=$rc after $(( $(date +%s) - start ))s ==="

  # Ship the log WHATEVER happened -- a failure log is the finding when an arm dies early.
  publish "$DEST" "$OUT/$flag.log" "$cfg" || RESCUE=1
done

echo
echo "================ THROUGHPUT RESULT ================"
for flag in false true; do
  rate=$(grep -aoE "train_samples_per_second[^,}]*" "$OUT/$flag.log" 2>/dev/null | tail -1)
  rt=$(grep -aoE "train_runtime[^,}]*" "$OUT/$flag.log" 2>/dev/null | tail -1)
  # Fall back to the progress bar if the run never reached its final summary.
  bar=$(tr '\r' '\n' < "$OUT/$flag.log" 2>/dev/null | grep -aoE "[0-9.]+(it/s|s/it)" | tail -1)
  echo "  event_records=$flag  ${rate:-<no summary>}  ${rt:-}  last-bar=${bar:-none}"
done
echo "=================================================="
echo "838,760 samples for the full run. cost = samples / (rate * 3600) * \$/hr"

if [ "$RESCUE" -ne 0 ]; then
  echo "[smoke] *** A PUBLISH FAILED -- holding the box for rescue; logs in $OUT ***"
  echo "[smoke] the hard-deadline watchdog still terminates this instance."
  sleep infinity
fi
