#!/bin/bash
# eb19 SMOKE: does the eb19 recipe train cleanly FROM SCRATCH before the ~40 h base run is bought?
#
# eb19 combines three loss changes never tested together or from scratch (every A/B was a 30k warm
# start from eb18): record_role_hard_negatives 8, proposal_gold identity, and the trigger x argument
# junction with its column loss -- which trains a cold record head from step 0 here. This answers:
#
#   1. finite, settling loss  -- loss at start / middle / end; non-finite lines must be 0;
#   2. the junction learns    -- the in-run per-column junction AUC lines must rise;
#   3. step time + memory     -- train_samples_per_second and nvidia-smi peak (sizes the full run);
#   4. the treatment applied  -- proposal_gold=identity and the column-loss lines are present.
#
# max_steps scales warmup and the gold-injection anneal to the smoke, so the annealed regime is
# exercised too. Eval is OFF and no best/ is saved, so the blind test is skipped. Log, generated
# config and gpu.csv ship WHATEVER the exit code (_publish.sh).
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1   # an sdpa fallback would make the timing meaningless

PY=./.venv/bin/python
STEPS=${STEPS:-1500}
NUM_WORKERS=${NUM_WORKERS:-}   # empty = the config's own value
PROFILE=${PROFILE:-}           # non-empty = py-spy the MAIN process for 120s from step 30
DEST=${DEST:-eb19_smoke}
SRC=tools/train/config/base/eb19.yaml
OUT=$HOME/smoke
mkdir -p "$OUT"
source tools/lambda/_publish.sh

# The generated config must sit beside SRC: labels_file resolves relative to the config.
cfg=$(dirname "$SRC")/smoke-eb19.yaml
trap 'rm -f "$cfg"' EXIT
$PY - "$SRC" "$cfg" "$STEPS" "$NUM_WORKERS" <<'PY'
import sys, yaml
src, dst, steps, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
c = yaml.safe_load(open(src))
t = c["training"]
t["max_steps"] = steps
t["eval_strategy"] = "no"
t["save_best"] = False
t["logging_steps"] = 10
t["output_dir"] = "./out/smoke-eb19"
t["experiment_name"] = "smoke_eb19"
if workers:
    t["num_workers"] = int(workers)
bh = c["model"]["boundary_head"]
print(f"[smoke] wrote {dst} max_steps={steps} max_gold_per_query={bh.get('max_gold_per_query')} "
      f"training_candidate_budget={bh.get('training_candidate_budget')} num_workers={t['num_workers']}")
yaml.safe_dump(c, open(dst, "w"), sort_keys=False, allow_unicode=True)
PY

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5 > "$OUT/gpu.csv" 2>&1 &
SMI=$!
# PROOF THE WORKER COUNT APPLIED: forked DataLoader workers share train.py's command line,
# so the count is 2 + live workers (the `timeout` wrapper also carries train.py). The trainer never logs its effective value.
( while true; do pgrep -fc "tools/train/train.py"; sleep 15; done ) > "$OUT/procs.txt" 2>&1 &
PROCS=$!
# PROFILE. 37% GPU utilisation with 4 or 12 workers alike means the GPU waits on the MAIN
# process. --idle keeps blocked samples, so a CPU-GPU sync shows as the Python line that
# forces it. The main PID is the child of THIS script's `timeout 5400` -- the outer
# box_run timeout and the wrapper also carry train.py on their command lines.
PROF=
if [ -n "$PROFILE" ]; then
  uv pip install --python "$PY" -q py-spy
  (
    for _ in $(seq 240); do
      s=$(tr '\r' '\n' < "$OUT/eb19.log" 2>/dev/null | grep -aoE "\| *[0-9]+/$STEPS \[" | tail -1 | grep -oE "[0-9]+" | head -1)
      [ "${s:-0}" -ge 30 ] && break
      sleep 10
    done
    T=$(pgrep -n -f "^timeout 5400"); M=$(pgrep -P "$T")
    echo "[profile] step ${s:-none}: attaching to main pid $M ($(ps -o comm= -p $M)) $(date -u)"
    sudo "$PWD/.venv/bin/py-spy" record --pid "$M" --idle --rate 50 --duration 120 --format raw -o "$OUT/pyspy.raw.txt"
    echo "[profile] py-spy rc=$? $(date -u)"
  ) > "$OUT/profile.log" 2>&1 &
  PROF=$!
fi

echo "[smoke] === training eb19 FROM SCRATCH, $STEPS steps  $(date -u) ==="
start=$(date +%s)
timeout 5400 $PY -u tools/train/train.py --config "$cfg" 2>&1 | tee "$OUT/eb19.log" | tail -3
rc=${PIPESTATUS[0]}
kill $SMI $PROCS $PROF 2>/dev/null
echo "[smoke] === done rc=$rc after $(( $(date +%s) - start ))s ==="

RESCUE=0
FILES=("$OUT/eb19.log" "$OUT/gpu.csv" "$OUT/procs.txt" "$cfg")
[ -n "$PROFILE" ] && FILES+=("$OUT/profile.log" "$OUT/pyspy.raw.txt")   # missing = a FAILED profile
publish "$DEST" "${FILES[@]}" || RESCUE=1

echo
echo "================ eb19 SMOKE RESULT ================"
echo "  rc=$rc"
# The losses and the rate live on the PROGRESS BAR (carriage-return separated); this log has no
# 'loss': dicts, so the earlier greps read 0 loss lines and a non-finite count that could not fail.
BAR=$(tr '\r' '\n' < "$OUT/eb19.log" | grep -aoE "loss=[^,]+, lr=[^,]+, samples/s=[^,]+")
echo "  last progress: $(echo "$BAR" | tail -1)"
echo "  peak memory.used: $(awk -F', ' 'NR>1{gsub(/ MiB/,"",$2); if($2+0>m+0)m=$2+0} END{print m" MiB"}' "$OUT/gpu.csv")"
echo "  batches that truncated gold: $(grep -ac "on_capacity_exceeded='truncate_with_warning'" "$OUT/eb19.log")"
echo "  non-finite loss readings: $(echo "$BAR" | grep -cE "loss=(NaN|Inf)") of $(echo "$BAR" | grep -c .) (the total must be > 0, or nothing was read)"
echo "  max train.py processes (2 + workers): $(sort -n "$OUT/procs.txt" | tail -1)"
echo "  loss first / middle / last:"
echo "$BAR" | grep -oE "loss=[^,]+" | awk 'NR==1{f=$0} {a[NR]=$0} END{print "    " f " | " a[int(NR/2)+1] " | " a[NR]}'
echo "  proposal_gold line: $(grep -ac "proposal_gold=identity" "$OUT/eb19.log") (must be >= 1)"
echo "  junction column AUC (in-run, per column; must RISE):"
grep -aoE "gold trigger beats false trigger in [0-9.]+ of [0-9]+ pairs.*over [0-9]+ natural groups" "$OUT/eb19.log" | sed 's/^/    /'
echo "  record negative-instance lines (must be 0 -- not in eb19): $(grep -ac "record negative instances" "$OUT/eb19.log")"
echo "==================================================="

if [ "$RESCUE" -ne 0 ]; then
  echo "[smoke] *** PUBLISH FAILED -- holding the box; the hard-deadline watchdog still terminates it ***"
  sleep infinity
fi
