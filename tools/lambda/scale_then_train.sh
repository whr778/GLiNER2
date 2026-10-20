#!/bin/bash
# SCALE, THEN TRAIN, on ONE box: a short scaling check picks the per-step batch for THIS card, then the full run.
#
# An 80 GB H100 is only faster if the job is scaled to it. The eb22b smoke on an A100-40GB ran 9.9 samples/s at
# ~57% GPU utilisation with batch 4 x accumulation 4: the GPU waits on the main process, so a bigger card at the
# same per-step batch mostly waits faster. Phase A measures, on THIS card, three variants that train the SAME
# thing (effective batch 16; checkpointing does not change the maths):
#     A  batch 4 x accum 4, gradient checkpointing ON   (the config as written)
#     B  batch 8 x accum 2, gradient checkpointing ON
#     C  batch 8 x accum 2, gradient checkpointing OFF
# each for $STEPS steps on a seeded 20,000-record draw of the real mix (same document lengths, short prep), eval
# off. Phase B takes the FASTEST variant whose peak GPU memory stayed under $MEM_CAP of the card, writes the run
# config, and hands off to event_base_run.sh (model pushed first, then metrics/logs). Scaling results publish
# before the full run starts, whatever happens next.
#
#   CFG=tools/train/config/base/eb22c.yaml OUTDIR=./out/eb22c REPO=whr778/gliner2-eb22c DEST=eb22c bash tools/lambda/scale_then_train.sh
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1
PY=./.venv/bin/python
CFG=${CFG:?CFG=<base config>}
DEST=${DEST:?DEST=<run-logs folder>}
STEPS=${STEPS:-250}
MEM_CAP=${MEM_CAP:-0.88}
MIN_RATE=${MIN_RATE:-8}   # samples/s; the eb22b smoke ran 9.9 on an A100-40GB
OUT=$HOME/scaling
mkdir -p "$OUT"
source tools/lambda/_publish.sh
echo "[scale] ===== START $(date -u) | $CFG | steps $STEPS | memory cap $MEM_CAP ====="

run_variant() {  # name batch accum checkpointing
  local name=$1 bs=$2 acc=$3 ckpt=$4
  local cfg; cfg=$(dirname "$CFG")/scale-$name.yaml
  $PY - "$CFG" "$cfg" "$STEPS" "$bs" "$acc" "$ckpt" <<'PY'
import sys, yaml
src, dst, steps, bs, acc, ckpt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] == "on"
c = yaml.safe_load(open(src))
t = c["training"]
t.update(max_steps=steps, eval_strategy="no", save_best=False, logging_steps=10, output_dir="./out/scaling",
         experiment_name="scaling", batch_size=bs, gradient_accumulation_steps=acc, gradient_checkpointing=ckpt,
         max_train_samples=20000, max_eval_samples=50)
yaml.safe_dump(c, open(dst, "w"), sort_keys=False, allow_unicode=True)
PY
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 5 > "$OUT/$name.gpu.csv" 2>&1 &
  local smi=$!
  local start; start=$(date +%s)
  timeout 2400 $PY -u tools/train/train.py --config "$cfg" > "$OUT/$name.log" 2>&1
  local rc=$?
  kill $smi 2>/dev/null
  rm -f "$cfg"
  echo "[scale] $name (batch $bs x accum $acc, checkpointing $ckpt): rc=$rc in $(( $(date +%s) - start )) s"
}

run_variant A 4 4 on
run_variant B 8 2 on
run_variant C 8 2 off

# Read each variant: samples/s at its last progress line, peak memory, mean utilisation while training.
$PY - "$OUT" "$MEM_CAP" > "$OUT/scaling.json" <<'PY'
import json, re, sys
out, cap = sys.argv[1], float(sys.argv[2])
spec = {"A": (4, 4, True), "B": (8, 2, True), "C": (8, 2, False)}
res = {}
for name, (bs, acc, ckpt) in spec.items():
    log = open(f"{out}/{name}.log", errors="replace").read().replace("\r", "\n")
    rates = [float(x) for x in re.findall(r"samples/s=([\d.]+)", log)]
    rows = [l.split(",") for l in open(f"{out}/{name}.gpu.csv") if l.count(",") == 2]
    used = [int(r[0]) for r in rows if r[0].strip().isdigit()]
    total = max((int(r[1]) for r in rows if r[1].strip().isdigit()), default=0)
    util = [int(r[2]) for r in rows if r[2].strip().isdigit() and int(r[2]) > 0]
    # The gate reads torch's peak ALLOCATED memory from the progress line; nvidia-smi's figure (smi_peak_mib) counts
    # the caching allocator's reserve and refused all three variants of launch 3, one at 98% that never ran out.
    alloc = [float(x) * 1024 for x in re.findall(r"peak_alloc_gb=([\d.]+)", log)]
    failed = "OutOfMemory" in log or "Traceback" in log or not rates or not alloc
    res[name] = {"batch_size": bs, "gradient_accumulation_steps": acc, "gradient_checkpointing": ckpt,
                 "samples_per_s": rates[-1] if rates else None, "peak_alloc_mib": round(max(alloc, default=0)),
                 "smi_peak_mib": max(used, default=0), "total_mib": total,
                 "mean_util": round(sum(util) / len(util)) if util else None, "failed": failed,
                 "fits": (not failed) and total > 0 and max(alloc, default=0) <= cap * total}
ok = [n for n in res if res[n]["fits"]]
choice = max(ok, key=lambda n: res[n]["samples_per_s"]) if ok else None
print(json.dumps({"variants": res, "chosen": choice, "memory_cap": cap}, indent=1))
PY
cat "$OUT/scaling.json"
publish "$DEST/scaling" "$OUT/scaling.json" "$OUT"/*.log "$OUT"/*.gpu.csv || echo "[scale] *** scaling results did not publish (continuing to the run) ***"

CHOSEN=$($PY -c "import json; print(json.load(open('$OUT/scaling.json'))['chosen'])")
# NO VARIANT RAN = THE JOB CANNOT RUN. The first eb22c launch fell back to A after all three died at startup, ran the
# full job into the same error, and then held the box idle for rescue (2026-10-19). Exit; box_run.sh terminates.
if [ "$CHOSEN" = "None" ]; then
  echo "[scale] *** FATAL: no variant trained (see A/B/C logs in $DEST/scaling) -- not starting the full run ***"
  exit 3
fi
# TOO SLOW = SOMETHING IS WRONG, NOT A LONG RUN. Launch 2 ran 2.9 samples/s on an H100 (cuDNN SDPA recompiling per
# length, GPU 2% busy): 5 epochs would have run into the 46 h timeout. Stop rather than pay for it.
RATE=$($PY -c "import json; s=json.load(open('$OUT/scaling.json')); print(s['variants'][s['chosen']]['samples_per_s'])")
if ! $PY -c "import sys; sys.exit(0 if $RATE >= $MIN_RATE else 1)"; then
  echo "[scale] *** FATAL: best variant $CHOSEN runs $RATE samples/s < MIN_RATE $MIN_RATE -- not starting the full run ***"
  exit 3
fi
RUN_CFG=$(dirname "$CFG")/$(basename "$CFG" .yaml)-run.yaml
$PY - "$CFG" "$RUN_CFG" "$OUT/scaling.json" <<'PY'
import json, sys, yaml
src, dst, scaling = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
v = scaling["variants"][scaling["chosen"]]
c = yaml.safe_load(open(src))
c["training"].update(batch_size=v["batch_size"], gradient_accumulation_steps=v["gradient_accumulation_steps"],
                     gradient_checkpointing=v["gradient_checkpointing"])
yaml.safe_dump(c, open(dst, "w"), sort_keys=False, allow_unicode=True)
print(f"[scale] chosen {scaling['chosen']}: batch {v['batch_size']} x accum {v['gradient_accumulation_steps']}, "
      f"checkpointing {v['gradient_checkpointing']} ({v['samples_per_s']} samples/s, peak allocated {v['peak_alloc_mib']}/{v['total_mib']} MiB) -> {dst}")
PY
echo "[scale] ===== PHASE B: full run $(date -u) ====="
exec env CFG="$RUN_CFG" bash tools/lambda/event_base_run.sh
