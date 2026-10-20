#!/bin/bash
# DO 2-GPU RANKS DRIFT APART? eb22c on 2 GPUs, two arms on ONE box, each with the [ddp-sync] weight-checksum gate
# at every eval (every 100 steps).
#
# 2026-10-20: on 2x H100 at the office the two ranks' in-run evals of the SAME 1,500 records "differ wildly", and
# eb22c's events collapsed in epoch 2 there, while every earlier run was single-GPU (no DDP). The trainer wraps DDP
# with static_graph=True, which does not all-reduce a parameter it recorded as unused -- heads only some batches
# touch could then drift per rank.
#     A  eb22c as configured (ddp_static_graph true, the default)
#     B  eb22c with ddp_static_graph false + ddp_find_unused_parameters true
#     C  eb22c with ddp_static_graph false ALONE (find_unused false): the minimal fix, if it holds
#     D  A's settings (static_graph true) on the FIXED trainer: every trainable parameter touched every step
# MEASURED 2026-10-20 (2x H100): A drifts 25 of 33 groups by step 100; B and C crash at step 1. ARMS picks the arms.
# A logging DIFFER and B identical proves the drift AND the fix. First runs the Linux-only DDP tests.
#
#   JOB="bash tools/lambda/ddp_sync_probe.sh" (launched by ddp_sync_launch.sh)
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
PY=./.venv/bin/python
OUT=$HOME/ddp_probe
mkdir -p "$OUT"
source tools/lambda/_publish.sh
echo "[ddp] ===== START $(date -u) | $(git rev-parse --short HEAD) | $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ') ====="

ARMS=${ARMS:-ABC}   # letters, e.g. ARMS=D: survives the launcher's nested ssh quoting
timeout 900 $PY -m pytest -q tests/test_trainer_distributed_integration.py tests/training/test_ddp_weight_sync_check.py \
  tests/training/test_ddp_full_touch.py \
  > "$OUT/ddp_tests.log" 2>&1
echo "[ddp] DDP tests rc=$? : $(tail -1 "$OUT/ddp_tests.log")"

run_arm() {  # name static_graph find_unused
  local name=$1 cfg=tools/train/config/base/ddp-probe-$1.yaml
  $PY - "$cfg" "$name" "$2" "$3" <<'PY'
import sys, yaml
dst, name, static, unused = sys.argv[1], sys.argv[2], sys.argv[3] == "true", sys.argv[4] == "true"
c = yaml.safe_load(open("tools/train/config/base/eb22c.yaml"))
c["training"].update(output_dir=f"./out/ddp-probe-{name}", experiment_name=f"ddp-probe-{name}", max_steps=600,
                     max_train_samples=20000, eval_strategy="steps", eval_steps=100, max_eval_samples=50,
                     save_best=False, logging_steps=20, ddp_static_graph=static, ddp_find_unused_parameters=unused)
yaml.safe_dump(c, open(dst, "w"), sort_keys=False, allow_unicode=True)
PY
  timeout 2400 $PY -m torch.distributed.run --standalone --nproc_per_node=2 tools/train/train.py --config "$cfg" \
    > "$OUT/$name.log" 2>&1
  echo "[ddp] arm $name (static_graph=$2, find_unused=$3) rc=$?"
  grep -a "\[ddp-sync\]" "$OUT/$name.log" | sed "s/^/[ddp] $name: /" | cut -c1-300
}
settings() { case $1 in A|D) echo "true false" ;; B) echo "false true" ;; C) echo "false false" ;; esac; }
for arm in $(echo "$ARMS" | grep -o .); do
  run_arm "$arm" $(settings "$arm")
done
publish "ddp-probe" "$OUT/ddp_tests.log" $(for a in $(echo "$ARMS" | grep -o .); do echo "$OUT/$a.log"; done) || echo "[ddp] *** publish failed ***"
echo "[ddp] ===== DONE $(date -u) ====="
