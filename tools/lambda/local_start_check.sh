#!/bin/bash
# PRE-LAUNCH GATE: the REAL train.py on the REAL config, 3 optimizer steps on this laptop, before any box is bought.
#
# The first eb22c launch died at startup on a config error (a stale corpus key in partial_annotation) that every
# local gate passed, because none of them ran train.py (2026-10-19). This runs the whole startup path -- data load,
# label maps, negative pools, model build, collator, forward, backward -- on 200 records; bf16 is off because the
# laptop cannot run it. Passes only on rc 0 AND a logged training rate.
#
#   bash tools/lambda/local_start_check.sh tools/train/config/base/eb22c.yaml
set -uo pipefail
CFG=${1:?usage: local_start_check.sh <config>}
DIR=/Volumes/Development/tmp/start_check/$(basename "$CFG" .yaml)
mkdir -p "$DIR"
TMP=$(dirname "$CFG")/.start-check-$(basename "$CFG")
uv run python - "$CFG" "$TMP" "$DIR" <<'PY'
import sys, yaml
src, dst, out = sys.argv[1:]
c = yaml.safe_load(open(src, encoding="utf-8"))
c["training"].update(max_steps=3, eval_strategy="no", save_best=False, logging_steps=1, output_dir=f"{out}/out",
                     experiment_name="start-check", batch_size=2, gradient_accumulation_steps=1, max_train_samples=200,
                     max_eval_samples=10, bf16=False, fp16=False)
yaml.safe_dump(c, open(dst, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
PY
PYTHONUNBUFFERED=1 uv run tools/train/train.py --config "$TMP" > "$DIR/train.log" 2>&1
rc=$?
rm -f "$TMP"
if [ "$rc" -ne 0 ] || ! grep -q "samples/s=" "$DIR/train.log"; then
  echo "[start-check] *** FAILED rc=$rc -- $CFG cannot start; see $DIR/train.log ***"
  grep -aE "Error|SystemExit|refus|\[pools\]" "$DIR/train.log" | tail -3
  exit 1
fi
echo "[start-check] OK: $CFG trained 3 steps locally ($(grep -ao 'loss=[0-9.]*' "$DIR/train.log" | tail -1))"
