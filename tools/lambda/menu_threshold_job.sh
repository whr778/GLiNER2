#!/bin/bash
# MENU THRESHOLD job (runs ON the box, under box_run.sh): re-pick each ARM's decision threshold
# under the news menu on sonnet55 val, then score the blind test once (MENU_SPEC.md s8). Results are
# published to whr778/gliner2-run-logs:menudose_threshold/ and verified; a failed publish HOLDS the box for
# rescue (the hard-deadline watchdog still terminates it). Returning lets box_run.sh terminate.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1
PY=./.venv/bin/python
LOG=$HOME/menu_threshold.log
source tools/lambda/_publish.sh
echo "[mt] start $(date -u)" | tee -a "$LOG"
# ARMS: COMMA-separated "<exp>-<arm>,..." (config tools/train/config/ab/<exp>-<arm>.yaml, model
# whr778/gliner2-<exp>-<arm>). Commas, not spaces: the job string is nested inside the provisioner's ssh
# quoting, and a quoted space-separated list broke it ("unexpected EOF while looking for matching").
# Each arm is PUBLISHED AS SOON AS IT FINISHES: the first run published only at the end, the job timeout
# killed the slow control sweep, and only a hand-published treatment result survived (2026-10-07).
ARMS=${ARMS:-menudose-treatment,menudose-control}
for a in ${ARMS//,/ }; do
  $PY -u tools/train/sweep_menu_threshold.py --config tools/train/config/ab/$a.yaml \
      --checkpoint whr778/gliner2-$a --menu app:news55 --out "$HOME/$a.sweep.json" 2>&1 | tee -a "$LOG"
  if [ -f "$HOME/$a.sweep.json" ]; then
    publish menu_threshold "$HOME/$a.sweep.json" "$LOG" || { echo "[mt] *** PUBLISH FAILED ($a) -- holding for rescue ***"; sleep infinity; }
  fi
done
echo "[mt] done $(date -u)" | tee -a "$LOG"
publish menu_threshold "$LOG" || { echo "[mt] *** PUBLISH FAILED -- holding for rescue ***"; sleep infinity; }
