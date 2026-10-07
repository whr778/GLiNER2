#!/bin/bash
# MENU THRESHOLD job (runs ON the box, under box_run.sh): re-pick each menudose arm's decision threshold
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
OUTS=""
for arm in treatment control; do
  $PY -u tools/train/sweep_menu_threshold.py --config tools/train/config/ab/menudose-$arm.yaml \
      --checkpoint whr778/gliner2-menudose-$arm --menu app:news55 --out "$HOME/menudose_$arm.sweep.json" 2>&1 | tee -a "$LOG"
  [ -f "$HOME/menudose_$arm.sweep.json" ] && OUTS="$OUTS $HOME/menudose_$arm.sweep.json"
done
echo "[mt] done $(date -u)" | tee -a "$LOG"
publish menudose_threshold $OUTS "$LOG" || { echo "[mt] *** PUBLISH FAILED -- holding for rescue ***"; sleep infinity; }
