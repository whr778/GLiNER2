#!/bin/bash
# MENU PROBE job (runs ON the box, under box_run.sh), 2026-10-08:
#  (2) a MENU-SIZE LADDER for dose 20 and the full recipe, each at its own chosen threshold -- why does full
#      win under the gold menu and lose under the 180-label news menu? gold and app:news55 exist already;
#      this adds widened:5, widened:20, corpus_full (entities ~15 / 30 / 74, events ~7 / 22 / 55);
#  (3) the dose-1 control's threshold re-pick on 0.6-0.9 only (an over-firing model's best sits high, and
#      high thresholds decode faster; the full grid hit the 4 h timeout twice).
# Each result is PUBLISHED as soon as it exists; a failed publish holds the box for rescue.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1
PY=./.venv/bin/python
LOG=$HOME/menu_probe.log
source tools/lambda/_publish.sh
echo "[probe] start $(date -u)" | tee -a "$LOG"
run() {  # <name> <args...>
  local name=$1; shift
  $PY -u tools/train/sweep_menu_threshold.py "$@" --out "$HOME/$name.json" 2>&1 | tee -a "$LOG"
  [ -f "$HOME/$name.json" ] && { publish menu_probe "$HOME/$name.json" "$LOG" || { echo "[probe] *** PUBLISH FAILED ($name) ***"; sleep infinity; }; }
}
LADDER="--test-menus widened:5 widened:20 corpus_full"
run ladder_menudose-treatment --config tools/train/config/ab/menudose-treatment.yaml --checkpoint whr778/gliner2-menudose-treatment \
    --menu app:news55 --threshold 0.5 $LADDER
run ladder_menuref-full --config tools/train/config/ab/menuref-full.yaml --checkpoint whr778/gliner2-menuref-full \
    --menu app:news55 --threshold 0.4 $LADDER
run repick_menudose-control --config tools/train/config/ab/menudose-control.yaml --checkpoint whr778/gliner2-menudose-control \
    --menu app:news55 --grid 0.6 0.7 0.8 0.9
echo "[probe] done $(date -u)" | tee -a "$LOG"
publish menu_probe "$LOG" || { echo "[probe] *** PUBLISH FAILED ***"; sleep infinity; }
