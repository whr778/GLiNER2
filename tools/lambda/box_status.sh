#!/bin/bash
# ONE status line per poll for a Lambda job, from the LAMBDA API and the BOX itself -- never from the launcher's log
# alone. The first eb22c launch crashed at startup and its box sat idle for 2 h while a watcher waited for a launcher
# line that never came (2026-10-19). This asks the two sources that cannot lie: is an instance billing, and is its
# GPU busy.
#
#   NAME=eb22c LAUNCH_LOG=/Volumes/Development/tmp/launch_eb22c/launch.log EVERY=600 bash tools/lambda/box_status.sh
#
# Prints ALERT lines for: a billing box whose GPU is idle twice running, a FATAL/Traceback/rescue hold in the box log,
# and a box that is gone while the job had not reported DONE.
set -uo pipefail
NAME=${NAME:?NAME=<instance name>}
EVERY=${EVERY:-600}
LAUNCH_LOG=${LAUNCH_LOG:-}
KEY=$HOME/.ssh/id_ed25519
LK=$(sed -n 's/^export LAMBDA_API_KEY=["]\{0,1\}\([^"]*\)["]\{0,1\}$/\1/p' ~/.zshrc)
idle=0
seen=0
box() { ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR "ubuntu@$1" "$2" 2>/dev/null; }
while true; do
  now=$(date -u +%H:%MZ)
  inst=$(curl -s -m 30 -u "$LK:" https://cloud.lambda.ai/api/v1/instances \
    | uv run --no-project python -c "import json,sys; d=[i for i in json.load(sys.stdin)['data'] if i.get('name')=='$NAME']; print(' '.join(f\"{i['status']} {i.get('ip','-')} {i['instance_type']['name']}\" for i in d))" 2>/dev/null || echo APIERR)
  if [ "$inst" = "APIERR" ]; then
    echo "[$now] ALERT $NAME: Lambda API did not answer -- billing state UNKNOWN"
  elif [ -z "$inst" ]; then
    last=$([ -n "$LAUNCH_LOG" ] && tail -1 "$LAUNCH_LOG" | cut -c1-120)
    [ "$seen" -eq 1 ] && echo "[$now] $NAME: instance GONE (terminated) | launcher: $last" && exit 0
    echo "[$now] $NAME: no instance yet (not billing) | launcher: $last"
  else
    seen=1
    set -- $inst
    if [ "$1" != "active" ]; then
      echo "[$now] $NAME: $1 ($3)"
    else
      s=$(box "$2" 'g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tr -d " ");
        L=~/box.log; p=$(grep -ao "samples/s=[0-9.]*\|epoch [0-9.]*\|[0-9]*/[0-9]* \[[0-9:]*<[0-9:]*" ~/event_base.log ~/scaling/*.log 2>/dev/null | tail -1);
        e=$(grep -aE "FATAL|Traceback|rescue|holding|OutOfMemory|DONE|chosen|\[scale\] [ABC] " $L 2>/dev/null | tail -2 | tr "\n" " " | cut -c1-220);
        echo "gpu=$g|prog=$p|log=$e"')
      g=${s%%|*}; util=${g#gpu=}; util=${util%%,*}
      echo "[$now] $NAME active $3 | ${s:-box unreachable}"
      if [ -n "$s" ] && [ "${util:-0}" -le 2 ] 2>/dev/null; then idle=$((idle+1)); else idle=0; fi
      [ "$idle" -ge 2 ] && echo "[$now] ALERT $NAME: GPU idle for $idle polls while BILLING"
      echo "$s" | grep -qE "FATAL|Traceback|rescue|holding|OutOfMemory" && echo "[$now] ALERT $NAME: failure/hold line in box.log"
    fi
  fi
  sleep "$EVERY"
done
