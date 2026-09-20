#!/bin/bash
# Poll for GPU capacity, launch the FIRST acceptable card, and hand straight to
# provision_box.sh. Lambda goes to zero capacity across every instance type for stretches,
# so "launch it now" is not always an option and sitting on the API by hand is not either.
#
# PREFERENCE ORDER IS MEASURED, NOT ALPHABETICAL. On this workload the A100 runs 2.26 it/s
# against the A10's 1.33 -- and per dollar it is still marginally ahead (1.14 vs 1.03 it/s
# per $). So A100 first, A10 as the fallback, which is the opposite of what the hourly rate
# suggests.
#
# It launches ONE instance and stops. The job it hands to terminates itself by three
# independent stops, so the downside of an unattended launch is bounded by those, not by
# whoever is awake.
#
#   NAME=pool-ab-3 JOB="bash tools/lambda/pool_ab.sh" CFG=... CKPT=... \
#     bash tools/lambda/launch_when_available.sh
set -uo pipefail
NAME=${NAME:?instance name required}
JOB=${JOB:?job command required}
TYPES=${TYPES:-"gpu_1x_a100_sxm4 gpu_1x_a10"}
SSH_KEY_NAME=${SSH_KEY_NAME:-gliner2-mac}
POLL=${POLL:-180}
MAX_WAIT=${MAX_WAIT:-21600}     # 6h of looking, then give up rather than poll forever
: "${LAMBDA_API_KEY:?not set}"

deadline=$(( $(date +%s) + MAX_WAIT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  for t in $TYPES; do
    # REGION PREFERENCE. This took caps[0] -- whichever region the API happened to list
    # first -- so arms of one A/B landed on different continents and boxes turned up in asia
    # when the work is driven from the US. REGIONS is a space-separated preference list of
    # prefixes; the first match wins, and an empty match falls back to any region with
    # capacity rather than refusing to launch.
    REGION=$(REGIONS="${REGIONS:-us-}" curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instance-types \
      | REGIONS="${REGIONS:-us-}" python3 -c "
import os, sys, json
d = json.load(sys.stdin)['data'].get('$t') or {}
caps = [c['name'] for c in (d.get('regions_with_capacity_available') or [])]
prefs = os.environ.get('REGIONS', '').split()
for pref in prefs:
    hit = [c for c in caps if c.startswith(pref)]
    if hit:
        print(hit[0]); break
else:
    print(caps[0] if caps else '')")
    [ -z "$REGION" ] && continue

    echo "[launch] $t has capacity in $REGION -- launching $NAME $(date -u)"
    RESP=$(curl -s -u "$LAMBDA_API_KEY:" -X POST \
      https://cloud.lambda.ai/api/v1/instance-operations/launch \
      -H "Content-Type: application/json" \
      -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$t\",\"ssh_key_names\":[\"$SSH_KEY_NAME\"],\"name\":\"$NAME\",\"quantity\":1}")
    ID=$(printf '%s' "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ids = (d.get('data') or {}).get('instance_ids') or []
print(ids[0] if ids else '')")
    if [ -z "$ID" ]; then
      # Capacity is a race: the listing said yes and someone else took it first.
      echo "[launch] launch refused: $(printf '%s' "$RESP" | head -c 200)"
      continue
    fi
    echo "[launch] launched $ID ($t in $REGION); provisioning"
    exec env ID="$ID" JOB="$JOB" bash tools/lambda/provision_box.sh
  done
  sleep "$POLL"
done
echo "[launch] *** gave up after ${MAX_WAIT}s with no capacity ***" >&2
exit 1
