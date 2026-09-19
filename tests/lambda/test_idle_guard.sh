#!/usr/bin/env bash
# The guard must FIRE when no runner exists and STAND DOWN when one does. Only `curl` is
# stubbed; the script under test is the real one. The runner pattern is a unique per-run
# token so the host's own processes cannot decide the answer -- the laptop's open ssh
# clients to other boxes matched a bare pattern and made this test lie.
set -uo pipefail
cd "$(dirname "$0")/../.."
GUARD=$PWD/tools/lambda/idle_guard.sh
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home" "$WORK/bin"
printf '%s' 'deadbeefdeadbeefdeadbeefdeadbeef' > "$WORK/home/.instance_id"
printf '%s' 'fake-key'                         > "$WORK/home/.lambda_key"
printf '#!/usr/bin/env bash\necho "CURL_CALLED $*" >> "$CURL_LOG"\n' > "$WORK/bin/curl"
chmod +x "$WORK/bin/curl"
export CURL_LOG=$WORK/curl.log; : > "$CURL_LOG"
TOKEN=guardtest_$$_$RANDOM
fail=0
run() { HOME=$WORK/home PATH=$WORK/bin:$PATH IDLE_GRACE=1 RUNNER_PATTERN=$TOKEN bash "$GUARD"; }

# 1. no runner -> terminates, and the call names THIS instance
run > "$WORK/out1" 2>&1
grep -q "CURL_CALLED" "$CURL_LOG" \
  && grep -q "deadbeefdeadbeefdeadbeefdeadbeef" "$CURL_LOG" \
  && grep -q "instance-operations/terminate" "$CURL_LOG" \
  && echo "PASS fires when no runner is alive" || { echo "FAIL no-runner case"; cat "$WORK/out1" "$CURL_LOG"; fail=1; }

# 2. runner alive -> stands down, no API call at all
: > "$CURL_LOG"
printf '#!/bin/sh\nsleep "$1"\n' > "$WORK/bin/$TOKEN"; chmod +x "$WORK/bin/$TOKEN"
"$WORK/bin/$TOKEN" 20 & RUNNER=$!
sleep 0.3
run > "$WORK/out2" 2>&1
kill $RUNNER 2>/dev/null
[ ! -s "$CURL_LOG" ] && grep -q "standing down" "$WORK/out2" \
  && echo "PASS stands down while a runner is alive" || { echo "FAIL runner-alive case -- the guard would have killed a live run"; cat "$WORK/out2" "$CURL_LOG"; fail=1; }

# 3. corrupt instance id -> refuses, no API call. Lambda has no metadata service, so a 404
#    HTML page is a real thing that can land in that file.
: > "$CURL_LOG"; printf '%s' '<html>404</html>' > "$WORK/home/.instance_id"
run > "$WORK/out3" 2>&1
[ ! -s "$CURL_LOG" ] && grep -q "no valid instance id" "$WORK/out3" \
  && echo "PASS refuses on a corrupt instance id" || { echo "FAIL corrupt-id case"; cat "$WORK/out3"; fail=1; }

# 4. the SHIPPED default stays path-anchored -- nobody loosens it back to a bare name
grep -q "RUNNER_PATTERN=\${RUNNER_PATTERN:-'tools/lambda/box_run\\\\.sh'}" "$GUARD" \
  && echo "PASS default pattern is path-anchored" || { echo "FAIL default pattern is not the anchored one"; fail=1; }

exit $fail
