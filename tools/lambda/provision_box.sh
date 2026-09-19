#!/bin/bash
# Bring a LAUNCHED Lambda instance to "running the job", and refuse to continue past any
# step that did not work.
#
# THIS FILE EXISTS BECAUSE PROVISIONING WAS DONE AD-HOC TWICE AND FAILED BOTH TIMES.
# 2026-09-17, attempt two of the pool A/B: the credential block printed `no hf token` and
# exited 1, the caller did not check it, bootstrap then died on `Illegal header value
# b'Bearer '`, and the job started against a box with no data. The box terminated itself
# correctly on box_run.sh's own token pre-flight, ~13 minutes billed. NOTHING WAS WRONG WITH
# THE BOX. The provisioner just did not stop when a step failed.
#
# TWO CREDENTIAL TRAPS, both previously paid for:
#   1. THERE ARE TWO HF TOKENS AND THEY DIFFER. `$HF_TOKEN` in the shell is full access;
#      `~/.cache/huggingface/token` is READ-ONLY and cannot create model repos. Copy the
#      ENV VAR. There is no `~/.hf_token` on the laptop -- that path exists only on boxes,
#      and reading it here is what wrote an empty token.
#   2. THE LOGIN GREETING GOES TO STDOUT on this account and has corrupted `.lambda_key`
#      before, billing two boxes idle. Every credential is written by a heredoc into a
#      file, never piped through a login shell, and every one is verified after writing.
#
#   ID=<instance id> JOB="bash tools/lambda/pool_ab.sh" CFG=<config> CKPT=<hf ckpt> \
#     bash tools/lambda/provision_box.sh
set -uo pipefail
ID=${ID:?instance id required}
JOB=${JOB:?job command required}
CFG=${CFG:-}
CKPT=${CKPT:-}
BRANCH=${BRANCH:-merge/main-20260805}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
JOB_TIMEOUT=${JOB_TIMEOUT:-14400}
HARD_DEADLINE=${HARD_DEADLINE:-18000}

die() { echo "[prov] *** FATAL: $* ***" >&2; exit 1; }
[ -f "$KEY" ] || die "no ssh key at $KEY"
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set in this shell -- it is the full-access
    token and the only one that can create a model repo. Do NOT substitute
    ~/.cache/huggingface/token, which is read-only."
[ -n "${LAMBDA_API_KEY:-}" ] || die "LAMBDA_API_KEY is not set -- the box could not
    terminate itself, which is the one failure that bills until someone notices."

# PROVE THE TOKEN WRITES BEFORE AN INSTANCE IS LAUNCHED, not after. A full-access token
# EXPIRED mid-session on 2026-09-17; the on-box pre-flight caught it, but only after a box
# had been launched, booted and bootstrapped. Checking here costs a second and no money.
#
# NOTE FOR ANYONE DEBUGGING A "WRONG TOKEN" HERE: a non-interactive shell does NOT source
# ~/.zshrc, so an `export HF_TOKEN=` line there is invisible to this script -- it inherits
# whatever the parent process had at ITS startup, which can be hours stale. Pass the token
# explicitly (HF_TOKEN=$(...) bash tools/lambda/provision_box.sh) rather than trusting the
# environment to be current.
python3 - <<'TOKCHK' || die "HF_TOKEN does not authenticate -- refusing to launch a box that
    would bootstrap, fail to publish, and have to be terminated."
import os, sys
try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit(0)                      # cannot check here; the box pre-flight still will
try:
    who = HfApi(token=os.environ["HF_TOKEN"].strip()).whoami()["name"]
except Exception as exc:
    print(f"[prov] token rejected: {str(exc).splitlines()[0][:100]}")
    sys.exit(1)
print(f"[prov] HF_TOKEN authenticates as {who}")
TOKCHK

# ServerAliveInterval keeps a long bootstrap from dying to an idle NAT timeout, which is
# how a laptop network blip once returned "job did not start" for a box that was training.
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"

terminate_and_die() {
  echo "[prov] terminating $ID rather than leaving it billing"
  curl -s -u "$LAMBDA_API_KEY:" -X POST \
    https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$ID\"]}" >/dev/null
  die "$*"
}

echo "[prov] waiting for $ID to become active"
IP=""
for _ in $(seq 1 60); do
  IP=$(curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances \
       | python3 -c "
import sys,json
for x in json.load(sys.stdin)['data']:
    if x['id']=='$ID' and x['status']=='active': print(x['ip'])")
  [ -n "$IP" ] && break
  sleep 20
done
[ -n "$IP" ] || die "instance never became active"
echo "[prov] active at $IP"

for _ in $(seq 1 40); do $SSH ubuntu@$IP true 2>/dev/null && break; sleep 15; done
$SSH ubuntu@$IP true 2>/dev/null || terminate_and_die "ssh never came up"

# ECC PRE-FLIGHT. A faulty A10 with 126 uncorrected errors cost a relaunch on 2026-09-16;
# it trains, it just produces nonsense.
ECC=$($SSH ubuntu@$IP 'nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader' 2>/dev/null | tr -d ' ')
echo "[prov] ECC uncorrected: ${ECC:-unknown}"
case "$ECC" in
  0|"[N/A]"|"") : ;;
  *) terminate_and_die "$ECC uncorrected ECC errors -- not training on this card";;
esac

# CREDENTIALS. Written by heredoc, verified on the box, and the exit status is CHECKED --
# which is the single thing whose absence caused this file to exist.
$SSH ubuntu@$IP "bash -s" <<CREDS || terminate_and_die "credentials did not land"
set -e
printf '%s' '$LAMBDA_API_KEY' > ~/.lambda_key && chmod 600 ~/.lambda_key
printf '%s' '$ID'             > ~/.instance_id
printf '%s' '$HF_TOKEN'       > ~/.hf_token   && chmod 600 ~/.hf_token
grep -qE '^[0-9a-f]{32}\$' ~/.instance_id || { echo "instance id corrupt"; exit 1; }
grep -qE '^hf_' ~/.hf_token                || { echo "hf token missing or corrupt"; exit 1; }
[ -s ~/.lambda_key ]                       || { echo "lambda key empty"; exit 1; }
echo "[prov] credentials written and verified"
CREDS

# PROVE the token actually authenticates, on the box, before spending on a bootstrap. A
# non-empty token that is expired or read-only fails at PUBLISH time -- after the GPU hours.
$SSH ubuntu@$IP "bash -s" <<'AUTH' || terminate_and_die "HF token does not authenticate on the box"
set -e
cd ~
python3 -c "
import sys
try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit(0)   # pre-venv box; bootstrap installs it and _publish verifies later
print('[prov] hub whoami:', HfApi().whoami(token=open('/home/ubuntu/.hf_token').read().strip())['name'])"
AUTH

$SSH ubuntu@$IP "bash -s" <<SETUP || terminate_and_die "repo checkout failed"
set -e
cd ~ && rm -rf gliner2
git clone -q https://github.com/whr778/GLiNER2.git gliner2
cd gliner2 && git checkout -q $BRANCH
echo "[prov] repo at \$(git log --oneline -1)"
SETUP

# STOP 4: arm the box-side idle guard BEFORE the bootstrap, so every laptop-side failure
# from here on is covered. Stops 1-3 all live inside box_run.sh and none of them exists
# until the job starts; this one does. Non-fatal if it cannot be armed -- it only covers a
# window the other three already handle in the normal case -- but the arming is VERIFIED by
# asking the box, not by trusting an ssh exit code.
IDLE_GRACE=${IDLE_GRACE:-1200}
$SSH ubuntu@$IP "bash -lc 'cd ~/gliner2 && IDLE_GRACE=$IDLE_GRACE nohup setsid bash tools/lambda/idle_guard.sh > ~/idle_guard.log 2>&1 < /dev/null & disown'" >/dev/null 2>&1
if $SSH ubuntu@$IP 'pgrep -f "tools/lambda/idle_guard\.sh" >/dev/null' 2>/dev/null; then
  echo "[prov] idle guard armed (${IDLE_GRACE}s)"
else
  echo "[prov] WARNING: idle guard NOT armed -- a dropped link during job start leaves this box unwatched"
fi

echo "[prov] bootstrap $(date -u)"
$SSH ubuntu@$IP "bash -lc 'cd ~/gliner2 && CFG=$CFG CKPT=$CKPT bash tools/lambda/bootstrap_box.sh'" \
  2>&1 | grep -v "^Restored session:" | tail -20
# PIPESTATUS, not $? -- the pipe through grep/tail always succeeds and hid the bootstrap
# failure that started all this.
[ "${PIPESTATUS[0]}" -eq 0 ] || terminate_and_die "bootstrap failed"

echo "[prov] starting job $(date -u)"
$SSH ubuntu@$IP "bash -lc 'cd ~/gliner2 && JOB=\"$JOB\" JOB_TIMEOUT=$JOB_TIMEOUT HARD_DEADLINE=$HARD_DEADLINE nohup bash tools/lambda/box_run.sh > ~/box.log 2>&1 & disown'" \
  || {
    # DO NOT TRUST THE TRANSPORT'S WORD. The job is detached with nohup+disown, so an SSH that
    # drops AFTER it starts returns non-zero and looks identical to one that never ran it.
    # On 2026-09-18 a `client_loop: send disconnect: Broken pipe` on the LAPTOP terminated two
    # A100s that were 20 minutes into training at step 553 -- the network blip killed the run,
    # which is the exact failure the three-stop design exists to prevent. Ask the BOX whether
    # the runner is alive before believing it is not.
    # THREE outcomes, not two. The old probe ran ONE ssh over the same link that had just
    # failed and treated "could not connect" as "no runner" -- so on 2026-09-19 it printed
    # "verified on the box" for two boxes it never reached, and terminated both. A gate
    # whose two failure modes are indistinguishable cannot verify anything. Terminate only
    # on an answer FROM THE BOX; never on silence.
    echo "[prov] ssh returned non-zero starting the job -- asking the box before terminating"
    verdict=""
    for attempt in 1 2 3 4 5; do
      sleep 20
      verdict=$($SSH ubuntu@$IP 'if pgrep -f "tools/lambda/box_run\.sh" >/dev/null || [ -s ~/box.log ]; then echo RUNNER_ALIVE; else echo RUNNER_DEAD; fi' 2>/dev/null \
                 | grep -E '^RUNNER_(ALIVE|DEAD)$' | tail -1)
      [ -n "$verdict" ] && break
      echo "[prov] probe $attempt: no answer from the box (transport), retrying"
    done
    case "$verdict" in
      RUNNER_ALIVE)
        echo "[prov] the job IS running; the ssh transport dropped, not the job -- continuing" ;;
      RUNNER_DEAD)
        terminate_and_die "job did not start (the box ANSWERED: no runner, no box.log)" ;;
      *)
        echo "[prov] *** UNREACHABLE after 5 probes -- NOT terminating $ID blind ***"
        echo "[prov] *** if the job started, its own three stops apply; if it did not, the"
        echo "[prov] *** box-side idle guard terminates within ${IDLE_GRACE}s of arming"
        echo "[prov] *** instance $ID at $IP -- check it when the link is back"
        exit 7 ;;
    esac
  }
echo "[prov] RUNNING -- ip=$IP  id=$ID"
