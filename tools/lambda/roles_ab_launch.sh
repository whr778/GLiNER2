#!/bin/bash
# OPTION 4 A/B -- launch one arm. Run it twice, once per arm, on two boxes.
#
#   ARM=control   bash tools/lambda/roles_ab_launch.sh
#   ARM=treatment bash tools/lambda/roles_ab_launch.sh
#
# THE ONE VARIABLE is whether data/cmnee_roles_ner is in the mix -- cmnee's event ARGUMENTS
# reframed as entity spans under namespaced Event<Role> labels. Everything else, including
# the schedule and precision, comes from the incumbent's own config.
#
# WHY THE CONTROL IS RETRAINED RATHER THAN REUSED. `whr778/gliner2-eb16-eventrecords-tr` IS
# this control's config -- but it was pushed 2026-09-16 11:27, and 5af9e00 ("the consistency
# loss clamp was a NO-OP in bf16, and it poisoned the whole model") landed 2026-09-17 10:45,
# with four other training-path commits in between. Comparing a fresh treatment against that
# model would credit the derived corpus with a loss fix. The corpora themselves have NOT
# drifted -- all 58 files predate the incumbent -- so code is the only reason to rebuild.
#
# SURVIVING THINGS GOING WRONG. Each box is autonomous once launched: it clones, trains,
# pushes the model, publishes metrics and logs, and terminates itself. This laptop can reboot
# or lose the network with no effect. On the box, `publish` retries six times over ~30 minutes
# and verifies against the Hub's file list, a start marker with the CODE COMMIT is published
# at minute zero, and the heartbeat plus a log tail go up every 30 minutes -- so a failure at
# hour 15 costs the last half hour, not the whole run. Three stops remain: the job timeout,
# the detached hard-deadline watchdog, and terminate on the normal path.
set -uo pipefail
ARM=${ARM:?ARM=control or ARM=treatment}
case "$ARM" in control|treatment) ;; *) echo "ARM must be control or treatment"; exit 2;; esac

export JOB_TIMEOUT=${JOB_TIMEOUT:-82800}     # 23h: the runner caps training at 20h, then pushes
export HARD_DEADLINE=${HARD_DEADLINE:-90000} # 25h absolute, whatever happens

JOB="CFG=tools/train/config/ab/roles-$ARM.yaml \
OUTDIR=./out/roles-$ARM \
REPO=whr778/gliner2-roles-$ARM \
DEST=roles_$ARM \
bash tools/lambda/event_base_run.sh"

# PRE-FLIGHT: a box has no data/, so every corpus must be fetchable from the Hub. The first
# launch of this A/B died in the data phase on BOTH arms ~12 minutes in -- scierc and
# paraloq_json had no hf_jsonl entry and `_fetch_corpus` returns SILENTLY when a corpus is
# unregistered. Both files were present on the laptop, which is why the configs looked fine.
# Free, local, and seconds; it runs before anything can be billed.
uv run python tools/train/check_corpora_fetchable.py \
    --config "tools/train/config/ab/roles-$ARM.yaml" --offline \
  || { echo "[roles-ab] *** REFUSING TO LAUNCH -- a corpus is unfetchable ***"; exit 3; }

echo "[roles-ab] arm=$ARM  job_timeout=${JOB_TIMEOUT}s  hard_deadline=${HARD_DEADLINE}s"
echo "[roles-ab] model -> whr778/gliner2-roles-$ARM"
echo "[roles-ab] logs  -> whr778/gliner2-run-logs : roles_$ARM/"
exec env NAME="roles-$ARM" JOB="$JOB" bash tools/lambda/launch_when_available.sh
