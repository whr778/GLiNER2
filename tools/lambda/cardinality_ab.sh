#!/bin/bash
# CARDINALITY A/B: does declaring a structure field SCALAR, when the corpus says it holds
# one filler, train a better record head than today's implicit ZERO_OR_MORE?
#
# Cardinality selects the loss -- `_scalar_field_nll` (softmax over candidates plus an
# explicit ABSENT) against `_list_field_bce` (each candidate independently). Measured
# across all of data/: 99.8% of structure-field occurrences hold exactly ONE filler, so
# today's regime trains almost every structure field as multi-label against its own data.
#
# ONE VARIABLE. The two configs differ in output_dir, experiment_name and the corpus --
# `casualty_loc_split` against `casualty_loc_split_card`, the same 23,744 documents with
# `fields: {f: {cardinality}}` stamped from the corpus's OWN train usage. Same base, same
# hyperparameters, same blind test.
#
# THE GATE, and it took three tries to get an instrument that can actually fail. Each arm
# prints ONE deterministic corpus-level line before training:
#
#   control    [composition]   cardinality casualty_report: NONE DECLARED -- ...
#   treatment  [composition]   cardinality casualty_report: optional_one=<n>
#
# If BOTH say NONE DECLARED, or both say the same thing, THE ARMS ARE NOT THE ARMS and the
# run measured nothing -- stop before reading a single F1.
#
# The two rejected versions are worth knowing, because both looked fine:
#   1. "control must read 0 scalar" -- WRONG. An ANCHOR is REQUIRED_ONE and therefore
#      scalar in both arms, so the control legitimately read `6 scalar, 15 list`.
#   2. `[records] compiled N field spec(s): X scalar, Y list` -- USELESS HERE. It samples
#      whichever batch a DataLoader worker saw first, so it is neither deterministic nor
#      per-arm, and with eval_strategy: epoch it can fire from the EVAL collate, where
#      `_schema_from_gold` rebuilds schemas carrying no cardinality at all. On 2026-09-14
#      it printed identical counts for both arms and I read that as "the treatment did not
#      apply" -- it was the instrument, not the treatment. The mechanism was fine, proven
#      through `collate_fn_train` afterwards.
#
# SURVIVING A NETWORK OUTAGE IS A DESIGN REQUIREMENT HERE, not a nicety. The disk dies
# with the instance and the box terminates on the normal path, so a publish that fails
# once loses the arm that paid for it. Three properties, in this order:
#
#   1. METRICS BEFORE MODEL. The metrics are the finding and are kilobytes; the model is
#      600MB and can fail on Hub quota (which it did, once, today). Publishing the small
#      artefact first means a quota failure cannot cost us the result.
#   2. EVERY PUBLISH RETRIES with backoff -- 6 attempts over ~30 minutes. A transient DNS
#      or TLS failure at minute 50 of a 60-minute arm is otherwise indistinguishable from
#      having never run it.
#   3. A FAILED PUBLISH IS LOUD AND HOLDS THE BOX. If both retries are exhausted the job
#      refuses to fall off its normal path, so the instance lives until the hard-deadline
#      watchdog kills it -- a window to rescue by hand. Losing a $4 model to save $1 of
#      idle is bad arithmetic, and the watchdog still bounds the downside.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1   # FA2 is a CORRECTNESS requirement on mmBERT in bf16

PY=./.venv/bin/python
OUT=$HOME/card_ab
DEST=${DEST:-cardinality_ab}
LOGREPO=whr778/gliner2-run-logs
RESCUE=0
mkdir -p "$OUT"

# Retry anything, with backoff. 6 attempts, ~30 minutes total.
retry() {
  local what=$1; shift
  local n
  for n in 1 2 3 4 5 6; do
    if "$@"; then
      [ "$n" -gt 1 ] && echo "[card] $what succeeded on attempt $n"
      return 0
    fi
    echo "[card] $what FAILED (attempt $n/6)"
    [ "$n" -lt 6 ] && sleep $((n * 120))
  done
  echo "[card] *** $what FAILED AFTER 6 ATTEMPTS ***"
  return 1
}

publish_metrics() {
  local arm=$1
  $PY - "$arm" "$DEST" <<'PY'
import os, sys
from huggingface_hub import HfApi
arm, dest = sys.argv[1], sys.argv[2]
out, repo = os.path.expanduser("~/card_ab"), "whr778/gliner2-run-logs"
api = HfApi(); api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
sent = []
for f in (f"{arm}.json", f"{arm}.log"):
    p = os.path.join(out, f)
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{dest}/{f}",
                        repo_id=repo, repo_type="dataset")
        sent.append(f"{dest}/{f}")
# Verify against the Hub's own file list rather than a clean return: upload_folder has
# returned successfully having written nothing, and it cost ~15h of A100 once.
on_hub = set(api.list_repo_files(repo, repo_type="dataset"))
missing = [f for f in sent if f not in on_hub]
raise SystemExit(f"*** NOT SAVED *** {missing}" if missing else 0)
PY
}

publish_model() {
  local arm=$1
  local ckpt="out/cardinality-$arm/best"
  if [ ! -d "$ckpt" ]; then
    echo "[card] no checkpoint at $ckpt -- nothing to publish"
    return 0
  fi
  $PY -u tools/train/push_to_hub.py --checkpoint "$ckpt" \
      --repo-id "whr778/gliner2-cardinality-$arm" --private 2>&1 | tail -4
}

for arm in control treatment; do
  echo "[card] ===== $arm  $(date -u) ====="
  $PY -u tools/train/train.py --config "tools/train/config/ab/cardinality-$arm.yaml" \
      2>&1 | tee "$OUT/$arm.log"
  echo "[card] $arm rc=${PIPESTATUS[0]}"
  cp "out/cardinality-$arm/test_metrics.json" "$OUT/$arm.json" 2>/dev/null \
    || echo "[card] NO METRICS for $arm"

  retry "publish metrics ($arm)" publish_metrics "$arm" || RESCUE=1
  retry "publish model ($arm)"   publish_model   "$arm" || RESCUE=1

  echo "[card] ===== $arm done $(date -u) ====="
done

if [ "$RESCUE" -ne 0 ]; then
  echo "[card] *** ONE OR MORE PUBLISHES FAILED ***"
  echo "[card] holding the box so the artefacts can be rescued by hand."
  echo "[card] checkpoints: ~/gliner2/out/cardinality-{control,treatment}/best"
  echo "[card] metrics+logs: $OUT"
  echo "[card] the hard-deadline watchdog still terminates this instance -- it is a"
  echo "[card] rescue window, not an open tab."
  sleep infinity
fi
echo "[card] ALL ARMS DONE $(date -u)"
