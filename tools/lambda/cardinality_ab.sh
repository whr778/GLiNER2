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
# READ THE GATE FIRST. Each arm prints `[records] compiled N field spec(s): X scalar ...`
# before scoring. The control must say 0 scalar and the treatment must say non-zero, or
# the arms are not the arms and the run measured nothing -- which has happened once
# already on exactly this mechanism (PROJECT_HISTORY Phase 32).
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1   # FA2 is a CORRECTNESS requirement on mmBERT in bf16

PY=./.venv/bin/python
OUT=$HOME/card_ab
DEST=${DEST:-cardinality_ab}
mkdir -p "$OUT"

for arm in control treatment; do
  echo "[card] ===== $arm  $(date -u) ====="
  $PY -u tools/train/train.py --config "tools/train/config/ab/cardinality-$arm.yaml" \
      2>&1 | tee "$OUT/$arm.log"
  echo "[card] $arm rc=${PIPESTATUS[0]}"
  cp "out/cardinality-$arm/test_metrics.json" "$OUT/$arm.json" 2>/dev/null \
    || echo "[card] NO METRICS for $arm"
  # Publish per arm: the disk dies with the instance, and an arm that is not published
  # when the box goes away is an arm paid for twice.
  $PY - "$arm" "$DEST" <<'PY'
import os, sys
from huggingface_hub import HfApi
arm, dest = sys.argv[1], sys.argv[2]
out, repo = os.path.expanduser("~/card_ab"), "whr778/gliner2-run-logs"
api = HfApi(); api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for f in (f"{arm}.json", f"{arm}.log"):
    p = os.path.join(out, f)
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{dest}/{f}",
                        repo_id=repo, repo_type="dataset")
        print("[card] uploaded", f)
PY
  echo "[card] ===== $arm done $(date -u) ====="
done
echo "[card] ALL ARMS DONE $(date -u)"
