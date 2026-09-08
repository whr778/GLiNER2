#!/bin/bash
# Mechanism-or-model: does the greedy -> joint structure deficit follow the DECODE PATH
# or the MODEL? Phase 31 measured it on one checkpoint (structure -0.0454, six heads
# inside the +/-0.02 floor) and RESEARCH_PROGRAM sec 3 records the question as open.
#
# Six arms, three checkpoints x {greedy, joint}, ONE config so ONE test set. The
# checkpoint is the only variable inside a pair, and only WITHIN-checkpoint deltas are
# comparable across pairs -- absolute cross-checkpoint scores are contaminated (the 137k
# line may have trained on records in eb16's test) and that does not matter here.
#
# ORDER IS DELIBERATE. The control first, so a wrong environment is caught before any
# money goes on the question; then the independent-lineage arm, because
# casualty-multilingual is a CHILD of the control and a no-replay fine-tune whose
# structure head may be dead (gate3-warm-cells), which would return "inconclusive".
# A box that dies after two pairs still answers the question.
#
# NO --batch-size AND NO --joint-beam-width: both take the config's value, which is what
# Phase 31 scored. Batch size is proven output-neutral to 0.002 but the control's job is
# to reproduce a number, not to test that proof again.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
# FA2 is a CORRECTNESS requirement on mmBERT in bf16, not a speedup: sdpa+bf16 goes
# non-finite. Fail loudly rather than fall back silently.
export GLINER2_STRICT_ATTN=1

CFG=tools/train/config/base/eb16-rebuild-tr.yaml
PY=./.venv/bin/python
OUT=$HOME/decode_arms
LOGREPO=whr778/gliner2-run-logs
# PAIRS selects which checkpoints run; DEST names the folder they publish into. A re-run
# on changed code MUST NOT overwrite the numbers it is being compared against.
PAIRS=${PAIRS:-"eb16-rebuild-tr mmbert-137k-clean casualty-multilingual-eb16tr"}
DEST=${DEST:-decode_arms}
mkdir -p "$OUT"

publish() {   # publish after EVERY arm: Phase 31's JSONs survive nowhere because they
  local tag=$1  # were kept on a disk that died with the box.
  $PY - "$tag" "$DEST" <<'PY'
import os, sys
from huggingface_hub import HfApi
tag, dest = sys.argv[1], sys.argv[2]
out, repo = os.path.expanduser("~/decode_arms"), "whr778/gliner2-run-logs"
api = HfApi(); api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for f in (f"{tag}.json", f"{tag}.log"):
    p = os.path.join(out, f)
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{dest}/{f}",
                        repo_id=repo, repo_type="dataset")
        print("[arms] uploaded", f)
PY
}

repo_for() {
  case "$1" in
    eb16-rebuild-tr)              echo whr778/gliner2-eb16-rebuild-tr ;;
    mmbert-137k-clean)            echo whr778/gliner2-joint-boundary-mmbert-137k-clean ;;
    casualty-multilingual-eb16tr) echo whr778/gliner2-casualty-multilingual-eb16tr ;;
    *) echo "" ;;
  esac
}

run_pair() {
  local tag=$1 repo=$2
  local dir=$HOME/ckpt/$tag
  if [ ! -f "$dir/model.safetensors" ]; then
    echo "[arms] downloading $repo"
    $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$dir')" || { echo "[arms] DOWNLOAD FAILED $repo"; return 1; }
  fi
  for mode in greedy joint; do
    local t="$tag.$mode"
    echo "[arms] ===== $t  $(date -u) ====="
    $PY -u tools/train/eval.py --config "$CFG" --checkpoint "$dir" \
        --split test --threshold 0.5 --decode-mode "$mode" 2>&1 | tee "$OUT/$t.log"
    if [ -f out/eb16-rebuild-tr/test_metrics.json ]; then
      cp out/eb16-rebuild-tr/test_metrics.json "$OUT/$t.json"
      rm -f out/eb16-rebuild-tr/test_metrics.json   # every arm writes THIS path; a stale
    else                                            # copy would silently duplicate an arm
      echo "[arms] NO METRICS for $t"
    fi
    publish "$t"
    echo "[arms] ===== $t done $(date -u) ====="
  done
}

for pair in $PAIRS; do
  repo=$(repo_for "$pair")
  [ -n "$repo" ] || { echo "[arms] unknown checkpoint $pair -- skipping"; continue; }
  run_pair "$pair" "$repo"
done

echo "[arms] ALL ARMS DONE $(date -u)"
