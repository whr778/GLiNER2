#!/bin/bash
# Four dose-curve arms, sequential, each pushed PRIVATE the moment it finishes.
#
# Pushing per arm rather than at the end is deliberate: this is a ~10h run and the disk
# dies with the instance, so an arm that is not pushed when the box goes away is an arm
# that has to be paid for twice.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)

# Measured 2.9 s/it on the A10 at max_len 8192: arms run ~2.7h, ~3.8h, ~6.0h, ~9.1h,
# so the per-arm cap is 12h -- generous enough not to truncate the largest arm, tight
# enough that a hung arm cannot eat the whole budget.
for d in 0 5000 15000 31263; do
  echo "=========== ARM $d  $(date -u) ==========="
  timeout 43200 ./.venv/bin/python -u tools/train/train.py \
    --config tools/train/config/tr-dose$d.yaml 2>&1 | tail -40
  rc=${PIPESTATUS[0]}
  echo "[arm $d] rc=$rc"
  if [ -d "./out/tr-dose$d/best" ]; then
    ./.venv/bin/python -u tools/train/push_to_hub.py \
      --checkpoint ./out/tr-dose$d/best \
      --repo-id whr778/gliner2-tr-dose-$d --private 2>&1 | tail -3
  else
    echo "[arm $d] NO best/ -- nothing to push"
  fi
done
echo "=========== ALL ARMS DONE $(date -u) ==========="
