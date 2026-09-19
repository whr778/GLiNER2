#!/bin/bash
# Push a corpus to the Hub PRIVATE as soon as its files exist, then verify and register.
#
# WHY IT EXISTS. A batch annotation lands whenever it lands, and the publish must not depend
# on a human or an agent being awake when it does. This waits, pushes, verifies against the
# Hub's FILE LIST (a clean return is not proof -- upload_folder has returned successfully
# having written nothing), and says what still needs registering.
#
# Idempotent: re-running after a successful push re-verifies and changes nothing.
#
#   bash tools/data/publish_when_ready.sh data/cmnee_ner whr778/cmnee_ner
set -uo pipefail
BASE=${1:?corpus base, e.g. data/cmnee_ner}
REPO=${2:?hub repo, e.g. whr778/cmnee_ner}
MAX_WAIT=${MAX_WAIT:-36000}          # 10h: a batch usually returns inside 1h

start=$(date +%s)
while :; do
  found=0
  for s in train val test; do [ -s "${BASE}.${s}.jsonl" ] && found=1; done
  [ "$found" -eq 1 ] && break
  now=$(date +%s)
  if [ $((now - start)) -gt "$MAX_WAIT" ]; then
    echo "[publish] *** gave up after ${MAX_WAIT}s -- ${BASE}.*.jsonl never appeared ***"
    exit 1
  fi
  sleep 120
done

echo "[publish] $(date -u) files present for $BASE -- pushing PRIVATE to $REPO"
uv run python tools/data/push_corpus.py "$BASE" --repo "$REPO" 2>&1 \
  | grep -viE "it/s|B/s|%\|" | tail -5
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "[publish] *** PUSH FAILED rc=$rc -- $BASE is NOT on the Hub ***"
  exit "$rc"
fi
echo "[publish] verified. REMEMBER: a corpus with no hf_jsonl entry in"
echo "[publish] tools/train/dataset_registry.yaml cannot be fetched by a fresh box --"
echo "[publish] that gap killed two A100s on 2026-09-18."
