#!/bin/bash
# OVERFIT MATRIX JOB: tools/train/overfit_relations.py for every ARM x CORPUS on the box's GPU, from one checkpoint on
# the Hub, publishing each run's JSON + log to run-logs DEST/ as it lands. Can the relation head fit 64 training records
# of a corpus, and which of the two leaks (proposal ranking, schema dropout) stops it? Each run is the harness's
# own control-vs-treatment read: F1 @0.3 / @0.05 on the training records and positive_pairs / gold_pairs.
#
# Evidence survives the box: a start marker before the first run, every run's log published as it finishes, the job
# log published on EVERY exit path.
#
#   JOB="CFG=tools/train/config/base/eb22c-ddp-trace.yaml DEST=overfit-matrix bash tools/lambda/overfit_job.sh"
#   ARMS: name=flags pairs separated by ';' (flags may be empty); CORPORA: space list; CKPT: repo:checkpoint-dir.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export CKPT_CACHE=$HOME/ckpts
PY=./.venv/bin/python
CFG=${CFG:?CFG}; DEST=${DEST:?DEST}
CKPT=${CKPT:-whr778/gliner2-eb22c-collapse-trace:checkpoint-30000}
ARMS=${ARMS:-"control=;dropoff=--no-schema-dropout;keepgold=--keep-gold-pairs;both=--no-schema-dropout --keep-gold-pairs"}
CORPORA=${CORPORA:-"biored professorbob_re scierc sentence_rex"}
RECORDS=${RECORDS:-64}; STEPS=${STEPS:-400}; EVERY=${EVERY:-50}; BATCH=${BATCH:-8}
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
OUT=$HOME/overfit
JOBLOG=${JOBLOG:-$HOME/job.log}
DRY=${DRY:-}
mkdir -p "$OUT"
source tools/lambda/_publish.sh
trap 'echo "[overfit] exit trap: publishing the job log $(date -u)"; [ -n "$DRY" ] || publish "$DEST" "$JOBLOG"' EXIT
echo "[overfit] ===== START $(date -u) | $(git rev-parse --short HEAD) | $CFG | ckpt $CKPT | arms: $ARMS | corpora: $CORPORA | $RECORDS records x $STEPS steps ====="
if [ -z "$DRY" ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "start $(date -u) commit $(git rev-parse --short HEAD) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)" > "$OUT/start.txt"
  publish "$DEST" "$OUT/start.txt" || { echo "[overfit] *** FATAL: cannot publish the start marker ***"; exit 1; }
  CKDIR=$($PY -c "import sys; sys.path.insert(0,'tools/train'); import probe_event_collapse as P; print(P.local_checkpoint('$CKPT'))" 2>"$OUT/ckpt.dl.log" | tail -1)
  [ -d "$CKDIR" ] || { echo "[overfit] *** FATAL: checkpoint $CKPT not downloaded ($CKDIR) ***"; tail -n 3 "$OUT/ckpt.dl.log"; exit 1; }
  echo "[overfit] checkpoint at $CKDIR"
else
  CKDIR=/dry/ckpt
fi
failed=0; done_n=0
run_one() {  # arm flags corpus   (separate `local` statements: `local a=$1 b=$a` is unbound under set -u)
  local arm=$1
  local flags=$2
  local corpus=$3
  local tag=$arm-$corpus
  local out=$OUT/$tag.json
  local log=$OUT/$tag.log
  local t0=$SECONDS
  if [ -n "$DRY" ]; then
    echo "[overfit] DRY $tag: $PY tools/train/overfit_relations.py --config $CFG --checkpoint $CKDIR --corpus $corpus --records $RECORDS --steps $STEPS --every $EVERY --batch-size $BATCH --device cuda $flags --out $out"
    return
  fi
  PYTHONUNBUFFERED=1 timeout "$RUN_TIMEOUT" $PY tools/train/overfit_relations.py --config "$CFG" --checkpoint "$CKDIR" --corpus "$corpus" \
     --records "$RECORDS" --steps "$STEPS" --every "$EVERY" --batch-size "$BATCH" --device cuda $flags --out "$out" > "$log" 2>&1
  local rc=$?
  echo "[overfit] $tag: exit $rc | $((SECONDS - t0)) s | $(date -u +%H:%M:%S) | last: $(grep -a '^\[overfit\] {' "$log" | tail -1 | cut -c1-260)"
  grep -a "schema dropout\|relation_keep_gold_pairs\|^relation pairs:" "$log" | tail -n 3 | cut -c1-200
  if [ "$rc" -eq 0 ] && [ -s "$out" ]; then done_n=$((done_n+1)); else failed=$((failed+1)); grep -aE "Error|Traceback" "$log" | tail -2; fi
  publish "$DEST" "$log" $([ -s "$out" ] && echo "$out") || echo "[overfit] publish of $tag failed (continuing; final publish retries)"
}
IFS=';' read -r -a arm_list <<< "$ARMS"
for corpus in $CORPORA; do
  for entry in "${arm_list[@]}"; do
    run_one "${entry%%=*}" "${entry#*=}" "$corpus"
  done
done
echo "[overfit] scored $done_n, failed $failed $(date -u)"
[ -n "$DRY" ] && exit 0
publish "$DEST" "$OUT"/*.json "$OUT"/*.log || { echo "[overfit] *** FATAL: FINAL PUBLISH FAILED ***"; exit 1; }
[ "$failed" -eq 0 ] || { echo "[overfit] *** FATAL: $failed run(s) failed (results published) ***"; exit 2; }
echo "[overfit] ===== DONE $(date -u) ====="
