#!/bin/bash
# GENERATION A/B: does splitting "write" from "annotate" produce text that distributes
# its labels more like real news?
#
# ARM A (one call). The model is shown the label menu, picks labels, and then writes a
# document that satisfies them. Its text is a witness for a conclusion it already reached.
# ARM B (two calls). Stage 1 writes with the ontology NEVER shown to the writer
# (prompts.py: WRITE_SYSTEM, "you will never be shown a label set"); stage 2 annotates
# that text cold. Neither stage can shape the other.
#
# PROVENANCE, and it is load-bearing. This is a FRESH DRAW, not a restoration. An earlier
# pair (500 one-call + 90 of stage 1) was generated on 2026-09-07 and lost with the disk
# it lived on; LLM output is not reproducible from a seed, so these numbers do not compare
# to that run, nor to the 160-document pilot that preceded it. BOTH ARMS ARE REGENERATED
# TOGETHER, which is what keeps the A/B internally valid -- the pilot is a prior, not a
# baseline.
#
# BOTH ARMS NOW ASK FOR `uncertain_labels` / `uncertain_types`, and that was a repair, not
# a choice. The block lived only in build_annotate_prompt, so arm B's annotator could
# abstain and arm A's could not -- a second treatment riding along inside the comparison.
# Now shared. These prompts still do NOT inject tools/data/annotation/GUIDELINES.md;
# wiring that in is a separate change and must be measured separately.
#
# Batch mode throughout: -50% pricing, ~$9 for both arms at 500 documents against ~$18
# sync. The batch id is written next to the output, so a killed poller can resume with
# provider.fetch_batch instead of paying twice.
set -euo pipefail

cd "$(dirname "$0")/../../.."      # repo root
N=${N:-500}
MODEL=${MODEL:-claude-haiku-4-5-20251001}
A=${A:-data/lg_onecall}
B=${B:-data/lg_twostage}
EXTRA=${EXTRA:-}                   # e.g. EXTRA=--dry-run for the keyless rehearsal
# BATCH="" runs synchronously at full price. Right ONLY for a few-document smoke, where a
# batch would make you wait an hour to discover the key or the workspace header is wrong.
HFREPO=${HFREPO:-whr778/gliner2-generation-ab}
LOG=${LOG:-out/ab_generation.log}
mkdir -p "$(dirname "$LOG")"

: "${ANTHROPIC_API_KEY:?set it in the calling shell; this script never reads it from disk}"
# An identity-linked key must name the workspace it acts in or the API answers 400.
export ANTHROPIC_WORKSPACE_ID=${ANTHROPIC_WORKSPACE_ID:-wrkspc_01LerUKAa1tvAocoerADEC7j}

# -u is not cosmetic here. Everything below is piped into `tee`, so Python block-buffers
# stdout, and the batch path then prints NOTHING for the hour it spends polling -- a job
# that is working reads as a job that is hung, and the only way to tell them apart is to
# query the API by hand. Unbuffered, the poll line lands every 30s.
GEN="uv run python -u tools/data/synthetic/generate.py --config default.yaml
     --provider anthropic --model $MODEL ${BATCH---batch} $EXTRA"
# Publish the moment a stage lands, not at the end. This rerun exists because the last
# copy of a finished corpus lived on one disk; a stage unpublished when the session ends
# is a stage paid for twice. push_corpus.py verifies the Hub's file list afterwards.
PUB="uv run python tools/data/push_corpus.py --repo $HFREPO"

{
echo "=== ARM A: one call, $N docs, $(date -u) ==="
$GEN --count "$N" --out "$A.jsonl"
$PUB "$A"

echo "=== ARM B stage 1: write only, no ontology, $N docs, $(date -u) ==="
# 1,0,0 on purpose: stage 2 reads ONE file, so an 80/10/10 stage 1 would silently
# annotate 400 of 500 and the arms would differ in size as well as in method.
$GEN --count "$N" --out "${B}_text.jsonl" --write-only --split-ratios 1,0,0
$PUB "${B}_text"

echo "=== ARM B stage 2: annotate that text cold, $(date -u) ==="
$GEN --count "$N" --out "$B.jsonl" --annotate-from "${B}_text.train.jsonl"
$PUB "$B"

echo "=== SCORE: TVD against real news, $(date -u) ==="
uv run python tools/data/compare_label_distributions.py \
  --reference data/cc_news_haiku45 \
  --arm onecall="$A" --arm twostage="$B" \
  --json out/ab_generation_tvd.json
echo "=== DONE $(date -u) ==="
} 2>&1 | tee "$LOG"
