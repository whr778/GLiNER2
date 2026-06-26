#!/usr/bin/env bash
# Run every converter under tools/data/ except ACE 2005 (which the user
# stratifies on their own LDC-licensed corpus). Writes split JSONL into
# data/ and a per-step log to /tmp/converters.log so failures are easy
# to diagnose without losing the rest of the run.
#
# Usage:
#   tools/data/run_all_converters.sh
#
# Prereqs:
#   - HuggingFace cache is reachable (most converters stream directly).
#   - data/maven/train.jsonl       — manual MAVEN download (skipped if absent).
#   - data/RAMS_1.0c/data/         — manual RAMS download (skipped if absent).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data
LOG="${CONVERTERS_LOG:-/tmp/converters.log}"
: > "$LOG"
echo "Log: $LOG"

run_step() {
  local name="$1"; shift
  echo "===== START: $name =====" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "===== OK:    $name =====" | tee -a "$LOG"
  else
    echo "===== FAIL:  $name (exit $?) =====" | tee -a "$LOG"
  fi
}

run_optional() {
  local name="$1" guard="$2"; shift 2
  if [[ ! -e "$guard" ]]; then
    echo "===== SKIP:  $name (missing input: $guard) =====" | tee -a "$LOG"
    return
  fi
  run_step "$name" "$@"
}

# Small / fast (HuggingFace streaming or single file)
run_step pile_ner_def        uv run python tools/data/convert_pile_ner_definition.py --out data/pile_ner_def.jsonl
run_step gliclass_logic      uv run python tools/data/convert_gliclass_logic.py --out data/gliclass_logic.jsonl
run_step biomed_ner          uv run python tools/data/convert_biomed_ner.py --out data/biomed_ner.jsonl
run_step events_biotech      uv run python tools/data/convert_events_biotech.py --out data/events_biotech.jsonl
run_step text2json           uv run python tools/data/convert_text2json.py --out data/text2json.jsonl
run_step paraloq_json        uv run python tools/data/convert_paraloq_json.py --out data/paraloq_json.jsonl
run_step sentence_rex        uv run python tools/data/convert_sentence_rex.py --out data/sentence_rex.jsonl
run_step bio_ner_relations   uv run python tools/data/convert_bio_ner_relations.py --out data/bio_ner_relations.jsonl
run_step pubmed_abstracts    uv run python tools/data/convert_pubmed_abstracts_ner.py --out data/pubmed_abstracts_ner.jsonl
run_step scientific_text     uv run python tools/data/convert_scientific_text.py --out data/scientific_text.jsonl

# GLiClass v2.0-RAC reuses the v3-logic converter with --repo / --task-name override.
run_step gliclass_rac        uv run python tools/data/convert_gliclass_logic.py \
                                 --repo knowledgator/gliclass-v2.0-RAC \
                                 --task-name topic_classification \
                                 --out data/gliclass_rac.jsonl

# Larger HF-streamed corpora.
run_step knowledgator_gliner uv run python tools/data/convert_knowledgator_gliner.py --out data/knowledgator_gliner.jsonl
run_step gliner_multilingual uv run python tools/data/convert_gliner_multilingual.py --out data/gliner_multilingual.jsonl
run_step nuner_full          uv run python tools/data/convert_nuner.py --split full --out data/nuner_full.jsonl

# DocRED — doc-level NER + relations. Auto-downloads the parquet revision; its
# 'train' split merges gold + distant-supervised docs (~270 MB output).
run_step docred              uv run python tools/data/convert_docred.py --out data/docred.jsonl

# Token-classification NER corpora (auto-download; parquet revision where the
# original ships a dataset script).
run_step kaznerd             uv run python tools/data/convert_hf_token_ner.py \
                                 --repo yeshpanovrustem/kaznerd --out data/kaznerd.jsonl
run_step bc4chemd            uv run python tools/data/convert_hf_token_ner.py \
                                 --repo chintagunta85/bc4chemd --revision refs/convert/parquet \
                                 --out data/bc4chemd.jsonl
run_step bc5cdr              uv run python tools/data/convert_hf_token_ner.py \
                                 --repo tner/bc5cdr --revision refs/convert/parquet --tags-col tags \
                                 --label-file dataset/label.json --out data/bc5cdr.jsonl
run_step stockmark_jpn       uv run python tools/data/convert_stockmark_ner.py --out data/stockmark_jpn.jsonl
# finer-ord is cc-by-nc-4.0 (non-commercial).
run_step finer_ord           uv run python tools/data/convert_finer_ord.py --out data/finer_ord.jsonl

# KLUE (Korean) NER + RE, read from the canonical KLUE-benchmark GitHub
# (its HF loader is broken). CC-BY-SA-4.0.
run_step klue_ner            uv run python tools/data/convert_klue.py --task ner --out data/klue_ner.jsonl
run_step klue_re             uv run python tools/data/convert_klue.py --task re  --out data/klue_re.jsonl

# BioRED (biomedical NER + RE) from the NCBI release (~2 MB zip).
run_step biored              uv run python tools/data/convert_biored.py --out data/biored.jsonl
# SciERC (scientific NER + RE) from the AI2 release. NB: the default download is
# ~695 MB (bundles ELMo); pre-extract once and pass --json to skip it.
run_step scierc              uv run python tools/data/convert_scierc.py --out data/scierc.jsonl

# Event corpora.
# WikiEvents auto-downloads from the public S3 bucket — no manual prep.
run_step wikievents_train uv run python tools/data/convert_wikievents.py --split train --out data/wikievents.train.jsonl
run_step wikievents_dev   uv run python tools/data/convert_wikievents.py --split dev   --out data/wikievents.dev.jsonl
run_step wikievents_test  uv run python tools/data/convert_wikievents.py --split test  --out data/wikievents.test.jsonl

# CASIE auto-downloads the GitHub tarball and emits stratified splits — no manual prep.
run_step casie            uv run python tools/data/convert_casie.py --out data/casie.jsonl

# CMNEE — Chinese military event extraction. Manual Google Drive download.
run_optional cmnee_train data/cmnee/CMNEE/train.json \
    uv run python tools/data/convert_cmnee.py \
        --input data/cmnee/CMNEE/train.json --out data/cmnee.train.jsonl
run_optional cmnee_val   data/cmnee/CMNEE/valid.json \
    uv run python tools/data/convert_cmnee.py \
        --input data/cmnee/CMNEE/valid.json --out data/cmnee.val.jsonl
run_optional cmnee_test  data/cmnee/CMNEE/test.json \
    uv run python tools/data/convert_cmnee.py \
        --input data/cmnee/CMNEE/test.json --out data/cmnee.test.jsonl

# LEVEN — Chinese legal event detection (trigger only). Manual Google Drive
# download (gdown wraps it); see ../train/TRAINING.md. test.jsonl has its annotations
# removed, so only train / valid are converted.
run_optional leven_train data/leven/LEVEN/train.jsonl \
    uv run python tools/data/convert_leven.py \
        --input data/leven/LEVEN/train.jsonl --out data/leven.train.jsonl
run_optional leven_val   data/leven/LEVEN/valid.jsonl \
    uv run python tools/data/convert_leven.py \
        --input data/leven/LEVEN/valid.jsonl --out data/leven.val.jsonl

# DocEE — manual Google Drive download required. Run each canonical
# split through the converter; existence-guard on the train file means
# the whole block is skipped cleanly when DocEE isn't present.
run_optional docee_train data/docee/DocEE-en/normal_setting/train.json \
    uv run python tools/data/convert_docee.py --no-stratify \
        --input data/docee/DocEE-en/normal_setting/train.json --out data/docee.train.jsonl
run_optional docee_val   data/docee/DocEE-en/normal_setting/dev.json \
    uv run python tools/data/convert_docee.py --no-stratify \
        --input data/docee/DocEE-en/normal_setting/dev.json --out data/docee.val.jsonl
run_optional docee_test  data/docee/DocEE-en/normal_setting/test.json \
    uv run python tools/data/convert_docee.py --no-stratify \
        --input data/docee/DocEE-en/normal_setting/test.json --out data/docee.test.jsonl

# MAVEN, RAMS — manual local downloads required (see ../train/TRAINING.md §2).
run_optional maven      data/maven/train.jsonl                 uv run python tools/data/convert_maven.py \
                            --input data/maven/train.jsonl --out data/maven.train.jsonl
run_optional rams_train data/RAMS_1.0c/data/train.jsonlines    uv run python tools/data/convert_rams.py \
                            --input data/RAMS_1.0c/data/train.jsonlines --out data/rams.train.jsonl
run_optional rams_dev   data/RAMS_1.0c/data/dev.jsonlines      uv run python tools/data/convert_rams.py \
                            --input data/RAMS_1.0c/data/dev.jsonlines   --out data/rams.dev.jsonl
run_optional rams_test  data/RAMS_1.0c/data/test.jsonlines     uv run python tools/data/convert_rams.py \
                            --input data/RAMS_1.0c/data/test.jsonlines  --out data/rams.test.jsonl

# ACE 2005 is not run here — it lives behind an LDC license; convert it
# separately with `tools/data/convert_ace2005.py --input <your-ace-root> ...`.

echo "===== ALL DONE =====" | tee -a "$LOG"
