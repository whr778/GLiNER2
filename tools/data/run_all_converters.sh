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
#
# SCOPE: base corpora only. This does NOT build the derived corpora (scaling slices,
# warmstart/replay mixes, the Turkish dose arms, loc_control, zh_multitask) and does NOT
# buy annotation — annotate_casualty.py / annotate_gate.py / annotate_multitask.py cost
# real money and are never run automatically. Build order and the commands are in
# tools/train/TRAINING.md section 3a.
#
# NOT HERE ON PURPOSE: tools/data/interleave_splits.py. It REPAIRS an existing corpus
# whose splits were written in per-source blocks; build_casualty_multilingual.py now
# shuffles before writing, so a fresh build does not need it. It is a repair tool for
# corpora built before that fix, not a pipeline step.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data
# /tmp does not survive a reboot and has eaten a run's log before now.
DEFAULT_LOG=/tmp/converters.log
[ -d /Volumes/Development/tmp ] && DEFAULT_LOG=/Volumes/Development/tmp/converters.log
LOG="${CONVERTERS_LOG:-$DEFAULT_LOG}"
# DocEE-zh needs a ~108 MB intermediate; park it beside the log, not in data/.
TMPD="$(dirname "$LOG")"
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
run_step professorbob_re     uv run python tools/data/convert_professorbob_re.py --out data/professorbob_re.jsonl
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

# Re-DocRED — revised DocRED with corrected gold annotations; canonical splits.
run_step redocred_train      uv run python tools/data/convert_redocred.py --split train      --out data/redocred.train.jsonl
run_step redocred_val        uv run python tools/data/convert_redocred.py --split validation  --out data/redocred.val.jsonl
run_step redocred_test       uv run python tools/data/convert_redocred.py --split test        --out data/redocred.test.jsonl

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

# MasakhaNER 2.0 (NER, 20 African languages) + MasakhaNEWS (news-topic
# classification, 16 African languages). Both keep their OFFICIAL splits and
# also emit per-language corpora (data/masakhaner_<lang>.*, data/masakhanews_<lang>.*).
# Pass --langs to subset; default 'all'. CC-BY-4.0 / AfricaNLP.
run_step masakhaner          uv run python tools/data/convert_masakhaner.py --out data/masakhaner.jsonl
run_step masakhanews         uv run python tools/data/convert_masakhanews.py --out data/masakhanews.jsonl

# KLUE (Korean) NER + RE, read from the canonical KLUE-benchmark GitHub
# (its HF loader is broken). CC-BY-SA-4.0.
run_step klue_ner            uv run python tools/data/convert_klue.py --task ner --out data/klue_ner.jsonl
run_step klue_re             uv run python tools/data/convert_klue.py --task re  --out data/klue_re.jsonl

# BioRED (biomedical NER + RE) from the NCBI release (~2 MB zip).
run_step biored              uv run python tools/data/convert_biored.py --out data/biored.jsonl
# SciERC (scientific NER + RE) from the AI2 release. NB: the default download is
# ~695 MB (bundles ELMo); pre-extract once and pass --json to skip it.
run_step scierc              uv run python tools/data/convert_scierc.py --out data/scierc.jsonl

# MTL-Bioinformatics-2016 biomedical NER (Crichton et al. 2017; CoNLL BIO, CC BY 4.0).
# One auto-downloading converter per corpus. Overlaps (BC4CHEMD, BC5CDR) and the POS
# corpus (GENIA-pos) are intentionally omitted; per-entity-type subset folders are
# skipped in favor of the full-type corpora (<DATASET>:<output-name>).
for mtl in AnatEM:anatem BC2GM:bc2gm BioNLP09:bionlp09 BioNLP11EPI:bionlp11epi \
           BioNLP11ID:bionlp11id BioNLP13CG:bionlp13cg BioNLP13GE:bionlp13ge \
           BioNLP13PC:bionlp13pc CRAFT:craft Ex-PTM:ex_ptm JNLPBA:jnlpba \
           NCBI-disease:ncbi_disease linnaeus:linnaeus; do
  run_step "mtl_${mtl##*:}" uv run python tools/data/convert_mtl_bio.py \
      --dataset "${mtl%%:*}" --out "data/${mtl##*:}.jsonl"
done

# Event corpora.
# WikiEvents auto-downloads from the public S3 bucket — no manual prep.
run_step wikievents_train uv run python tools/data/convert_wikievents.py --split train --out data/wikievents.train.jsonl
run_step wikievents_dev   uv run python tools/data/convert_wikievents.py --split dev   --out data/wikievents.dev.jsonl
run_step wikievents_test  uv run python tools/data/convert_wikievents.py --split test  --out data/wikievents.test.jsonl

# CASIE auto-downloads the GitHub tarball and emits stratified splits — no manual prep.
run_step casie            uv run python tools/data/convert_casie.py --out data/casie.jsonl

# ChFinAnn / DocFEE / DuEE / Mendeley-ED auto-download (zip / GitHub / HF / tarball) — no manual prep.
run_step chfinann         uv run python tools/data/convert_chfinann.py --out data/chfinann.jsonl
run_step docfee           uv run python tools/data/convert_docfee.py --out data/docfee.jsonl
run_step duee             uv run python tools/data/convert_duee.py --out data/duee.jsonl
run_step mendeley_ed      uv run python tools/data/convert_mendeley_ed.py --out data/mendeley_ed.jsonl

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

# DocEE's published normal_setting splits OVERLAP EACH OTHER -- 56 train/val, 12
# train/test, 26 val/test documents, plus 84 duplicates inside a single split. The
# converter honours them 1:1, so it faithfully reproduces that contamination on every
# run. Without this repair a fresh build silently overwrites the fixed splits with
# broken ones. Precedence is test > val > train, so the blind test keeps its documents.
run_optional docee_dedupe data/docee.train.jsonl \
    uv run python tools/data/dedupe_splits.py data/docee

# DocEE-zh -- 226 MB that sat unconverted because convert_docee.py had no Chinese path.
# TWO STEPS, not one: prepare_docee_zh is a PREPROCESSOR that unwraps the upstream
# one-record-per-list shape (a 1-element list fails the len(raw) >= 4 check, returning
# None for EVERY record) and maps 26 zh event-type spellings onto DocEE-en's canonical
# 59; the existing, tested converter then does the rest.
# --keep-classification-only is load-bearing: without it convert_docee drops the 17,013
# documents of 36,729 whose arguments all fall outside the 4 surface-verified roles, and
# does so WITHOUT ERROR -- the corpus just arrives smaller than it should be.
run_optional docee_zh_prepare data/DocEE/DocEE-zh/DocEE-zh-20230105.json \
    uv run python tools/data/prepare_docee_zh.py \
        --input data/DocEE/DocEE-zh/DocEE-zh-20230105.json --out "$TMPD/docee_zh_prepared.json"
run_optional docee_zh_convert "$TMPD/docee_zh_prepared.json" \
    uv run python tools/data/convert_docee.py --keep-classification-only \
        --input "$TMPD/docee_zh_prepared.json" --out data/docee_zh.jsonl

# Turkish event corpus: the deterministic JOIN of two annotation passes already bought
# (type + per-type role spans). The annotation itself costs money and is never run here
# -- see SCOPE -- but the join over the purchased files belongs in the rebuild path.
run_optional turkish_event data/turkish_gate/ev_ann_tr.jsonl \
    uv run python tools/data/merge_turkish_event.py --out-prefix data/turkish_event

# The three DocEE arms present DIFFERENT label menus -- 59 en / 58 zh / 60 tr. Labels are
# an INPUT at inference, so a model whose English menu lacks `none` cannot answer it, and
# whose Chinese menu lacked `Armed Conflict` was never asked to consider it. Unify to the
# union of 60, leaving true_label untouched. Runs LAST because it reads all three.
run_optional docee_menus data/turkish_event.train.jsonl \
    uv run python tools/data/unify_docee_menus.py

# MAVEN, RAMS — manual local downloads required (see ../train/TRAINING.md §2).
run_optional maven      data/maven/train.jsonl                 uv run python tools/data/convert_maven.py \
                            --input data/maven/train.jsonl --out data/maven.train.jsonl
run_optional rams_train "data/RAMS_1.0c/data/train.jsonlines"   uv run python tools/data/convert_rams.py \
                            --input data/RAMS_1.0c/data/train.jsonlines --out data/rams.train.jsonl
run_optional rams_dev   "data/RAMS_1.0c/data/dev.jsonlines"      uv run python tools/data/convert_rams.py \
                            --input data/RAMS_1.0c/data/dev.jsonlines   --out data/rams.dev.jsonl
run_optional rams_test  "data/RAMS_1.0c/data/test.jsonlines"     uv run python tools/data/convert_rams.py \
                            --input data/RAMS_1.0c/data/test.jsonlines  --out data/rams.test.jsonl

# ACE 2005 is not run here — it lives behind an LDC license; convert it
# separately with `tools/data/convert_ace2005.py --input <your-ace-root> ...`.

# Labels, not text. The Chinese converters emit the SOURCE's labels: convert_duee.py
# passes DuEE's own `label` through, and convert_docfee.py emits Chinese entity keys and a
# Chinese classification MENU. That is the exact state the label unification removed, so a
# converter run without this step silently rebuilds a label space the 137k base never
# learned -- and nothing downstream errors, it just scores badly.
#
# Spans stay Chinese; only labels are rewritten. See CLAUDE.md -> TRAINING DATA LABELS.
run_step labels_zh_to_en  uv run python tools/data/apply_label_map.py \
                            --map tools/data/label_map_zh_all.json \
                            data/duee.train.jsonl data/duee.val.jsonl \
                            data/docfee.train.jsonl data/docfee.val.jsonl data/docfee.test.jsonl \
                            data/text2json.train.jsonl data/text2json.val.jsonl data/text2json.test.jsonl

# Prove it, rather than trusting the step above returned 0.
echo "===== START: labels_verify =====" | tee -a "$LOG"
if uv run python -c "
import sys, json, re, glob
sys.path.insert(0, 'tools/train')
from build_label_maps import labels_by_category
han = re.compile(r'[\u4e00-\u9fff]')
bad = 0
for f in glob.glob('data/duee.*.jsonl') + glob.glob('data/docfee.*.jsonl') + glob.glob('data/text2json.*.jsonl'):
    for line in open(f, encoding='utf-8'):
        for _, label in labels_by_category(json.loads(line)):
            if han.search(label):
                bad += 1
print(f'Chinese labels remaining: {bad}')
sys.exit(1 if bad else 0)
" 2>&1 | tee -a "$LOG"; then
  echo "===== OK:    labels_verify =====" | tee -a "$LOG"
else
  echo "===== FAIL:  labels_verify -- CHINESE LABELS PRESENT =====" | tee -a "$LOG"
  echo "Do not train on this build. See $LOG."
  exit 1
fi

# Gate, not a report. Aggregated train/val/test must be mutually disjoint before any
# of this is trained on -- cross-set contamination invalidates every number downstream,
# and the blind test has to stay blind. A converter run that ends in "ALL DONE" while
# leaving overlapping splits looks successful and is not.
echo "===== START: leakage_gate =====" | tee -a "$LOG"
if uv run python tools/data/check_leakage.py 2>&1 | tee -a "$LOG"; then
  echo "===== OK:    leakage_gate =====" | tee -a "$LOG"
else
  echo "===== FAIL:  leakage_gate -- SPLITS ARE CONTAMINATED =====" | tee -a "$LOG"
  echo "Do not train on this build. See $LOG."
  exit 1
fi

echo "===== ALL DONE =====" | tee -a "$LOG"
