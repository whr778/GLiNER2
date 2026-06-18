# Running GLiNER2 Training from Scratch on mmBERT

End-to-end instructions for training a custom GLiNER2 model on top of `jhu-clsp/mmBERT-small` or `jhu-clsp/mmBERT-base`, using the full **NuNER** and **Pile-NER-definition** corpora as pre-training data. Works on NVIDIA GPUs and Apple M-series (MPS); the trainer auto-selects the best available device.

---

## 1. Install

```bash
git clone <this-repo> GLiNER2 && cd GLiNER2
uv sync                       # installs torch>=2.1, transformers>=4.48, gliner, etc.
uv add datasets               # one-time, only needed for the dataset converters
```

The first training run downloads the mmBERT weights (~550 MB for small, ~1.2 GB for base) into the HuggingFace cache.

---

## 2. Convert the datasets to GLiNER2 JSONL

Every converter writes **three** sibling files based on the `--out` base path: `<base>.train.jsonl`, `<base>.val.jsonl`, `<base>.test.jsonl`. The default split is 80 / 10 / 10 and is deterministic (seeded RNG). Pass `--split-ratios` and/or `--split-seed` to customise.

```bash
# Default 80/10/10 — both forms below produce the same three files:
uv run python tools/data/convert_nuner.py --split full --out data/nuner_full.jsonl
uv run python tools/data/convert_nuner.py --split full --out data/nuner_full      # no .jsonl suffix

# Explicit 80/10/10
uv run python tools/data/convert_nuner.py --split full --out data/nuner_full.jsonl \
    --split-ratios 0.8,0.1,0.1

# 90/5/5 (larger train slice for production runs)
uv run python tools/data/convert_pile_ner_definition.py --out data/pile_ner_def.jsonl \
    --split-ratios 0.9,0.05,0.05

# Reproducible with a custom seed (default seed is 42)
uv run python tools/data/convert_sentence_rex.py --out data/sentence_rex.jsonl \
    --split-seed 1337

# Train-only — useful when you already have separate held-out evaluation data
uv run python tools/data/convert_biomed_ner.py --out data/biomed_ner.jsonl \
    --split-ratios 1.0,0.0,0.0
```

The same flags work on every converter (the helpers in `tools/data/_split.py` add them uniformly). The split assignment is per-record, deterministic across runs with the same seed, and runs in O(1) extra memory — no buffering of the whole corpus.

```bash
mkdir -p data

# NuNER `full` split — 1,000,000 rows with LLM-generated entity descriptions
uv run python tools/data/convert_nuner.py \
    --split full \
    --out data/nuner_full.jsonl

# Pile-NER-definition — ~47,671 conversations, each contributing many entity-type queries
uv run python tools/data/convert_pile_ner_definition.py \
    --out data/pile_ner_def.jsonl

# knowledgator/GLINER-multi-task-synthetic-data — multi-task synthetic NER
# (~10 entity types per record on average; prompt prefix is stripped automatically)
uv run python tools/data/convert_knowledgator_gliner.py \
    --out data/knowledgator_gliner.jsonl

# knowledgator/text2json-training-data — schema-driven structured extraction
# (each record defines its own field names; nested objects are skipped)
uv run python tools/data/convert_text2json.py \
    --out data/text2json.jsonl

# knowledgator/gliner-multilingual-synthetic — multilingual NER
# (German, Polish, French, etc.; pair with mmBERT for non-English extraction)
uv run python tools/data/convert_gliner_multilingual.py \
    --out data/gliner_multilingual.jsonl

# knowledgator/gliclass-v3-logic-dataset — classification (NOT NER)
# Trains GLiNER2's classification head; the trainer interleaves NER and
# classification records cleanly.
uv run python tools/data/convert_gliclass_logic.py \
    --out data/gliclass_logic.jsonl

# knowledgator/gliclass-v2.0-RAC — sibling of v3-logic, same converter,
# general-domain multi-label classification (~612k rows). Override --repo
# and --task-name so the two GLiClass corpora stay namespaced apart.
uv run python tools/data/convert_gliclass_logic.py \
    --repo knowledgator/gliclass-v2.0-RAC \
    --task-name topic_classification \
    --out data/gliclass_rac.jsonl

# knowledgator/Scientific-text-classification — single-label classification
# of scientific abstracts (10 broad domains: math, quantum physics, ...)
uv run python tools/data/convert_scientific_text.py \
    --out data/scientific_text.jsonl

# knowledgator/biomed_NER — domain-specific biomedical NER
# (35 classes: CHEMICALS, DISORDER, GENE AND GENE PRODUCTS, ...)
uv run python tools/data/convert_biomed_ner.py \
    --out data/biomed_ner.jsonl

# knowledgator/events_classification_biotech — multi-label classification
# (29 biotech "event types"; despite the name, NO structured event extraction)
uv run python tools/data/convert_events_biotech.py \
    --out data/events_biotech.jsonl

# knowledgator/sentence_rex — sentence-level relation extraction
# (~850 Wikidata-property labels; <e1>/<e2> markup stripped)
uv run python tools/data/convert_sentence_rex.py \
    --out data/sentence_rex.jsonl

# knowledgator/bio-NER-relations — doc-level biomedical NER + RE
# (50 entity types, 48 relation types; umlsterm noise dropped by default)
uv run python tools/data/convert_bio_ner_relations.py \
    --out data/bio_ner_relations.jsonl

# knowledgator/PubMedAbstractsNER — 35k PubMed abstracts with ~470 UMLS-style
# biomedical entity types; descriptions parsed out of the label string and
# put into entity_descriptions for the model to condition on.
uv run python tools/data/convert_pubmed_abstracts_ner.py \
    --out data/pubmed_abstracts_ner.jsonl
```

### Event extraction corpora (manual download)

ACE 2005, MAVEN, and RAMS are the standard NLP event-extraction benchmarks. None of them is on HuggingFace under those names — you download from the source and point the converters at the local files. Output is a single JSONL file per call (the benchmarks ship canonical train/dev/test splits we want to preserve), not the 3-file split used by the other converters.

```bash
# RAMS (Ebner et al., ACL 2020) — multi-sentence trigger + typed arguments.
#   Download: https://nlp.jhu.edu/rams/  ->  RAMS_1.0c.tar.gz
uv run python tools/data/convert_rams.py \
    --input data/RAMS_1.0c/data/train.jsonlines \
    --out data/rams.train.jsonl

# MAVEN (Wang et al., EMNLP 2020) — large general-domain trigger detection.
#   Download: github.com/THU-KEG/MAVEN-dataset (Tsinghua Cloud / Google Drive)
#   Trigger-only — arguments will be empty, so only the trigger-detection
#   path of the joint event loss benefits from this corpus.
uv run python tools/data/convert_maven.py \
    --input data/maven/train.jsonl \
    --out data/maven.train.jsonl

# ACE 2005 (LDC2006T06) — LDC-licensed. Point --input at the directory
# that contains the .sgm + .apf.xml pairs (typically
# ace_2005_td_v7/data/English). Default emits hierarchical event types
# like "Conflict.Attack"; pass --no-subtypes for "Conflict".
uv run python tools/data/convert_ace2005.py \
    --input /path/to/ace_2005_td_v7/data/English \
    --out data/ace2005.jsonl

# WikiEvents (Li et al., NAACL 2021) — KAIROS ontology event extraction
# co-trained with typed entity mentions; --split auto-downloads from the
# public S3 bucket.
uv run python tools/data/convert_wikievents.py --split train --out data/wikievents.train.jsonl
uv run python tools/data/convert_wikievents.py --split dev   --out data/wikievents.dev.jsonl
uv run python tools/data/convert_wikievents.py --split test  --out data/wikievents.test.jsonl

# CASIE (Satyapanich et al., AAAI 2020) — cybersecurity event extraction
# co-trained with typed entity mentions; auto-downloads the GitHub tarball
# and emits a stratified 80/10/10 split.
uv run python tools/data/convert_casie.py --out data/casie.jsonl

# CMNEE (Zhu et al., LREC-COLING 2024) — Chinese military news event
# extraction with triggers + typed arguments. Google Drive download:
#   mkdir -p data/cmnee && uv run --with gdown gdown --folder \
#       'https://drive.google.com/drive/folders/1nfKiSsu88oBeykUSYm7NGn4Q50_2GPS1' \
#       -O data/cmnee/
uv run python tools/data/convert_cmnee.py \
    --input data/cmnee/CMNEE/train.json --out data/cmnee.train.jsonl
uv run python tools/data/convert_cmnee.py \
    --input data/cmnee/CMNEE/valid.json --out data/cmnee.val.jsonl
uv run python tools/data/convert_cmnee.py \
    --input data/cmnee/CMNEE/test.json  --out data/cmnee.test.jsonl

# DocEE (Tong et al., NAACL 2022) — largest doc-level event-extraction
# corpus (27k docs, 59 types, 356 roles, 180k arg instances). One event
# per doc, no triggers — maps to role-typed entities + 59-way doc
# classification by default. Google Drive download (gdown wraps it):
#   mkdir -p data/docee && uv run --with gdown gdown --folder \
#       'https://drive.google.com/drive/folders/1_cRnc2leAmOKT9Ma8koz6X8Ivl-_lapp' \
#       -O data/docee/
uv run python tools/data/convert_docee.py --no-stratify \
    --input data/docee/DocEE-en/normal_setting/train.json --out data/docee.train.jsonl
uv run python tools/data/convert_docee.py --no-stratify \
    --input data/docee/DocEE-en/normal_setting/dev.json   --out data/docee.val.jsonl
uv run python tools/data/convert_docee.py --no-stratify \
    --input data/docee/DocEE-en/normal_setting/test.json  --out data/docee.test.jsonl

```

All converters stream from HuggingFace (no need to hold the dataset in RAM). Each prints a final summary line: records emitted, records dropped (for NER converters: because no span appeared verbatim; for the classification converters: because the labels couldn't form a valid classification task), and the count of distinct entity types or label counts, followed by per-split counts and file paths.

Approximate output sizes after conversion (totals across all three splits combined):

| Source | Task | Records out | Disk |
|---|---|---:|---:|
| NuNER `full` | NER | ~990,000 | ~0.8 GB |
| Pile-NER-definition | NER | ~45,000 | ~0.2 GB |
| knowledgator/GLINER-multi-task-synthetic-data | NER | ~210,000 | ~0.4 GB |
| knowledgator/text2json-training-data | NER (extraction) | ~80,000 | ~0.2 GB |
| knowledgator/gliner-multilingual-synthetic | NER (multilingual) | ~400,000 | ~0.3 GB |
| knowledgator/gliclass-v3-logic-dataset | Classification (multiple-choice) | ~5,700 | ~10 MB |
| knowledgator/gliclass-v2.0-RAC | Classification (multi-label, general-domain) | ~550,000 | ~1.2 GB |
| knowledgator/Scientific-text-classification | Classification (single-label) | ~50,000 | ~80 MB |
| knowledgator/biomed_NER | NER (biomedical, 35 classes) | ~4,800 | ~30 MB |
| knowledgator/events_classification_biotech | Classification (multi-label) | ~2,750 | ~10 MB |
| knowledgator/sentence_rex | Relation extraction | ~44,000 | ~30 MB |
| knowledgator/bio-NER-relations | Biomedical NER + RE | ~10,400 | ~80 MB |
| knowledgator/PubMedAbstractsNER | NER (biomedical, ~470 UMLS types, with descriptions) | ~35,000 | ~100 MB |
| RAMS (manual download) | Event extraction (trigger + args) | ~9,000 | ~10 MB |
| MAVEN (manual download) | Event detection (trigger only) | ~4,500 | ~20 MB |
| ACE 2005 (LDC) | Event extraction (trigger + args, 33 subtypes) | ~600 | ~3 MB |
| WikiEvents (NAACL 2021) | NER + event extraction (KAIROS, 49+ types) | ~246 docs (206/20/20) | ~2 MB |
| CASIE (AAAI 2020) | NER + cybersecurity event extraction (5 event subtypes, ~21 entity types) | 1,000 docs (794/100/106) | ~8 MB |
| CMNEE (LREC-COLING 2024) | Chinese military event extraction (8 types, 11 roles, 29k events) | 13,617 docs (9,284/1,606/2,727) | ~19 MB |
| DocEE (NAACL 2022) | Role-typed NER + 59-way doc classification (356 roles, 180k args) | 27,485 docs (21,966/2,748/2,771) | ~140 MB |

You can pass any subset of the JSONL files to the trainer at once — they're concatenated and shuffled. Mixing all eleven is a good recipe: NuNER contributes scale and descriptions, Pile-NER contributes long natural-language type definitions, GLINER-multi-task contributes dense multi-type schemas, text2json contributes bespoke per-document field names, gliner-multilingual contributes non-English passages (essential when training on top of `mmBERT` — without it the multilingual encoder weights drift toward English-only extraction), gliclass-logic teaches multiple-choice classification with arbitrary candidate sets, Scientific-text-classification teaches single-label classification with a fixed vocabulary, biomed_NER adds domain-specific biomedical extraction, events_biotech adds multi-label business-news classification, sentence_rex introduces general-domain relation extraction, and bio-NER-relations couples biomedical NER with co-occurring relations.

**Event-extraction training.** The Phase-1-6 event additions land entirely within the existing JSONL/Schema machinery: an event record looks like
```json
{"input": "John fired Bob in Paris.",
 "output": {"events": [
     {"event_type": "Attack", "trigger": "fired",
      "arguments": [
          {"role": "Attacker", "entity": "John"},
          {"role": "Victim",   "entity": "Bob"},
          {"role": "Place",    "entity": "Paris"}
      ]}
 ]}}
```
and an inference schema is `schema.events({"Attack": ["Attacker", "Victim", "Place", "Time"], ...})`. Add the ACE 2005 / MAVEN / RAMS JSONLs to `train_data=` like any other corpus; `compute_metrics` will report `eval_event_trigger_*` and `eval_event_argument_*` micro/macro F1 alongside the entity/relation/classification metrics, with per-event-type and per-role classification reports.

The training scripts under `tools/train/` already wire all three splits into `trainer.train(train_data=…)` / `eval_data=…` / the final blind-test pass. The equivalent inline pattern is:

```python
trainer.train(train_data=[
    "data/nuner_full.train.jsonl",
    "data/pile_ner_def.train.jsonl",
    "data/knowledgator_gliner.train.jsonl",
    "data/text2json.train.jsonl",
    "data/gliner_multilingual.train.jsonl",
    "data/gliclass_logic.train.jsonl",
    "data/scientific_text.train.jsonl",
    "data/biomed_ner.train.jsonl",
    "data/events_biotech.train.jsonl",
    "data/sentence_rex.train.jsonl",
    "data/bio_ner_relations.train.jsonl",
])
```

Swap `.train.jsonl` for `.val.jsonl` to build the `eval_data` list, and `.test.jsonl` for the final blind-test pass.

---

## 3. Pick a hardware profile

The committed training scripts use **`max_len=8192`** — mmBERT's full sequence window. ModernBERT's local-global attention keeps memory near-linear in sequence length, but activations and the optimizer state still scale with it, so batch sizes at 8192 are much smaller than at 512. The cells below assume `max_len=8192`. To regain bigger batches at the cost of context window, lower `max_len` in `TrainingConfig` (e.g. `max_len=2048` or `1024`) and roughly 2–4× each batch-size cell.

| Device | mmBERT-small (148 M params) | mmBERT-base (314 M params) |
|---|---|---|
| **Single A100/H100 80 GB** | `batch_size=16–24`, `bf16=True` | `batch_size=8–12`, `bf16=True` |
| **Single A100/4090 40–48 GB** | `batch_size=8–12`, `bf16=True` | `batch_size=4` + `gradient_accumulation_steps=2`, `bf16=True` |
| **Single 24 GB GPU (3090/4090)** | `batch_size=4–6`, `bf16=True` | `batch_size=2` + `gradient_accumulation_steps=4`, `bf16=True` |
| **Apple M-series (MPS, 32–96 GB unified)** | `batch_size=1–2`, no AMP, dev / small-slice only | `batch_size=1`, no AMP, dev / small-slice only |
| **CPU** | dev / smoke only | dev / smoke only |

Per-step wall-clock at `max_len=8192` is roughly 6–10× longer than at `max_len=512` for the same effective batch size; full-corpus runs on smaller GPUs become multi-day. Tune to your real corpus before committing to a long run.

> On MPS, mixed precision (`fp16`/`bf16`) is disabled automatically: `torch.amp.GradScaler` is CUDA-only and MPS autocast adds little speed-up for this model. The trainer logs the choice it made.
>
> For an Apple M-series dev workflow at `max_len=8192`, an epoch over the full mixed corpus is prohibitive — drop `max_len` to 1024–2048 and/or pass `--max-records 50_000` to the converters to train on a slice. Use a real GPU for the full 8192 corpus.

---

## 4. Train

> **Run the training scripts under `tools/train/` rather than piping a heredoc.** Piping the script via `python - <<PY ... PY` breaks DataLoader workers on macOS + Python 3.12+, where multiprocessing uses the **spawn** start method: workers need to re-import the main module by path, but stdin has no path. Symptom: `FileNotFoundError: '<stdin>'` followed by `BrokenPipeError`.

### Config-driven training (recommended)

A single entry point, `tools/train/train.py`, runs any training setup from a YAML file — no editing Python. Every run is defined by a config in `tools/train/config/`; copy one and edit it to add a corpus or retune hyperparameters.

```bash
uv run python tools/train/train.py --config tools/train/config/mmbert-small.yaml
```

Each YAML has four sections:

| section    | maps to                                  | notes |
|------------|------------------------------------------|-------|
| `model`    | `GLiNER2.from_encoder(encoder, **rest)`  | `encoder`, `max_width`, `max_len`, and the struct-loss knobs (`struct_loss` plus the variant's params — see the structure-loss section below). Omit a knob to take its default. |
| `training` | `TrainingConfig(**training)`             | any `TrainingConfig` field (epochs, LRs, `bf16`, `sliding_window`, `logging_steps`, ...). Omitted fields take their defaults. |
| `eval`     | `make_compute_metrics()` + blind test    | `batch_size`, `threshold`. |
| `data`     | corpus / event-file lists                | `corpora` (base paths, expanded to `<name>.{train,val,test}.jsonl`) and `event_files` (a `{name: {train,val,test}}` map; each split is used only if the file is present on disk). |

Provided configs:

| config | encoder | `struct_loss` | data |
|--------|---------|---------------|------|
| `mmbert-base.yaml` | mmBERT-base | focal | full multi-corpus + all event corpora |
| `mmbert-small.yaml` | mmBERT-small | focal | ACE 2005 |
| `mmbert-small-bce.yaml` | mmBERT-small | bce | WikiEvents |
| `mmbert-small-bce-posweight.yaml` | mmBERT-small | bce_posweight | WikiEvents |
| `mmbert-small-focal.yaml` | mmBERT-small | focal | WikiEvents |
| `mmbert-small-asl.yaml` | mmBERT-small | asl | WikiEvents |
| `mmbert-small-dice.yaml` | mmBERT-small | dice | WikiEvents |
| `mmbert-small-bce-dice.yaml` | mmBERT-small | bce_dice | WikiEvents |

The six `mmbert-small-*` configs share encoder, hyperparameters, and data — they differ only in `struct_loss` and write to separate `output_dir`s, so they form a ready-made loss-variant sweep.

Every run writes `train_results.json` (per-epoch loss + metric history) and `test_metrics.json` (blind-test metrics) into the config's `output_dir`, and prints a compact micro precision/recall/F1 summary on every eval pass.

### Sliding-window chunking (instead of truncation)

By default, records longer than `max_len` word-tokens get truncated by the collator. Set `sliding_window=True` in `TrainingConfig` to switch behaviour: each record's `input` is expanded **at dataset-load time** into overlapping subword-token windows, and each chunk inherits a filtered copy of the gold annotations.

```python
TrainingConfig(
    ...
    sliding_window=True,
    max_len=1024,         # window size, now measured in SUBWORD tokens
    window_stride=512,    # subword stride between consecutive chunks
)
```

Per-task filter rules (see `gliner2/training/chunking.py`):

| task | rule |
|---|---|
| Entities | keep mention only in chunks where its surface verbatim appears |
| Entity descriptions | keep only for entity types that survived the entity filter |
| Classifications | doc-level label is inherited by **every** chunk |
| Relations | emit only if **both** head and tail appear in the same chunk |
| Events | emit if the trigger appears; per-event arguments independently filtered |
| JSON structures | passed through; the processor's verbatim filter handles missing fields |

Notes:

* `max_len`'s **meaning changes**: under sliding window it is the subword window size, not the word-truncation limit. The processor's word-level truncation is suppressed when sliding window is on (chunks are already sized).
* Docs that fit in the window are emitted as a single record (no chunking).
* With `window_stride < max_len`, annotations whose surfaces fall in the overlap region naturally repeat across multiple adjacent chunks — that's intentional, just slightly more supervision on those spans.
* Chunks that wind up with **no usable supervision** after filtering are dropped (e.g. a generic-prose chunk that contains none of the gold spans).
* Currently training-only; evaluation and the blind-test path still truncate at `max_len` as today.

### Per-split deterministic shuffle

After the trainer aggregates JSONLs into a single `TRAIN_DATA` / `EVAL_DATA` / `TEST_DATA` list, every split is shuffled once with `TrainingConfig.seed` before the DataLoader sees it. This means corpus-file ordering can never bias the model. The training DataLoader still shuffles per-epoch on top, so train data is re-shuffled every epoch. Eval and test data stay in the same shuffled-once order between epochs, which keeps eval/test metrics deterministic (set/aggregate metrics like F1 are order-independent anyway).

### 4a. mmBERT-small

Recommended baseline — runs on a single 24 GB GPU and converges in 2–3 epochs of mixed NuNER + Pile-NER-definition.

```bash
uv run python tools/train/train.py --config tools/train/config/mmbert-small.yaml
```

The config sets `model.encoder: jhu-clsp/mmBERT-small`, which `train.py` passes to `GLiNER2.from_encoder` — use `from_encoder` for training from scratch; `from_pretrained` is for loading saved GLiNER2 checkpoints. Edit the YAML (batch size, learning rates, epochs, `max_len`, ...) to retune; defaults target a 24 GB GPU, see the hardware table above for other profiles.

`train.py` also:

* Wires the `.val.jsonl` files into `eval_data=` so the trainer scores them at the end of every epoch with `make_compute_metrics()` — micro/macro precision/recall/F1 for entities, relations, and classifications, plus a per-label `classification_report` string. `eval_loss` drives `save_best=True`, so `<output_dir>/best/` always holds the lowest-val-loss checkpoint.
* After `trainer.train()` returns, calls `evaluate_checkpoint(<output_dir>/best, test_data)` against the `.test.jsonl` splits and writes `test_metrics.json`.

Checkpoints land in the config's `output_dir` (e.g. `out/mmbert-small/`). The `final` checkpoint is the last step; intermediate `checkpoint-<step>` directories are rotated.

### 4b. mmBERT-base

Same shape; lower learning rates, smaller batch, longer wall-clock:

```bash
uv run python tools/train/train.py --config tools/train/config/mmbert-base.yaml
```

Edit `tools/train/config/mmbert-base.yaml` to retune for your hardware. Defaults use `bf16: true` (prefer on Ampere+/Hopper; switch to `fp16: true` elsewhere) and `max_len: 8192`. Eval-on-val and final blind test on `best` work the same way as for the small variant.

### Structure-loss variant (span scoring for entities/relations/events)

The structure head scores every `(start, width)` span against each schema field on a dense grid, so positives (the few gold spans) are vastly outnumbered by negatives. The loss applied to that grid is selected by the `struct_loss` key in the `model:` section of a YAML config — `train.py` forwards the whole `model:` block to `GLiNER2.from_encoder`, so a config sets only the keys its variant uses:

| `struct_loss`    | Behavior                                                        | Extra `model:` keys |
|------------------|-----------------------------------------------------------------|-------------|
| `bce`            | Plain BCE-with-logits (default; original behavior).             | —           |
| `bce_posweight`  | BCE up-weighting positive spans, a principled alternative to the random negative masking. | `struct_pos_weight` (≈ #neg / #pos, e.g. `8.0`) |
| `focal`          | Focal loss — down-weights easy negatives, focuses on hard spans. | `focal_gamma` (default `2.0`), `focal_alpha` (default `0.25`) |
| `asl`            | Asymmetric loss (Ben-Baruch et al.) — decoupled focusing for positives vs negatives plus probability-shifting of easy negatives; built for multi-label extreme negative imbalance. | `asl_gamma_neg` (default `4.0`), `asl_gamma_pos` (default `1.0`), `asl_clip` (default `0.05`) |
| `dice`           | Soft-Dice overlap (F1-like) objective; robust to the dense-grid imbalance by construction. | `dice_smooth` (default `1.0`) |
| `bce_dice`       | `BCE + soft-Dice` — per-cell calibration plus the overlap objective. | `dice_smooth` |

All variants keep the sigmoid + 0.5 decode at inference (`engine.py`), so they slot in without touching the decode path. One ready-to-run config per variant (mmBERT-small, WikiEvents only) ships under `tools/train/config/`:

```bash
uv run python tools/train/train.py --config tools/train/config/mmbert-small-bce.yaml
uv run python tools/train/train.py --config tools/train/config/mmbert-small-bce-posweight.yaml
uv run python tools/train/train.py --config tools/train/config/mmbert-small-focal.yaml
uv run python tools/train/train.py --config tools/train/config/mmbert-small-asl.yaml
uv run python tools/train/train.py --config tools/train/config/mmbert-small-dice.yaml
uv run python tools/train/train.py --config tools/train/config/mmbert-small-bce-dice.yaml
```

Each writes to its own `output_dir` (`./out/mmbert-small/<variant>`), so a sweep can run side by side. The `model:` block to select a variant looks like:

```yaml
model:
  encoder: jhu-clsp/mmBERT-small
  max_width: 20
  max_len: 8192
  struct_loss: asl          # bce | bce_posweight | focal | asl | dice | bce_dice
  asl_gamma_neg: 4.0        # only used by asl
  asl_gamma_pos: 1.0
  asl_clip: 0.05
```

Note: `focal`, `asl`, and the Dice variants handle class imbalance inside the loss, so the 50% random negative masking in `compute_struct_loss` becomes partly redundant — for `focal`/`asl` it still runs (it randomly drops hard negatives the loss wants to learn from), while `dice`/`bce_dice` skip it entirely. For the cleanest comparison, evaluate each variant against the `bce` baseline on your own val splits.

### Optional: hold out an evaluation slice

```python
from gliner2.training.data import TrainingDataset
ds = TrainingDataset.load(["data/nuner_full.jsonl", "data/pile_ner_def.jsonl"])
train_ds, val_ds, _ = ds.split(train_ratio=0.99, val_ratio=0.01, test_ratio=0.0, seed=42)
# then pass train_ds / val_ds to trainer.train(..., eval_data=val_ds) and set
# eval_strategy="steps", eval_steps=2000 in the config.
```

---

## 5. Use the trained model

```python
from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("./out/mmbert-small/final")   # or .../best
print(model.extract_entities(
    "Marie Curie discovered radium in Paris.",
    ["scientist", "element", "city"],
))
```

`GLiNER2.from_pretrained` auto-selects CUDA → MPS → CPU. Pass `map_location="cpu"` to force CPU, or `quantize=True` for fp16 inference on GPU/MPS.

---

## 6. Push to Hugging Face Hub

Once you're happy with a checkpoint, upload it so it can be loaded by anyone with `GLiNER2.from_pretrained("<username>/<repo>")`.

First, authenticate (once per machine):

```bash
uv run huggingface-cli login        # paste a write token from https://huggingface.co/settings/tokens
# or: export HF_TOKEN=hf_xxx        # non-interactive (CI, headless boxes)
```

Then push:

```bash
uv run python tools/train/push_to_hub.py \
    --checkpoint ./out/mmbert-small/final \
    --repo-id <username>/gliner2-mmbert-small \
    --private
```

Flags:

- `--checkpoint` — local path to a saved GLiNER2 checkpoint directory (`final`, `best`, or any `checkpoint-<step>`).
- `--repo-id` — target repo, created if missing.
- `--private` (default) / `--public` — repo visibility.
- `--commit-message` — optional commit message (defaults to `Upload GLiNER2 checkpoint`).

The script calls `model.save_pretrained()` into a temp directory and uploads the folder via `HfApi.upload_folder`, so the repo layout matches what `GLiNER2.from_pretrained` expects (`config.json` + `encoder_config/config.json` + `model.safetensors` + tokenizer files).

Sanity-check the upload:

```python
from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("<username>/gliner2-mmbert-small")
print(model.extract_entities("Marie Curie discovered radium in Paris.",
                             ["scientist", "element", "city"]))
```

---

## 7. Practical tips

- **Run a 50-record smoke test first** before launching a multi-day run:
  ```bash
  uv run python tools/data/convert_nuner.py --split full --out /tmp/nuner_smoke.jsonl --max-records 50
  # then train with num_epochs=1, batch_size=2, max_steps=20 to confirm the loss falls
  ```
- **Mix the two datasets in one pass.** Passing them as a list to `train_data=` interleaves them; the converter scripts already produce compatible JSONL.
- **Resume mid-run** by pointing a fresh `from_pretrained` at the latest checkpoint directory (`./out/.../checkpoint-<step>`) and starting a new trainer. The trainer always starts a new optimizer/scheduler — that's intentional, not a bug.
- **Watch the loss curve.** Healthy mmBERT-small on this data starts at ~500 (batch 1), drops below 100 in the first ~50 steps, then drifts down toward 40–60 over an epoch. Loss collapsing to ~0 means the data labels aren't reaching the loss head — stop and inspect.
- **W&B**: set `report_to_wandb=True` and `wandb_project="..."` in `TrainingConfig` to stream live metrics.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: datasets` | Converters need the HF `datasets` lib | `uv add datasets` |
| `FileNotFoundError: '<stdin>'` + `BrokenPipeError` | Running training inside a heredoc (`python - <<PY`) with `num_workers>0`; spawn workers can't re-import stdin | Save the script as a `.py` file and run `uv run python <file>.py`, or set `num_workers=0` |
| `out of memory` on CUDA | Batch too large | Halve `batch_size`, double `gradient_accumulation_steps` |
| `out of memory` on MPS | Same | Same; also try `max_len=384` and `num_workers=0` |
| Loss stuck at exactly the initial value | LR too low or all params frozen | Confirm `use_lora=False` (or pass LoRA modules); raise `task_lr` |
| Loss explodes (`NaN`/`Inf`) | LR too high or mixed precision on a fragile encoder | Drop `encoder_lr` 5x, set `fp16=False` |
| `train_metrics_history` is empty | `logging_steps` larger than total steps | Lower `logging_steps`, or run more epochs |
| Tokenizer error about unknown special tokens | mmBERT tokenizer didn't accept GLiNER2's specials | Should never happen with `transformers>=4.48` — file an issue and include the tokenizer version |
| `401 Unauthorized` from `push_to_hub.py` | No HF auth token | `uv run huggingface-cli login` (with a **write**-scope token), or `export HF_TOKEN=hf_xxx` |
| `403 Forbidden` writing to `<repo-id>` | Token lacks write access or repo is owned by someone else | Issue a new write token, or pick a repo-id you own |
