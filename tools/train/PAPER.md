# Schema-Driven Information Extraction Beyond the Sentence: Event Extraction, Multilingual Training, and Document-Level Global Decoding for GLiNER2

**William Roe**¹ (whr778@gmail.com) and **Claude**² (noreply@anthropic.com)

¹ Project author and maintainer  ·  ² AI assistant (Anthropic, Claude Opus 4.8) — design, implementation, and drafting

*Working paper — engineering and methodology contributions on the `mmbert_training` branch. Results tables are placeholders pending the current training run.*

---

## Abstract

GLiNER2 is a single encoder model that performs named-entity recognition, text
classification, relation extraction, and hierarchical structured extraction from
a schema prompt in one forward pass. This work extends it along four axes. First, we add **event
extraction** — ACE-style triggers with typed, multi-valued arguments — as a
first-class task threaded through the schema, processor, inference engine, API,
and training data model. Second, we make the model **encoder-agnostic and
trainable from scratch on any Hugging Face backbone** (e.g. the multilingual,
long-context `jhu-clsp/mmBERT-base`), with training infrastructure for Apple
MPS, multi-GPU (DDP and DataParallel), gradient checkpointing, sliding-window
chunking, and mid-run resume. Third, we build an **evaluation methodology** with
strict and relaxed regimes, multilingual (language-ID-driven) blind testing,
fine-grained "fair" error analysis after Ortmann (2022), and post-training
threshold calibration. Fourth, we address the **document-level** setting — where
an event's arguments are dispersed across a document longer than the encoder's
window — with an opt-in, OneIE-style (Lin et al., 2020) **global graph decoder**
that reconnects events across overlapping windows. A pipeline of ~30 corpus
converters normalizes public NER, relation, classification, and event datasets
into a single JSONL format. All additions are opt-in and default-off, preserving
the base model's behavior.

---

## 1. Introduction

Information extraction (IE) is usually split across task-specific models and
sentence-bounded assumptions. GLiNER2 collapses entities, classification, and
structured extraction into one schema-conditioned encoder. Two gaps motivated
this work:

1. **Task coverage.** GLiNER2 lacked event extraction — the task of detecting an
   event trigger and filling its typed argument roles — which is central to
   converting narrative text into structured records.
2. **Scale and locality.** The model and its training/evaluation code assumed
   short, single-window inputs. Document-level event corpora (e.g. WikiEvents;
   Li et al., 2021) violate this: arguments routinely sit several sentences from
   their trigger, and documents exceed the encoder's position limit.

We close both while keeping the base package's inference behavior unchanged: new
capabilities are additive and off by default.

## 2. Background: the GLiNER2 baseline

GLiNER2 encodes `[schema tokens] + [text]`, computes span representations, and
scores candidate spans against schema-derived label embeddings. A fluent
`create_schema()` builder composes tasks; a single `extract(text, schema)` call
returns entities, classifications, structures, and relations. The base models
(`fastino/gliner2-base-v1`, `-large-v1`) use a DeBERTa-v3 encoder (512-position
cap). Everything below is built on this substrate.

## 3. Contributions at a glance

- **Event extraction** as an additional first-class task — alongside NER, text
  classification, relation extraction, and hierarchical structured extraction —
  consistent across all layers (§4).
- **Encoder-agnostic loading + `from_encoder()`** to train from a raw backbone,
  and infrastructure for MPS / DDP / DataParallel / gradient checkpointing /
  sliding-window chunking / checkpoint-resume (§5).
- **Configurable structure losses** (BCE, positive-weighted BCE, focal,
  asymmetric, Dice, BCE+Dice) with independent per-task (event) overrides (§6).
- **A ~30-converter data pipeline** normalizing public corpora to UTF-8 NFKC
  JSONL (§7).
- **Evaluation methodology**: strict/relaxed regimes; per-language blind test
  with language ID; multilingual stopword-aware relaxed matching; CJK
  tokenization; fine-grained fair error analysis (Ortmann, 2022); post-training
  threshold calibration (§8).
- **Document-level global event decoding** (OneIE-style) as an opt-in mode on
  the inference API, the eval config, and a CLI (§9).

## 4. Event extraction as a first-class task

Events are modeled as multi-field structures: each event type expands to a
schema row `[trigger, role_1, …, role_k]`. The feature spans five classes across
five files (`Schema`, `SchemaInput`, `SchemaTransformer`, the `GLiNER2` engine,
and `SchemaAPI`/`GLiNER2API`), plus the training data model:

- **Schema** — `Schema.events({event_type: [roles]})` (also a richer form with
  descriptions and per-type trigger/argument thresholds), normalized into
  `_event_metadata` / `_event_order` and wired through `build()` / `from_dict()`
  / `from_pydantic()`.
- **Processor** — a `[V]` trigger token; `_process_events()` handles both the
  inference (schema-only) and training (list-of-mentions) shapes, unioning roles
  across occurrences.
- **Engine** — `_extract_events()` decodes the top trigger span and all
  role-argument spans per predicted mention, bucketed under `event_extraction`;
  public `extract_events()` / `batch_extract_events()`.
- **Training data** — `Event` / `EventArgument` dataclasses that validate every
  trigger and argument surface appears verbatim in the source text; wired
  through `validate()` / `sanitize()` / `to_dict()` / `from_dict()`.

A dedicated event loss path (§6) lets event supervision be tuned independently.

## 5. Training from arbitrary encoders and at scale

**Encoder-agnostic loading.** `from_pretrained` resolves the input-embedding
parameter dynamically (via `get_input_embeddings()`), calls `tie_weights()`
after resize/load for encoders that tie embeddings (e.g. ModernBERT/mmBERT), and
auto-selects device (CUDA → MPS → CPU). A new `from_encoder()` classmethod
bootstraps a fresh GLiNER2 (pretrained encoder + randomly-initialized task
heads) from a raw HF backbone, enabling training from scratch on
`jhu-clsp/mmBERT-base` and similar.

**Infrastructure.** The trainer adds: an MPS device tier (with a cuDNN-SDPA
workaround for long variable-length sequences); multi-GPU via torchrun **DDP**
(rank-0 eval + broadcast early-stop) and opt-in `nn.DataParallel` (with an
autocast-preserving replica wrapper); **gradient checkpointing**;
**sliding-window chunking** of long training records into overlapping subword
windows; **checkpoint-restart** (optimizer/scheduler/RNG state for true mid-run
resume); generalized OOM handling and cross-backend cache management; the modern
`torch.amp` API; and pre-training ETA estimation.

## 6. Configurable structure losses

`ExtractorConfig.struct_loss` selects among `bce`, `bce_posweight`, `focal`,
`asl` (asymmetric loss), `dice`, and `bce_dice`, with per-variant
hyperparameters. Region-based Dice bypasses the random-negative-masking path the
pointwise variants use. Crucially, `event_struct_loss` /
`event_struct_pos_weight` override the loss **for events only**, so entity and
relation supervision can be tuned separately from the sparser, harder event
supervision. Loss is tracked as four components
(`classification` / `structure` / `event_structure` / `count`). Each loss
variant ships with a matching YAML config for controlled comparison.

## 7. Data pipeline

`tools/data/` contains ~30 converters that normalize public corpora into a
single GLiNER2 JSONL schema, all routed through a shared writer that emits
UTF-8, NFKC-normalized records with stray Unicode line separators stripped
(so records never fragment across lines). Coverage includes:

- **NER**: NuNER, Pile-NER, PubMed, BC4CHEMD, BC5CDR, FiNER-ORD, Stockmark (ja),
  KazNERD, KLUE (ko), SciERC, BioRED, and generic HF token-NER.
- **Relations**: DocRED, Re-DocRED, SciERC, BioRED, sentence-REx, bio-NER-relations.
- **Classification**: gliclass-logic, Scientific-text, paraloq schema extraction.
- **Events**: ACE 2005 (with NAM/NOM/PRO mention-filter variants), WikiEvents,
  DocEE, RAMS, MAVEN, CASIE, CMNEE (zh), LEVEN (zh).

Converters are config-driven (e.g. entity mention-type filtering, label
roll-up/remap) and stratify train/val/test splits deterministically.

## 8. Evaluation methodology

**Strict and relaxed regimes.** Every category (entity, relation, classification,
and the event sub-metrics `event_type`, `event_trigger`, `event_argument`, and
combined `event`) is scored twice: *strict* (exact surface, one-to-one) and
*relaxed* (type exact + surface overlap, stopword-aware and normalized, so
relaxed never scores below strict). Details in [`METRICS.md`](../../METRICS.md).

**Multilingual.** A per-language blind test uses language ID to bucket records,
reports per-language then combined, and a multilingual stopword set (auto-detected
from corpus language codes) drives relaxed matching; CJK text is tokenized
per-character so Asian-language spans are scored correctly.

**Fine-grained fair error analysis (Ortmann, 2022).** Standard P/R/F1
double-penalize a near-miss (a right-surface/wrong-label or right-label/wrong-
boundary span counts as both a false positive and a false negative). We add an
additive diagnostic that matches predictions to gold *across labels* and tags
each with one typed error — `COR`, `LE` (labeling), `BES`/`BEL`/`BEO` (boundary
sub-types), `LBE`, `FP`, `FN` — records label confusions, and reports a *fair*
P/R/F1 that charges each near-miss as half an error. It is a selectable regime
(`eval_<cat>_fair_micro_f1`) for entities and event spans, never selected by
default.

**Threshold calibration.** After training, `sweep_thresholds` re-scores the
validation set over a small grid and picks the decision threshold maximizing a
support-weighted micro-F1, writing the choice to the model card and
`threshold_sweep.json`.

## 9. Document-level event extraction via global decoding

### 9.1 Problem

Model trigger↔argument association is **window-bounded**: training uses a
384-word sliding window, and the encoder caps at 512 positions, but WikiEvents
records are whole documents (median ~500 words; 50–70% exceed 512 words).
Evaluation historically fed whole documents to the model in a single pass —
running it out of its trained window and past its position cap. The measured
consequence is that **arguments are the bottleneck**: on the recorded WikiEvents
blind test, `event_type` strict F1 is 0.82–0.93 but `event_argument` strict F1
is only 0.068 (base) / 0.28 (large).

### 9.2 Method

We scan the document with **overlapping** windows (window = training `max_len`;
overlap = 128 words → step 256, matching training's stride), remap each window's
spans to global document offsets, and treat every trigger/argument/entity
candidate — each with a confidence — as a node in a document-level graph. A
**greedy** substrate clusters event mentions across windows by trigger IoU and
unions their arguments (deduplicating by role and span, pooling confidence). A
**beam decoder** then refines the result under global constraints — role
validity (hard), single-filler cardinality, a cross-event span-conflict penalty,
and a trigger-confidence floor — mirroring the local-scoring-plus-global-decode
design of OneIE (Lin et al., 2020). The assembler emits the exact standard
`event_extraction` shape, so all downstream formatting and metrics are unchanged.

### 9.3 Interfaces

The mode is opt-in on three surfaces, all default-off:

- **Inference API**: `batch_extract_long(..., global_decode=True)` /
  `extract_long(...)`.
- **Evaluation**: `compute_metrics(..., chunk_size=…, global_decode=True)`,
  driven from the training YAML (`eval.global_decode`, `eval.chunk_size`,
  `eval.chunk_overlap`); windowing and the decoder are separable knobs so a
  three-way A/B (whole-doc baseline vs. chunk+simple-merge vs. chunk+beam) is
  measurable.
- **CLI**: `tools/infer.py --global-decode`.

### 9.4 Honest limitations

OneIE *learns* its global-feature weights; we have no such training signal, so
the beam uses **heuristic/config-set** weights (`GlobalDecodeConfig`). Recall is
still bounded by within-window candidate recall — an argument more than ~one
window from its trigger is never emitted and cannot be recovered by any
post-hoc decode (this would require trigger-anchored windows or a two-stage
document reader; see [`RECOMMENDATIONS.md`](RECOMMENDATIONS.md)). Trigger
clustering by span overlap can, rarely, merge two adjacent same-type events. The
value the beam adds over greedy is cross-event conflict resolution.
Correspondingly, the mode is only beneficial when documents exceed the encoder's
one-pass window; on a long-context backbone that sees the whole document, chunk-
and-reassemble would be strictly worse — hence opt-in, never default.

## 10. Results

*To be populated from the current training run. The tables below give the
evaluation harness and the pre-existing baselines to compare against.*

### 10.1 WikiEvents blind test (strict micro-F1)

| Model | entity | event_type | event_trigger | event_argument | event |
|---|---|---|---|---|---|
| base-v1 (baseline, whole-doc eval) | — | 0.816 | 0.515 | **0.068** | 0.331 |
| large-v1 (baseline, whole-doc eval) | 0.780 | 0.925 | 0.524 | **0.283** | 0.447 |
| *base-v1 (this run, global_decode)* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

### 10.2 Ablation — windowing vs. global decode (`event_argument` strict F1)

| Configuration | F1 |
|---|---|
| whole-doc single pass (baseline) | *TBD* |
| chunk (overlap 128) + simple merge | *TBD* |
| chunk + OneIE global decode | *TBD* |

*Preliminary signal (5 held-out documents, `large-v1` checkpoint):
`event_argument` strict F1 rose from 0.086 (whole-doc) to 0.135 (chunk + global
decode), with both precision and recall improving. This small-slice result is
illustrative only and will be superseded by the full run.*

### 10.3 Loss-variant comparison

| struct_loss | entity F1 | event_argument F1 | notes |
|---|---|---|---|
| bce / bce_posweight / focal / asl / dice / bce_dice | *TBD* | *TBD* | one config per variant |

## 11. Reproducibility

- **Train** from a raw backbone: `uv run python tools/train/train.py --config
  tools/train/config/<name>.yaml`. WikiEvents configs enable windowed global
  decoding at eval (`eval.global_decode: true`, `chunk_size: 384`,
  `chunk_overlap: 128`).
- **Convert** corpora with the scripts in [`tools/data/`](../data/) (see
  `run_all_converters.sh` and [`TRAINING_DATA.md`](../data/TRAINING_DATA.md)).
- **Infer** at the document level: `tools/infer.py --model <ckpt> --input
  <doc> --events '{"Attack":["Attacker","Target","Place"]}' --global-decode`.
- Design and verification notes:
  [`DOCUMENT_EXTRACTION_PLAN.md`](DOCUMENT_EXTRACTION_PLAN.md),
  [`METRICS.md`](../../METRICS.md), [`CORE_CHANGES.md`](../../CORE_CHANGES.md).

## 12. Limitations and future work

- The global decoder's beam weights are heuristic, not learned; a learned
  global scorer (true OneIE) would need a training signal over the candidate
  graph.
- Long-range arguments beyond one window remain an upstream recall ceiling;
  trigger-anchored windows or a two-stage document-level argument reader are the
  natural next steps (RECOMMENDATIONS Options 2–3).
- Argument coreference is not resolved; document-level benchmarks score against
  any coreferential mention, so measured argument F1 is a lower bound.
- Packaging: several training-only dependencies and the raised Python floor
  should move to a `training` extra before any of this lands in the base
  inference package (see CORE_CHANGES §4).

## 13. References

### Methods and models

- Zaratiana, U., Pasternak, G., Boyd, O., Hurn-Maloney, G., Lewis, A. (2025).
  *GLiNER2: Schema-Driven Multi-Task Learning for Structured Information
  Extraction.* EMNLP 2025 System Demonstrations, pp. 130–140. ACL.
  https://aclanthology.org/2025.emnlp-demos.10/
- Zaratiana, U., Tomeh, N., Holat, P., Charnois, T. (2024). *GLiNER: Generalist
  Model for Named Entity Recognition using Bidirectional Transformer.* NAACL.
  https://aclanthology.org/2024.naacl-long.300/
- Lin, Y., Ji, H., Huang, F., Wu, L. (2020). *A Joint Neural Model for
  Information Extraction with Global Features* (OneIE). ACL.
  https://aclanthology.org/2020.acl-main.713/
- Ortmann, K. (2022). *Fine-Grained Error Analysis and Fair Evaluation of
  Labeled Spans.* LREC. https://aclanthology.org/2022.lrec-1.150/
- Marone, M., Weller, O., Fleshman, W., Yang, E., Lawrie, D., Van Durme, B.
  (2025). *mmBERT: A Modern Multilingual Encoder with Annealed Language
  Learning.* arXiv:2509.06888. https://arxiv.org/abs/2509.06888
- Warner, B., et al. (2024). *Smarter, Better, Faster, Longer: A Modern
  Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning
  and Inference* (ModernBERT). arXiv:2412.13663. https://arxiv.org/abs/2412.13663
- He, P., Gao, J., Chen, W. (2021). *DeBERTaV3: Improving DeBERTa using
  ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.*
  arXiv:2111.09543. https://arxiv.org/abs/2111.09543
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). *Focal Loss
  for Dense Object Detection.* ICCV. https://arxiv.org/abs/1708.02002
- Ben-Baruch, E., et al. (2021). *Asymmetric Loss for Multi-Label
  Classification.* ICCV. https://arxiv.org/abs/2009.14119
- Li, X., Sun, X., Meng, Y., Liang, J., Wu, F., Li, J. (2020). *Dice Loss for
  Data-imbalanced NLP Tasks.* ACL. https://arxiv.org/abs/1911.02855

### Datasets

Datasets used for training/evaluation on this branch (full corpus list and
provenance in [`tools/data/TRAINING_DATA.md`](../data/TRAINING_DATA.md)):

- Li, S., Ji, H., Han, J. (2021). *Document-Level Event Argument Extraction by
  Conditional Generation* (WikiEvents / BART-Gen). NAACL.
- Walker, C., Strassel, S., Medero, J., Maeda, K. (2006). *ACE 2005 Multilingual
  Training Corpus.* LDC2006T06, Linguistic Data Consortium.
- Tong, M., et al. (2022). *DocEE: A Large-Scale and Fine-grained Benchmark for
  Document-level Event Extraction.* NAACL.
  https://aclanthology.org/2022.naacl-main.291/
- Ebner, S., Xia, P., Culkin, R., Rawlins, K., Van Durme, B. (2020).
  *Multi-Sentence Argument Linking* (RAMS). ACL.
  https://aclanthology.org/2020.acl-main.718/
- Wang, X., et al. (2020). *MAVEN: A Massive General Domain Event Detection
  Dataset.* EMNLP. https://aclanthology.org/2020.emnlp-main.129/
- Satyapanich, T., Ferraro, F., Finin, T. (2020). *CASIE: Extracting
  Cybersecurity Event Information from Text.* AAAI.
- Yao, F., et al. (2022). *LEVEN: A Large-Scale Chinese Legal Event Detection
  Dataset.* ACL Findings. https://aclanthology.org/2022.findings-acl.17/
- Zhu, M., et al. (2024). *CMNEE: A Large-Scale Document-Level Event Extraction
  Dataset based on Open-Source Chinese Military News.* LREC-COLING.
  https://aclanthology.org/2024.lrec-main.299/
- Bogdanov, S., et al. (2024). *NuNER: Entity Recognition Encoder Pre-training
  via LLM-Annotated Data.*
- Zhou, W., et al. (2024). *UniversalNER: Targeted Distillation from Large
  Language Models for Open Named Entity Recognition* (Pile-NER). ICLR.
- Yao, Y., et al. (2019). *DocRED: A Large-Scale Document-Level Relation
  Extraction Dataset.* ACL.
- Tan, Q., Xia, L., Bing, L. (2022). *Revisiting DocRED — Addressing the False
  Negative Problem in Relation Extraction* (Re-DocRED). EMNLP.
- Luan, Y., He, L., Ostendorf, M., Hajishirzi, H. (2018). *Multi-Task
  Identification of Entities, Relations, and Coreference for Scientific
  Knowledge Graph Construction* (SciERC). EMNLP.
- Luo, L., et al. (2022). *BioRED: A Rich Biomedical Relation Extraction
  Dataset.* Briefings in Bioinformatics.
- Park, S., et al. (2021). *KLUE: Korean Language Understanding Evaluation.*
  NeurIPS Datasets and Benchmarks.

## Appendix A: module map

| Area | Location |
|---|---|
| Events (schema/engine/api/data) | `gliner2/inference/schema.py`, `engine.py`, `api_client.py`, `training/data.py` |
| Encoder-agnostic loading, losses | `gliner2/model.py` |
| Training infrastructure | `gliner2/training/trainer.py`, `parallel.py`, `eta.py`, `chunking.py` |
| Metrics + fair error analysis | `gliner2/training/metrics.py`, `METRICS.md` |
| Global event decoder | `gliner2/inference/global_decode.py`, `chunking.py` |
| Config-driven training | `tools/train/train.py`, `tools/train/config/` |
| Data converters | `tools/data/convert_*.py` |
| Inference CLI | `tools/infer.py` |

## Appendix B: scope

122 commits on `mmbert_training` vs. `main`. Core-package changes vs. `main` are
catalogued in [`CORE_CHANGES.md`](../../CORE_CHANGES.md); the training tooling in
`tools/train/` and `tools/data/` is net-new to this branch.
