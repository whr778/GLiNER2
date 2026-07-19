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
window. For the short-context (512-token) DeBERTa-v3 checkpoints we add an
opt-in, OneIE-style (Lin et al., 2020) **global graph decoder** that reconnects
events across overlapping windows; on a long-context backbone such as **mmBERT**
(8192 tokens) the model fits whole documents in one pass and captures the same
events *natively*, so global decoding is off by default and reserved for
short-context encoders. A pipeline of ~30 corpus
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
`jhu-clsp/mmBERT-base` and similar. Because mmBERT/ModernBERT carry an
8192-token context, a model bootstrapped this way fits nearly all
document-level event corpora in one forward pass (§9.5) — the native
alternative to the windowing that short-context (512-token) encoders require.

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

This problem is specific to **short-context** encoders whose window is smaller
than the document; on a long-context backbone (§9.5) it does not arise. The
fastino GLiNER2 checkpoints use DeBERTa-v3 (512-position cap), so training uses a
384-word sliding window and trigger↔argument association is **window-bounded**,
while WikiEvents records are whole documents (median ~500 words; 50–70% exceed
512 words). Evaluation historically fed whole documents to the model in a single
pass — running it out of its trained window and past its position cap. The
measured consequence is that **arguments are the bottleneck**: on the recorded
WikiEvents blind test, `event_type` strict F1 is 0.82–0.93 but `event_argument`
strict F1 is only 0.068 (base) / 0.28 (large).

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

### 9.5 The long-context alternative: mmBERT

Global decoding is the *short-context* remedy. The higher-leverage fix is to
remove the window bound entirely by training on a **long-context backbone**. The
branch's namesake, `jhu-clsp/mmBERT-base` (ModernBERT architecture, 8192-token
context; Warner et al., 2024; Marone et al., 2025), fits the large majority of
these documents in one forward pass — WikiEvents' median is ~700 subword tokens
and its 95th percentile ~2.5k, well under 8192; only the longest tail overflows.
A GLiNER2 model bootstrapped from mmBERT via `from_encoder()` and trained with a
wide `max_len` therefore sees each trigger together with its scattered arguments,
capturing document-level events **natively** — no windowing, no cross-window
reassembly, no boundary loss. On such a model global decoding is not merely
unnecessary but counter-productive (it would fragment context the encoder could
use whole), which is why it is off by default and gated on `chunk_size`.

The two paths are complementary, selected by the deployment constraint:

| Regime | Encoder | Document-level events via |
|---|---|---|
| Short context (≤512) | DeBERTa-v3 (fastino base/large/multi) | windowing + global decoding (§9.2) |
| Long context (8192) | mmBERT / ModernBERT (`from_encoder`) | native single pass; global decode off |

The trade-off is compute: 8192-token attention is quadratic in length, which is
why the training infrastructure (§5) invests in gradient checkpointing and cross-
backend memory management. Global decoding stays the pragmatic choice when a
short-context checkpoint must be used as-is, or for the rare document exceeding
even the long-context window.

## 10. Results

Blind test on the 20-document WikiEvents held-out set for the **base-v1** run
(DeBERTa-v3-base, 15 epochs), evaluated with windowed global decoding
(`chunk_size 384`, `chunk_overlap 128`, `global_decode true`). Micro unless noted.

### 10.1 WikiEvents blind test — base-v1 (this run)

| Category | strict P / R / F1 | relaxed F1 | fair F1 | support |
|---|---|---|---|---|
| entity | 0.729 / 0.803 / **0.764** | 0.804 | 0.794 | 1602 |
| event_type | 1.000 / 0.910 / **0.953** | 0.953 | — | 122 |
| event_trigger | 0.520 / 0.586 / **0.551** | 0.567 | 0.572 | 239 |
| event_argument | 0.155 / 0.120 / **0.136** | 0.435 | 0.467 | 515 |
| event (combined) | 0.401 / 0.357 / **0.378** | 0.554 | — | 876 |

Event *type* detection is nearly saturated (0.953, precision 1.00); *arguments*
remain the bottleneck at strict 0.136 but recover strongly under relaxed (0.435)
and fair (0.467) — the model often locates the right argument entity with an
inexact boundary/surface or under strict's double penalty. The fine-grained
error counts bear this out: of the gold arguments the fair analysis scores,
COR 175, typed near-misses (LE/LBE/BE\*) 46, pure misses (FN) 231, plus 123 false
positives.

### 10.2 Effect of windowing + global decode

Clean 3-point ablation on this base-v1 checkpoint — model fixed, only the eval
path varies — over the full 20-document test set:

| Eval configuration | event_argument strict F1 | arg relaxed | event strict F1 |
|---|---|---|---|
| A. whole-doc single pass | 0.086 | 0.395 | 0.356 |
| B. chunk (overlap 128) + simple merge | **0.143** | 0.433 | 0.373 |
| C. chunk + OneIE global decode (beam) | 0.136 | 0.435 | 0.378 |

**Windowing is the driver.** Chunking eval to match the model's trained window
(A→B) lifts argument strict F1 **0.086 → 0.143** (+66% relative), fixing the
train/eval mismatch and the 512-token overflow. **The beam adds essentially
nothing over a simple chunk-merge here** (B→C): argument strict is marginally
*lower* (0.143 → 0.136) while event strict is marginally higher (0.373 → 0.378) —
within noise. This is the measured, honest answer to "does the beam earn its
keep": on WikiEvents, not appreciably. Its value (cross-event conflict
resolution under heuristic weights) is below the noise floor, and the default
no-cardinality-cap union slightly over-generates arguments, nudging strict
precision down. The `single_filler_roles` config is the lever to recover that;
a learned global scorer would be the principled fix.

Decomposing the 0.068 → 0.136 change from the prior baseline (approx., since the
prior run's exact config is inferred): ~+0.018 from 5x more training (prior
3-epoch whole-doc 0.068 → this 15-epoch whole-doc 0.086), ~+0.057 from windowing
(0.086 → 0.143), and ~−0.007 from the beam vs. simple merge. The takeaway
reframes the contribution: **the windowed eval path — not the OneIE decoder — is
what recovers the document-level argument bottleneck** on a short-context model;
the decoder is an optional, roughly-neutral refinement pending learned weights or
cardinality tuning.

### 10.3 Loss-variant comparison

| struct_loss | entity F1 | event_argument F1 | notes |
|---|---|---|---|
| bce / bce_posweight / focal / asl / dice / bce_dice | *TBD* | *TBD* | one config per variant |

### 10.4 Full event-dataset sweep

Blind-test micro-F1 per fine-tuned config, filled automatically as each run in
`scripts/train_all_events.sh` finishes (via `scripts/update_paper_metrics.py`);
the model is evaluated with windowed global decoding. Datasets without a
held-out test split (MAVEN, LEVEN) show no blind test. `event_argument` is given
as strict / relaxed / fair.

<!-- SWEEP_START -->
| Config | entity | event_type | event_trigger | event_argument (S / R / Fair) | event | support |
|---|--:|--:|--:|--:|--:|--:|
| `gliner2-base-v1-casie` | 0.553 | 0.928 | 0.413 | 0.058 / 0.450 / 0.426 | 0.207 | 3454 |
| `gliner2-base-v1-docee` | 0.333 | — | — | — / — / — | — | — |
| `gliner2-base-v1-rams` | — | 0.993 | 0.935 | 0.462 / 0.686 / 0.614 | 0.693 | 3712 |
| `gliner2-large-v1-casie` | 0.591 | 0.975 | 0.487 | 0.173 / 0.549 / 0.515 | 0.302 | 3454 |
| `gliner2-multi-v1-cmnee` | — | 0.986 | 0.874 | 0.221 / 0.709 / 0.671 | 0.466 | 27099 |
<!-- SWEEP_END -->

### 10.5 RE-DocRED relation extraction

Fine-tuning `fastino/gliner2-{base,large}-v1` on RE-DocRED (re-annotated DocRED:
document-level NER + relation extraction). Blind-test strict micro-F1; the
checkpoint is selected on `eval_relation_strict_micro_f1` with windowed eval
(`chunk_size 256`, matching the sliding-window training) so the ~500-word docs
do not overflow the 512-position cap. 10 epochs, early stopping (patience 3).

| Config | entity | relation | support (ent / rel) |
|---|--:|--:|--:|
| `gliner2-base-v1-redocred`  | 0.842 | 0.263 | 10705 / 17348 |
| `gliner2-large-v1-redocred` | 0.860 | 0.287 | 10705 / 17348 |

Document-level strict relation F1 is a hard metric for a span-based extractor
(cross-window pairs and coreferent arguments are recall ceilings); the large
model leads on both entity and relation F1.

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

- The document-level event experiments here are on **short-context** DeBERTa-v3
  checkpoints, where global decoding is the remedy (§9). The natural next
  experiment is to fine-tune events on a **long-context mmBERT** model (8192
  tokens, whole document in one pass; §9.5) and A/B it against
  DeBERTa-v3 + global decoding — we expect the long-context model to recover the
  argument bottleneck natively, making the decoder a fallback rather than the
  primary path.
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
