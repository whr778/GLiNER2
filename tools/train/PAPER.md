# Schema-Driven Information Extraction Beyond the Sentence: Event Extraction, Multilingual Training, and Document-Level Global Decoding for GLiNER2

**William Roe**¹ (whr778@gmail.com) and **Claude**² (noreply@anthropic.com)

¹ Project author and maintainer  ·  ² AI assistant (Anthropic, Claude Opus 4.8) — design, implementation, and drafting

*Working paper — engineering and methodology contributions on the `mmbert_training` branch. The from-encoder / long-context mmBERT results and the head-initialization finding (§10.6), plus a first broad-data head-init A/B (§10.7, a documented negative result), are complete (model: `whr778/mmbert-base-rams`); the §10.1–10.5 tables are the fastino DeBERTa-v3 baselines.*

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
  DocEE, RAMS, MAVEN, CASIE, CMNEE (zh).

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
document reader; see [`RECOMMENDATIONS.md`](../events_working_papers/RECOMMENDATIONS.md)). Trigger
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
| D. chunk + beam, config fit on val | 0.129 | 0.402 | 0.387 |

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

**Fitting the decoder on validation (row D) does not change the verdict.**
`sweep_global_decode` calibrates the beam's `GlobalDecodeConfig` on the dev set
exactly as `sweep_thresholds` calibrates the decision threshold (§8) — a grid
over `conflict_penalty` × `min_trigger_conf`, choosing the support-weighted
strict micro-F1. The fit moves `conflict_penalty` 0.5 → 1.0, but
`min_trigger_conf` is entirely inert (every WikiEvents trigger clears the floor),
and on the test set the fitted config lands within noise of the default beam:
event strict +0.009 (0.378 → 0.387), argument strict −0.006 (0.136 → 0.129),
argument relaxed −0.033. So the flat response is **structural, not a tuning
artifact** — the loss is recall-bound (§10.1: 231 FN), which no post-hoc
re-ranking reaches. A fully learned OneIE scorer would optimize the same
recall-bound dimension (§12).

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
held-out test split show no blind test. `event_argument` is given
as strict / relaxed / fair.

<!-- SWEEP_START -->
| Config | entity | event_type | event_trigger | event_argument (S / R / Fair) | event | support |
|---|--:|--:|--:|--:|--:|--:|
| `gliner2-base-v1-casie` | 0.553 | 0.928 | 0.413 | 0.058 / 0.450 / 0.426 | 0.207 | 3454 |
| `gliner2-base-v1-docee` | 0.333 | — | — | — / — / — | — | — |
| `gliner2-base-v1-maven` | 0.742 | — | — | — / — / — | — | — |
| `gliner2-base-v1-mendeley-ed` | 0.741 | — | — | — / — / — | — | — |
| `gliner2-base-v1-rams` | — | 0.993 | 0.935 | 0.462 / 0.686 / 0.614 | 0.693 | 3712 |
| `gliner2-large-v1-casie` | 0.591 | 0.975 | 0.487 | 0.173 / 0.549 / 0.515 | 0.302 | 3454 |
| `gliner2-large-v1-docee` | 0.360 | — | — | — / — / — | — | — |
| `gliner2-large-v1-rams` | — | 1.000 | 0.903 | 0.444 / 0.697 / 0.627 | 0.684 | 3712 |
| `gliner2-large-v1-wikievents` | 0.792 | 0.962 | 0.583 | 0.119 / 0.467 / 0.505 | 0.366 | 876 |
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

### 10.6 From-encoder training and the head-initialization bottleneck

We ran the experiment §12 proposed: train events from a *raw* encoder via
`from_encoder()` (fresh GLiNER2 heads) on RAMS, on the long-context mmBERT-base
backbone (8192, whole document natively; §9.5). As an **encoder-isolation
control** we ran the identical recipe on DeBERTa-v3-base — same fresh heads,
same loss (`bce_posweight`, `pos_weight 4`), same 15 epochs and argument-strict
checkpoint selection — but with DeBERTa's native short-context handling (384-word
windows + global decode). Both contrast with fastino `gliner2-base-v1-rams`,
whose heads are IE-curriculum-pretrained on ~254K examples (Zaratiana et al.,
2025) *before* RAMS fine-tuning. RAMS blind test, strict micro-F1:

| Model | Encoder | Heads | event_type | event_trigger | event_arg (S / R) | event |
|---|---|---|--:|--:|--:|--:|
| mmBERT-base `from_encoder` | mmBERT (8192, native) | fresh | 0.964 | 0.611 | **0.050** / 0.213 | 0.247 |
| DeBERTa-v3-base `from_encoder` (control) | DeBERTa-v3 (384 window) | fresh | 0.981 | 0.612 | **0.042** / 0.204 | 0.340 |
| `gliner2-base-v1-rams` (fastino) | DeBERTa-v3 (384 window) | **pretrained** | 0.993 | 0.935 | **0.462** / 0.686 | 0.693 |

**The long-context model did not recover the argument bottleneck** — and the
control says why. Hold the heads fresh and swap the *encoder* (mmBERT ↔ DeBERTa):
arguments barely move (0.050 vs 0.042), triggers not at all (0.611 vs 0.612).
Hold the *encoder* fixed (DeBERTa-v3) and swap fresh → pretrained heads:
arguments jump ~11× (0.042 → 0.462) and triggers 0.612 → 0.935. The RAMS
argument gap is therefore dominated by **head initialization** — the
IE-curriculum pretraining of the fastino heads — **not** the encoder or the
context window. mmBERT's 8192 tokens cannot help because the binding constraint
is untrained extraction heads, not window-bound truncation (RAMS documents fit
in 512 regardless). This refines §9.5: long context remedies window-bound
*recall loss* on genuinely long documents, but it is **orthogonal** to the
head-competence bottleneck that dominates argument extraction from a cold start.

**What the fastino head-init actually is** (verified against Zaratiana et al., 2025;
see [`FASTINO_GLINER2_TRAINING.md`](../events_working_papers/FASTINO_GLINER2_TRAINING.md)).
The fastino heads were trained **fully supervised** on **254,334 examples** — 53% real
text (news/Wikipedia/legal/PubMed/ArXiv) and 47% synthetic, *all* GPT-4o-annotated
(LLM knowledge-distillation, not human labels) — for just **5 epochs** with a
**differential learning rate** (task heads 2×10⁻⁵, encoder 1×10⁻⁵, so heads adapt ~2×
faster). Crucially, the training tasks were **entities, hierarchical structure
extraction, and classification — no events and no relations.** The competence that
transfers is the **hierarchical-structure head**: a shared span-matching scorer
(bilinear dot-product + sigmoid), a 20-way instance-count MLP, and occurrence-ID
conditioning for per-instance fields. Our event path *reuses those exact heads*
(trigger/argument = the span scorer, multiple mentions = the count head, per-event
arguments = occurrence-ID conditioning), which is why warm-starting a model that never
saw an event still lifts RAMS/WikiEvents arguments — the head-init effect is
**structural, not event-specific**.

The implication is constructive: the remedy is to give a from-encoder model the same
*scale* of structure-extraction supervision. Because it is the structure head that
transfers, the target is **not** ~10⁵–10⁶ *event* documents but a large
**hierarchical-structure / argument** curriculum in the fastino mold (mixed
real+synthetic text, LLM-annotated and validated). [`tools/data/synthetic/`](../data/synthetic/)
builds exactly that shape — broad-label, multi-task supervision (entities, relations,
document-level events with triggers + arguments, classification, structures), either
fully generated or projected as synthetic annotations onto real corpora, at the
~10⁵–10⁶ scale the fastino heads saw. Head-init pretraining, *then* task fine-tuning,
is the path to closing this gap on a long-context backbone.

The WikiEvents from-encoder run (mmBERT, warm-started from the RAMS checkpoint
above) reinforces the same point as a **documented negative result**: with only
206 training documents the argument signal never rises above ~0, so
argument-strict checkpoint selection is degenerate and the model is weak
(entity 0.133, event_type 0.970, event_trigger 0.127, event_argument 0.000). It
is reported here but **not released** — 206 documents cannot substitute for head
pretraining. Only `whr778/mmbert-base-rams` is published.

**Methodology — per-epoch threshold sweep.** `bce_posweight` up-weights positive
spans, shifting the score distribution so a fixed 0.5 decision threshold no
longer reflects the model's operating point: argument-strict F1 *at 0.5*
collapses to near-zero noise for every epoch, making fixed-threshold checkpoint
selection effectively random. We added `eval.metric_sweep`, which evaluates
`metric_for_best` each epoch at the threshold that maximizes it over a small grid
(0.1–0.9), so the saved checkpoint is the genuine argument peak at its own
calibrated operating point. The end-of-run blind-test sweep keeps the aggregate
support-weighted objective (§8) for apples-to-apples comparison with the
baselines — which, for argument-sparse categories, *understates* the peak, so the
from-encoder argument numbers above are conservative.

### 10.7 Head-init pretraining: a first broad-data A/B (negative result)

§10.6 argued the remedy for the argument bottleneck is to give a from-encoder
model the same IE curriculum the fastino heads saw. We ran a first, deliberately
cheap version of that experiment and report it as a **negative result**.

A broad **combined base** — mmBERT-base `from_encoder`, warmed for 2 epochs on our
multi-task synthetic corpus (`synthetic_sonnet5`: entities, relations, events,
classification, structures), GLiNER multilingual NER, GLiNER multi-task NER, and
RAMS document events (~96K records, `eval_loss` checkpoint selection) — was then
fine-tuned on WikiEvents under the **identical** 15-epoch, argument-strict recipe
used for the RAMS-only base (`whr778/mmbert-base-rams`). Only the base checkpoint
differs, isolating the base-data effect. WikiEvents blind test (20 docs; 239
trigger / 515 argument mentions), strict micro-F1:

| WikiEvents fine-tune | base | event_type | event_trigger | event_argument (S) |
|---|---|--:|--:|--:|
| control | RAMS-only (`mmbert-base-rams`) | **0.944** | 0.085 | 0.0046 |
| treatment | broad combined base | 0.573 | **0.133** | 0.0066 |

The control reproduces the near-floor WikiEvents-from-RAMS result of §10.6
(event_argument ~0, event_type ~0.95, event_trigger ~0.1) within run-to-run noise
on 20 documents. **The broad+synthetic base gave no reliable downstream lift.**
Argument-strict F1
stays at the floor for both (0.007 vs 0.005; identical 0.004 recall — roughly two
correct arguments each, i.e. noise on 20 documents). The trigger edge to the
treatment (0.133 vs 0.085) is precision-only on 239 mentions, within run-to-run
noise, and the broad base *regressed* event_type (0.573 vs 0.944). For context, the
combined base's own RAMS argument head was also weak (arg-strict 0.028) — though
under a lighter regime than the 0.050 RAMS-only figure in §10.6 (2 epochs +
`eval_loss` selection vs 15 epochs + argument-strict selection), so it is context,
not a controlled base-to-base claim; the controlled comparison is the WikiEvents
A/B above.

This does **not** refute §10.6 — it shows a *light* broad-data pass is not a
substitute for real head-init pretraining. The confounds point the same way: only
2 base epochs, `eval_loss` selection (dominated by the ~77K multilingual-NER
records), and argument dilution mean the argument head was never warmed at the
~10⁵–10⁶ scale, argument-strict-selected, that the fastino curriculum used. Both
combined checkpoints are retained privately (`whr778/mmbert-base-combined`,
`whr778/mmbert-base-combined-wikievents`), not released.

**Sanity check — the synthetic data itself trains cleanly.** To rule out "bad
synthetic data" as the cause of the null result above, we fine-tuned the pretrained
`fastino/gliner2-base-v1` on `synthetic_sonnet5_1k` *alone* (warm-start, 10 epochs,
`eval_loss` selection). On the synthetic held-out split it scores well across all five
tasks — strict micro-F1: entity 0.904, relation 0.657, event-type 0.956, event-trigger
0.838, event-argument **0.702** (0.894 relaxed), classification 0.835. So the synthetic
corpus is coherent and learnable (in-distribution — this is not a claim about the public
benchmarks); the combined-base null result is about the **head-init regime** (2 epochs,
`eval_loss` selection, NER dilution), not the data. Checkpoint:
`whr778/gliner2-base-v1-synthetic`.

**Conversely, the synthetic data cannot bootstrap fresh heads.** Training the *same* data
from a raw DeBERTa-v3-base encoder (`from_encoder`, fresh heads, 15 epochs) instead of
warm-starting collapses every span/relation task: strict micro-F1 entity 0.141, relation
0.000, event-trigger 0.221, event-argument 0.000 (0.168 relaxed), classification 0.356 —
only coarse event-type (0.998, few classes) survives. So ~1.5K synthetic records are an
*adaptation* corpus, not a from-scratch pretraining set: the extraction heads need either
a warm start or the fastino curriculum's ~10⁵–10⁶ scale. This restates the head-init
thesis (§10.6) from the data side. Checkpoint: `whr778/deberta-base-fromenc-synthetic`.

### 10.8 How much data warms the head? An mmBERT data-scaling curve

§10.7's light broad pass did not lift arguments, but it was confounded (2 epochs,
NER-diluted, `eval_loss`-selected). This experiment isolates the one variable that
matters — **Stage-A corpus size** — and measures it directly. We warm mmBERT-base
fresh heads (`from_encoder`) on a structure/argument-dense event corpus of size N,
then fine-tune each on RAMS under the **fixed** `mmbert-base-rams` recipe (15 epochs,
`bce_posweight` 4.0, native long-context, argument-strict selection), and read RAMS
blind-test argument-strict F1 vs N. The Stage-A corpus is assembled from the
multilingual event corpora already on disk (docee, chfinann, docfee, duee, cmnee,
maven, text2json, events_biotech, mendeley_ed, casie), nested and proportional at
10K/40K/100K, with RAMS/WikiEvents held out (leakage). Spec:
`events_working_papers/SCALING_CURVE_EXPERIMENT.md`.

Two endpoints are already known on their respective encoders: **N=0 = 0.050**
(mmBERT `from_encoder` straight to RAMS, §10.6); the DeBERTa-v3 fastino warm-start
(254K) reaches **0.462** as a *cross-encoder reference*, not an mmBERT point (there
is no fastino-scale warm-start on mmBERT). RAMS blind test (871 docs), strict micro-F1:

| Stage-A size N | event_argument (S) | event_trigger (S) | event_type (S) |
|--:|--:|--:|--:|
| 0 (fresh → RAMS) | 0.050 | 0.611 | — |
| 10K | 0.050 | 0.598 | 0.952 |
| 40K | **0.115** | 0.706 | 0.931 |
| ~100K | _(running)_ | | |
| 254K (DeBERTa ref) | 0.462 | 0.935 | — |

**10K of head-warming moved the argument head essentially zero** (0.050, identical
to the N=0 floor), but **40K lifts it ~2.3× to 0.115** (trigger 0.611 → 0.706 as
well), so **the knee falls between 10K and 40K**: mmBERT's argument head *does* warm
from broad structure/argument data, just at a higher threshold than DeBERTa and
confirming the prediction that a cold multilingual encoder needs more of it. The
~100K point is training; it will show whether the curve keeps climbing toward the
DeBERTa reference (0.462) or plateaus. Checkpoints (private):
`whr778/scaling-mmbert-{10k,40k,100k}` and `-rams`.

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
  [`DOCUMENT_EXTRACTION_PLAN.md`](../events_working_papers/DOCUMENT_EXTRACTION_PLAN.md),
  [`METRICS.md`](../../METRICS.md), [`CORE_CHANGES.md`](../events_working_papers/CORE_CHANGES.md).

## 12. Limitations and future work

- We ran the long-context mmBERT experiment (§10.6) and the result overturned the
  expectation: a from-encoder mmBERT does **not** recover the argument bottleneck,
  because — as the DeBERTa-v3 encoder-isolation control shows — that bottleneck is
  **head initialization** (IE-curriculum pretraining), not the context window.
  Long context still remedies window-bound recall loss on genuinely long
  documents (its intended role, §9.5), but it is orthogonal to head competence.
  We ran a first, cheap version of the head-init A/B (§10.7): a 2-epoch,
  `eval_loss`-selected broad+synthetic base did **not** lift downstream WikiEvents
  events (arguments stayed at the floor; event_type regressed). The open work is
  therefore a *heavier* head-init pass — the broad-label, multi-task synthetic
  curriculum in [`tools/data/synthetic/`](../data/synthetic/) at ~10⁵–10⁶ scale
  with **argument-strict checkpoint selection**, matching the fastino curriculum —
  rather than a light multi-task warm-up, then re-run RAMS/WikiEvents and A/B
  against the fastino heads.
- The global decoder's weights are **heuristic, not learned**: the beam
  optimizes one fixed objective — summed argument confidence minus a flat
  `conflict_penalty` per reused span, under hard trigger-floor and single-filler
  rules (`GlobalDecodeConfig`) — with no graph-level feature vector. A true
  OneIE-style scorer would add global features over the candidate graph (role
  co-occurrence, cross-event argument sharing, entity-type↔role compatibility)
  with weights fit by **structured-perceptron beam training** against gold
  document graphs; WikiEvents/RAMS supply the graphs, and the windowed candidates
  this module already emits are the training substrate. We implemented the lighter
  intermediate — `sweep_global_decode` fits the `GlobalDecodeConfig` scalars on
  validation, as `sweep_thresholds` fits the decision threshold (§8) — and
  measured it within noise on the WikiEvents test set (§10.2, row D), so the flat
  response is structural, not a tuning artifact; only the fully learned scorer
  remains open. Either way the gain is **precision, not recall**: the scorer only re-ranks candidates already generated, so it cannot
  recover an argument that fell outside every window — the dominant error term
  (§10.1: 231 FN vs. 123 FP) — which is why the beam is already neutral over the
  greedy merge (§10.2). Crucially, a long-context backbone does **not** retire
  this decoder: the windowed regime reappears on mmBERT whenever the *trained*
  `max_len` is narrower than the document — from genuine overflow (documents past
  8192: common for book-length corpora, rare for WikiEvents) or from training at
  a reduced `max_len` for compute (8192-token attention is quadratic; §9.5). Long
  context shrinks the population where the decoder fires, it does not eliminate
  it; and as the window grows, more trigger↔argument pairs are already co-located
  within it, so the residual decode collapses toward the dedup the greedy merge
  already performs. The learned scorer is therefore the polish for the windowed
  regime whenever it is active — second-order to the within-window recall ceiling
  throughout.
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
- Adelani, D.I., et al. (2022). *MasakhaNER 2.0: Africa-centric Transfer Learning
  for Named Entity Recognition.* EMNLP.
  https://aclanthology.org/2022.emnlp-main.298/
- Adelani, D.I., et al. (2023). *MasakhaNEWS: News Topic Classification for
  African Languages.* IJCNLP-AACL.
  https://aclanthology.org/2023.ijcnlp-main.10/
- Pan, X., Zhang, B., May, J., Nothman, J., Knight, K., Ji, H. (2017).
  *Cross-lingual Name Tagging and Linking for 282 Languages* (WikiANN). ACL.
  https://aclanthology.org/P17-1178/
- Rahimi, A., Li, Y., Cohn, T. (2019). *Massively Multilingual Transfer for NER*
  (PAN-X balanced splits). ACL. https://aclanthology.org/P19-1015/

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
catalogued in [`CORE_CHANGES.md`](../events_working_papers/CORE_CHANGES.md); the training tooling in
`tools/train/` and `tools/data/` is net-new to this branch.

## Appendix C: Running the models — Hub download and the viewer

The fine-tuned checkpoints are published as public Hub repos `whr778/<config>`
(e.g. `whr778/gliner2-large-v1-docee`). Two steps to obtain and run them.

**Download from the Hub.** `scripts/pull_from_hf.sh` fetches each model into
`out/fastino/<config>/best/` (where the viewer auto-discovers it) via `hf
download` with Xet high-performance transfer — parallel and CDN-accelerated, so
far faster than a single rsync-over-SSH stream on a high-latency link:

```bash
bash scripts/pull_from_hf.sh                         # all trained models
bash scripts/pull_from_hf.sh gliner2-large-v1-docee  # or specific ones
```

**Launch the viewer.** An interactive app (NextJS + FastAPI) that runs any local
checkpoint over pasted / uploaded / URL-imported text and renders entities,
relations, events, classifications, and structures with confidences:

```bash
cd viewer/frontend && npm install     # one-time
bash viewer/viewer.sh start           # backend :8000 + frontend :3000 (waits until up)
# open http://localhost:3000 ; stop both with: bash viewer/viewer.sh stop
```

In the browser, pick a model in the **Model** box (downloaded checkpoints are
listed automatically) — its training-time schema loads to match — then enter
text and click **Extract**. Set a default model at launch with
`GLINER2_MODEL=out/fastino/gliner2-large-v1-docee/best bash viewer/viewer.sh start`.

**Or run headless** at the document level from the CLI:

```bash
uv run python tools/infer.py --model out/fastino/gliner2-large-v1-docee/best \
  --input <doc.txt> --events '{"Attack":["Attacker","Target","Place"]}' --global-decode
```
