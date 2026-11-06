# Coreference training data: sources, licenses, coverage

**Status update 2026-11-06 (`eb22g`, `PAPER_3_DOCUMENT_EVENTS_AND_COREF.md`):** This document now covers all three coreference supervision channels active in `eb22g`:
1. **Entity-mention coreference (`entity_coref_link: true`)** — trained on the four LLM-annotated news/event coreference corpora (`*_coref_llm`: **910,782 clusters, 1,808,625 mentions, 40.6% pronouns** across 27,152 train docs) plus `cmnee` and `wikievents`. The out-of-genre `gum_coref`, `litbank_coref`, and `gap_coref` corpora were dropped from training after `corefC` (`aab28b3`) and are retained as held-out out-of-genre evaluation splits.
2. **Argument coreference expansion (`training.expand_argument_coref: true`)** — `expand_argument_coreferents` (`gliner2/processing/argument_coref.py`) attaches `coref_clusters` from the `*_coref_llm` records onto matching event records by normalized text (`11,335` event records matched in `eb22g`, `27,158` coref-only records retained for `entity_coref_link`), tagging coreferent argument mentions with `_coref_group` so `_build_record_field_target` compiles them as **span alternatives of a single required filler** (`len(ft.values) = 1`, fixed in `0607b29`).
3. **Trigger × trigger event coreference (`record_coref_link: true`, `record_coreferent_ownership: true`)** — trained on multi-trigger event clusters in `cc_news_events_sonnet55_v2` (`3,760 / 361 / 355` multi-trigger events across `train / val / test`, `27.1%` of events; `9,528` positive coreferent trigger pairs and `15,622` hard-negative same-type trigger pairs in train).

Every entity-coreference corpus stores clusters the same way, a top-level `coref_clusters`: a list of clusters, each a list of mentions `{text, start, end}` with code-point offsets into the record's cleaned `input` (`tools/data/add_coref_clusters.py` and `tools/data/annotate_coref.py`).

## 1. Entity & Argument Coreference Sources (`coref_clusters`)

### 1a. Active in `eb22g` Training (`train_only` for `*_coref_llm`)

| corpus (`data/`, Hub) | language | what it holds | size (`train / val / test` docs; total clusters / mentions) | pronoun share | license / provenance | role in `eb22g` |
|---|---|---|---|---|---|---|
| `cc_news_haiku45_coref_llm` (`whr778/cc_news_haiku45_coref_llm`) | en | LLM-clustered entity + pronoun coref over `cc_news_haiku45` (`annotate_coref.py`, voted eval draws) | `15,806 / 1,696 / 1,670` docs; **520,438 clusters / 1,007,289 mentions** | **40.9%** | CC-News text + Haiku 4.5 silver clusters (private) | **Active** (`entity_coref_link` + `expand_argument_coref`) |
| `cc_news_events_haiku45_coref_llm` (`whr778/cc_news_events_haiku45_coref_llm`) | en | LLM-clustered entity + pronoun coref over `cc_news_events_haiku45` | `6,549 / 717 / 810` docs; **211,066 clusters / 431,267 mentions** | **42.7%** | CC-News text + Haiku 4.5 silver clusters (private) | **Active** (`entity_coref_link` + `expand_argument_coref`) |
| `cc_news_events_sonnet55_v2_coref_llm` (`whr778/cc_news_events_sonnet55_v2_coref_llm`) | en | LLM-clustered entity + pronoun coref over `cc_news_events_sonnet55_v2` | `4,584 / 439 / 457` docs; **157,362 clusters / 334,669 mentions** | **36.7%** | CC-News text + Haiku 4.5 silver clusters (private) | **Active** (`entity_coref_link` + `expand_argument_coref`) |
| `wikievents_coref_llm` (`whr778/wikievents_coref_llm`) | en | LLM-clustered entity + pronoun coref over `wikievents` (near-dup train doc 113 removed) | `213 / 18 (dev) / 19` docs; **21,916 clusters / 35,400 mentions** | **17.3%** | WikiEvents text + Haiku 4.5 silver clusters (private) | **Active** (`entity_coref_link` + `expand_argument_coref`) |
| `cmnee` (`whr778/cmnee`) | zh | human-annotated ARGUMENT clusters (`coref_arguments`), events too | `16,599 / 2,794 / 4,710` clusters | **~0%** (1 of 55,465) | see source ([CMNEE](https://github.com/2086482524/CMNEE)) | **Active** (Chinese name/alias coreference) |
| `wikievents` (`whr778/wikievents`) | en | human-annotated ENTITY clusters (`coref/*.jsonlines`), events + entities too | `200 / 20 / 20` docs (`4,636 / 402 / 451` clusters) | high (44% of val pairs) | see source ([gen-arg S3](https://github.com/raspberryice/gen-arg)) | **Active** (human gold eval + train) |

### 1b. Out-of-Genre Evaluation Corpora (Dropped from Training After `corefC`, `aab28b3`)

| corpus (`data/`, Hub) | language | what it holds | size (`train / val / test`) | pronoun share | license | origin |
|---|---|---|---|---|---|---|
| `gum_coref` (`whr778/gum_coref`) | en | full clusters incl. singletons, 15 genres (news, interview, academic, fiction, ...) | `149 / 16 / 26` docs, 27,260 clusters, 52,716 mentions | 21.0% | **per genre**, recorded per record | [CorefUD 1.2 English-GUM](https://huggingface.co/datasets/FrancophonIA/CorefUD_1.2), [GUM](https://gucorpling.org/gum/) |
| `litbank_coref` (`whr778/litbank_coref`) | en | full clusters, ~2,000-word fiction excerpts | `72 / 8 / 10` docs, 7,052 clusters, 26,261 mentions | 51.8% | CC-BY-4.0 | [CorefUD 1.2 English-LitBank](https://huggingface.co/datasets/FrancophonIA/CorefUD_1.2), [LitBank](https://github.com/dbamman/litbank) |
| `gap_coref` (`whr778/gap_coref`) | en | one ambiguous pronoun + two candidate names per Wikipedia passage | `2,000 / 454 / 2,000` passages | 33.3% (by construction) | Apache-2.0 | [GAP](https://huggingface.co/datasets/coref-data/gap_raw) ([paper](https://arxiv.org/abs/1810.05201)) |

## 2. Event-Trigger Coreference Sources (`triggers: [...]` on `output.events`)

| corpus (`data/`, Hub) | language | what it holds | size (`train / val / test` docs) | multi-trigger events | positive / hard-neg trigger pairs (train) | role in `eb22g` |
|---|---|---|---|---|---|---|
| `cc_news_events_sonnet55_v2` (`whr778/cc_news_events_sonnet55_v2`) | en | LLM-annotated news events listing all coreferent triggers per event (`ENGLISH_ANNOTATION_SPEC.md`) | `4,597 / 500 / 500` docs (`13,855 / 1,470 / 1,474` events) | **`3,760 / 361 / 355`** (`27.1% / 24.6% / 24.1%`) | **9,528 positive / 15,622 hard-negative** (same-type, different-cluster) | **Primary training & `eval.coref_calibration` source** for `TriggerLink` and `record_coreferent_ownership: true` |
| `cmnee` & `maven` | zh / en | single-trigger event corpora (`0%` multi-trigger) | `cmnee.val` (`1,606` docs), `maven.val` (`402` docs) | **0** (`0.0%`) | 0 positive, all same-type pairs negative | **`eval.coref_control` corpora** in `coref_calibration.py` (ensures trigger merging does not fuse distinct events; < 5% fusion gate) |

**GUM licenses, per genre** (from GUM's LICENSE.md; annotations are CC-BY-4.0 throughout):
academic, court CC-BY-4.0; bio, voyage CC-BY-SA-3.0; news, interview CC-BY-2.5;
**essay, fiction, letter, podcast, whow CC-BY-NC-SA (non-commercial)**; **conversation, speech,
textbook, vlog UNVERIFIED** (not listed in LICENSE.md, 51 of 191 docs). Filter on
`provenance.license` before any commercial distribution of a model trained on them.

**Not used:** PreCo (large, pronouns included, but `license: unknown` on the Hub, not verified);
OntoNotes (LDC-licensed). ReDocRED's `vertexSet` clusters exist but are 8% multi-surface and
carry no pronouns.

## 3. How Each Was Built

- **`*_coref_llm` (`cc_news_haiku45_coref_llm`, `cc_news_events_haiku45_coref_llm`, `cc_news_events_sonnet55_v2_coref_llm`, `wikievents_coref_llm`):** Built by `tools/data/annotate_coref.py` (following `tools/data/coref_llm_pilot.py`). Candidate mentions (named entities, nominals, and pronouns) are numbered in document order and clustered by Haiku 4.5 in batch mode, with multi-run self-consistency voting on the evaluation draws (`val` and `test`). Each record carries `coref_clusters` with verified character offsets into `input`.
- **cmnee, wikievents:** augmented IN PLACE on the deduped disk files (`add_coref_clusters.py`,
  every other byte asserted unchanged); the converters emit the same clusters for a fresh build.
- **gum_coref, litbank_coref:** `convert_corefud.py --corefud <corpus>` (downloads from the
  CorefUD mirror). CorefUD publishes train and dev only, so dev is TEST and a seeded 10% of train
  documents is VAL. Contractions are placed inside their token (`It's` -> `It`).
- **gap_coref:** `convert_gap.py` (downloads `coref-data/gap_raw`); GAP's own splits. The
  coreferent name joins the pronoun's cluster, the distractor is a singleton (a hard negative).
  3 passages repeat inside a split with a different pronoun, by design.

All are registered in `dataset_registry.yaml` (`hf_jsonl`) and restorable with `restore_from_hf.py`. Uniqueness: 0 cross-split overlaps (`coref_eval_near_dup_skip.json` records near-duplicate exclusions).

## Language coverage

CorefUD 1.2 also covers Catalan, Czech, French, German, Hungarian, Lithuanian, Norwegian (Bokmal,
Nynorsk), Polish, Russian, Spanish and **Turkish** (plus three historical languages); the same
converter handles them. **It has no Chinese**: Chinese name/alias coreference is supervised by
CMNEE, Chinese PRONOUN coreference by nothing yet.

## What the data buys (probe A, eb21 epoch 3, VAL pair AUC; EXPERIMENT_CATALOG 2026-10-15)

| | string match | untrained cosine | head-only |
|---|---|---|---|
| CMNEE, different surfaces | 0.500 | 0.877 | 0.887 |
| WikiEvents, pronoun pairs | 0.548 | 0.578 | 0.724 (WikiEvents-only head) / 0.676 (mixed-English head) |
| GUM, pronoun pairs | 0.709 | 0.671 | 0.878 (mixed head) |
| LitBank, pronoun pairs | 0.628 | 0.632 | 0.846 (mixed head) |
| GAP, pronoun pairs | 0.500 | 0.563 | 0.510 (mixed head) |

A frozen encoder with a linear head learns each corpus's own coreference but does not transfer
the extra English to WikiEvents, and GAP's ambiguous pronouns are at chance. Encoder training
(option B) is the next test; LLM-annotated English news (option 4) is priced in the catalog.

## What the data buys, trained (corefC, 2026-10-16 to 2026-10-17; EXPERIMENT_CATALOG)

corefC warm-started eb21 on 34% exact replay plus the bought LLM news coref. The arms were control (link
weight 0), treatment (0.3), and treatment-free (0.3 + GUM/LitBank/GAP).

- **Linking, val (pair AUC), WikiEvents dev pronouns:** treatment link 0.811 vs control cosine 0.685. The
  document-bootstrap floor puts the gain at +0.127, 95% CI [+0.088, +0.179] (20 docs). It is the first link
  that learns news pronouns.
- **GUM/LitBank/GAP add little on news:** treatment-free minus treatment, WikiEvents pronoun -0.023
  [-0.041, +0.013], unresolved. GAP's own task gains +0.041 [+0.013, +0.068]. Dropped from training (aab28b3);
  their val splits stay as an out-of-genre check.
- **Linking, blind test (gold mentions, LLM-silver gold),** cut-off picked on val (link logit +0.20):
  - mention-pair F1 0.598 vs string match 0.437;
  - pronoun pairs 0.591 vs 0.353.
  - CPU, MPS and CUDA agree.
- **End to end (predicted mentions), v2 val:**
  - mention-pair F1 0.452 vs 0.588 with gold mentions;
  - pronoun pairs 0.499.
  - 30% of gold coreferent mentions are never proposed (candidate recall 0.698), so mention detection, not
    the link, caps recall.
- eb21's other heads: corefC treatment minus control is inside the +/-0.02 floor on every blind-test head.

The coref eval purchase found WikiEvents test #10 92% inside a train document. Fixed 2026-10-17 by dropping
train doc 113 from `wikievents` and `wikievents_coref_llm`
(`tools/data/dedupe_splits.py --near-dup 0.2`). eb21 and the corefC arms trained before the fix, so their
WikiEvents test scores include it.
