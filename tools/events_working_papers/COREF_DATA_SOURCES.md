# Coreference training data: sources, licenses, coverage

Entity coreference supervision for the entity-mention link head (coref plan step 3): which corpora
carry gold clusters, under what license, and what each contributes. Every corpus stores clusters the
same way, a top-level `coref_clusters`: a list of clusters, each a list of mentions
`{text, start, end}` with code-point offsets into the record's cleaned `input`
(`tools/data/add_coref_clusters.py`). Training ignores the field until the head reads it (traced:
identical dataset items, train and eval batches with and without it).

## Sources

| corpus (`data/`, Hub) | language | what it holds | size (train / val / test) | pronoun share | license | origin |
|---|---|---|---|---|---|---|
| `cmnee` (`whr778/cmnee`) | zh | ARGUMENT clusters (`coref_arguments`), events too | 16,599 / 2,794 / 4,710 clusters | **~0** (1 of 55,465) | see source | [CMNEE](https://github.com/2086482524/CMNEE) |
| `wikievents` (`whr778/wikievents`) | en | ENTITY clusters (`coref/*.jsonlines`), events + entities too | 4,636 / 402 / 451 clusters (200 / 20 / 20 docs) | high (pronoun pairs 44% of val pairs) | see source | [gen-arg S3](https://github.com/raspberryice/gen-arg) |
| `gum_coref` (`whr778/gum_coref`) | en | full clusters incl. singletons, 15 genres (news, interview, academic, fiction, ...) | 149 / 16 / 26 docs, 27,260 clusters, 52,716 mentions | 21.0% | **per genre**, recorded per record | [CorefUD 1.2 English-GUM](https://huggingface.co/datasets/FrancophonIA/CorefUD_1.2), [GUM](https://gucorpling.org/gum/) |
| `litbank_coref` (`whr778/litbank_coref`) | en | full clusters, ~2,000-word fiction excerpts | 72 / 8 / 10 docs, 7,052 clusters, 26,261 mentions | 51.8% | CC-BY-4.0 | [CorefUD 1.2 English-LitBank](https://huggingface.co/datasets/FrancophonIA/CorefUD_1.2), [LitBank](https://github.com/dbamman/litbank) |
| `gap_coref` (`whr778/gap_coref`) | en | one ambiguous pronoun + two candidate names per Wikipedia passage | 2,000 / 454 / 2,000 passages | 33.3% (by construction) | Apache-2.0 | [GAP](https://huggingface.co/datasets/coref-data/gap_raw) ([paper](https://arxiv.org/abs/1810.05201)) |

**GUM licenses, per genre** (from GUM's LICENSE.md; annotations are CC-BY-4.0 throughout):
academic, court CC-BY-4.0; bio, voyage CC-BY-SA-3.0; news, interview CC-BY-2.5;
**essay, fiction, letter, podcast, whow CC-BY-NC-SA (non-commercial)**; **conversation, speech,
textbook, vlog UNVERIFIED** (not listed in LICENSE.md, 51 of 191 docs). Filter on
`provenance.license` before any commercial distribution of a model trained on them.

**Not used:** PreCo (large, pronouns included, but `license: unknown` on the Hub, not verified);
OntoNotes (LDC-licensed). ReDocRED's `vertexSet` clusters exist but are 8% multi-surface and
carry no pronouns.

## How each was built

- **cmnee, wikievents:** augmented IN PLACE on the deduped disk files (`add_coref_clusters.py`,
  every other byte asserted unchanged); the converters emit the same clusters for a fresh build.
- **gum_coref, litbank_coref:** `convert_corefud.py --corefud <corpus>` (downloads from the
  CorefUD mirror). CorefUD publishes train and dev only, so dev is TEST and a seeded 10% of train
  documents is VAL. Contractions are placed inside their token (`It's` -> `It`).
- **gap_coref:** `convert_gap.py` (downloads `coref-data/gap_raw`); GAP's own splits. The
  coreferent name joins the pronoun's cluster, the distractor is a singleton (a hard negative).
  3 passages repeat inside a split with a different pronoun, by design.

All are in `run_all_converters.sh`, registered in `dataset_registry.yaml` (`hf_jsonl`), and
restorable with `restore_from_hf.py`. Uniqueness: 0 cross-split and 0 cross-corpus overlaps.

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
