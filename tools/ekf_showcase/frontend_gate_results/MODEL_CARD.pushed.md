---
library_name: gliner2
license: other
license_name: unverified-review-required
base_model: jhu-clsp/mmBERT-base
language:
  - en
  - zh
datasets:
  - knowledgator/sentence_rex
  - knowledgator/bio-NER-relations
  - vblagoje/cc_news
  - nlhappy/DuEE
  - knowledgator/events_classification_biotech
  - knowledgator/text2json-training-data
tags:
  - gliner2
  - information-extraction
  - named-entity-recognition
  - relation-extraction
  - event-extraction
  - text-classification
metrics:
  - f1
  - precision
  - recall
pipeline_tag: token-classification
---

# ekf_frontend_mmbert

A [GLiNER2](https://github.com/fastino-ai/GLiNER2) multi-task information-extraction model (entities, relations, events, and classification) fine-tuned from `jhu-clsp/mmBERT-base`.

## ⚠️ License at a glance

- **Effective license:** Unverified — review required
- **Commercial use:** Unverified
- **All dataset licenses verified:** No

See [License](#license) for the full determination and per-dataset terms.

## Model details

- **Base model:** [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base)
- **Library:** `gliner2`
- **Tasks:** entity, relation, event, and classification extraction
- **Experiment:** `ekf_frontend_mmbert`

## Training data

**19** datasets used for this run. 189,284 training records (val: 11,733, test: 21,461).

| Dataset | Task(s) | Train | Val | Test | Language | License | Source |
|---|---|--:|--:|--:|---|---|---|
| sentence_rex | Relation extraction | 34,314 | 4,268 | 4,283 | en | Apache-2.0 | [link](https://huggingface.co/datasets/knowledgator/sentence_rex) |
| bio-NER-relations | NER + relations | 2,084 | 256 | 258 | en | see card | [link](https://huggingface.co/datasets/knowledgator/bio-NER-relations) |
| BioRED | NER + relations (biomedical) | 308 | 47 | 45 | en | NLM / NCBI | [link](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) |
| RAMS | Event extraction (trigger + args) | 7,329 | 439 | 432 | en | see source | [link](https://nlp.jhu.edu/rams/) |
| Multi-event casualty corpus, EVENTS form | event extraction (trigger + typed arguments, multi-instance) | 23,627 | 2,666 | 2,852 | en | see source | — |
| CC-News real text, Haiku-4.5 annotations | multi-task IE (NER, relations, classification, events, structures) | 15,839 | 2,075 | 2,043 | en | see source | [link](https://huggingface.co/datasets/vblagoje/cc_news) |
| ⚠️ `synthetic_coerced` | unknown | — | — | — | — | **UNKNOWN — not in registry** | — |
| ⚠️ `synthetic_sonnet5` | unknown | — | — | — | — | **UNKNOWN — not in registry** | — |
| WikiEvents | NER + event extraction | 200 | — | — | en | see source | [link](https://github.com/raspberryice/gen-arg) |
| CASIE | Event extraction (cybersecurity) | 798 | 95 | 107 | en | see source | [link](https://github.com/Ebiquity/CASIE) |
| DuEE 1.0 | Event extraction (Chinese, trigger + args) | 11,603 | 150 | — | zh | see source | [link](https://www.luge.ai/#/luge/dataDetail?id=6) |
| CMNEE | Event extraction (Chinese military) | 9,281 | 150 | 2,724 | zh | see source | [link](https://github.com/2086482524/CMNEE) |
| MAVEN | Event detection (trigger) | 2,913 | — | — | en | see source | [link](https://github.com/THU-KEG/MAVEN-dataset) |
| Event Detection Dataset (Mendeley) | Event detection (trigger) | 1,420 | 150 | 156 | en | cc-by-4.0 | [link](https://data.mendeley.com/datasets/7d54rvzxkr/1) |
| DocEE | Event extraction (doc-level) | 21,842 | 150 | 2,744 | en | none declared (see source; not redistributable) | [link](https://github.com/tongmeihan1995/docee) |
| ChFinAnn | Event extraction (Chinese financial, doc-level) | 25,632 | 150 | 3,204 | zh | MIT | [link](https://github.com/dolphin-zs/Doc2EDAG) |
| DocFEE | Event extraction (Chinese financial, doc-level) | 16,384 | 150 | 800 | zh | cc-by-4.0 | [link](https://figshare.com/articles/dataset/28632464) |
| events_classification_biotech | Classification (multi-label) | 2,217 | 150 | 263 | en | ODC-BY | [link](https://huggingface.co/datasets/knowledgator/events_classification_biotech) |
| text2json-training-data | Schema-driven structured extraction | 7,976 | 150 | 872 | en | see card | [link](https://huggingface.co/datasets/knowledgator/text2json-training-data) |

**Dataset notes**

- **sentence_rex** — Sentence-level relation extraction over Wikidata-property labels; 818 relation types.
- **bio-NER-relations** — Biomedical NER + relation extraction; 48 entity types, 5 relation types.
- **BioRED** — Biomedical document-level NER + RE from NCBI BioC; 6 entity types, 8 relation types.
- **RAMS** — Multi-sentence event extraction with triggers and typed arguments; 139 event types, 65 argument roles.
- **Multi-event casualty corpus, EVENTS form** — The same documents, splits and figures as casualty_loc_split, re-emitted as trigger + typed arguments instead of anchored json_structures records, for Track B: does the EVENT formulation bind figures better than the structure one? Built by build_multievent_corpus.py --emit events. 23,627 / 2,666 / 2,852 documents, mean ~2.4 instances, 8 event types, 57,726 triggers and 124,561 arguments all verbatim in their own document. TRIGGERS ARE DERIVED, NOT GOLD: DocEE gives a type and never a span, so _locate_trigger matches a per-type surface list inside the snippet's own slice -- 97.6% coverage, worst type Road Crash at 8.7% missing. A gain here is a gain over that trigger definition. The corpus carries 8 event TYPES and no named identities, so it can teach trigger->argument binding and cannot teach same-type discrimination (Helene vs Katrina), which is the live defect in TODO item 2.
- **CC-News real text, Haiku-4.5 annotations** — REAL news documents with model-written labels (19,957 records: 15,839 train / 2,075 val / 2,043 test) -- the real-text half of the real/synthetic mixture, and the counterpart to synthetic_haiku45_5k where the documents themselves were generated. Text is English CC-News (LID-filtered with lumi_language_id; the corpus is ~98.75% en, remainder `und` junk), annotated by claude-haiku-4-5 through synthetic/generate.py --annotate-from. Deduplicated on the document key AT COLLECTION, before annotation was paid for: news syndication republishes the same wire story and a first 10K pull dropped 512 such copies. Verified 0 overlap against every other corpus in data/. Entity spans are checked verbatim against the source (~5% dropped), and absent types are seeded as negatives. CAVEATS: the upstream card declares license `unknown` and the articles remain publisher copyright, so this is a private research cache, not redistributable; domain coverage is skewed (244 domains, but taiwannews.com.tw alone is 18%); and events/structures are sparse on real news (0.26 and 0.08 per document) versus synthetic, where they are guaranteed by construction.
- **WikiEvents** — KAIROS-ontology event extraction co-trained with typed entity mentions; 49 event types, 57 argument roles.
- **CASIE** — Cybersecurity event extraction co-trained with typed entity mentions; 5 event types, 26 argument roles.
- **DuEE 1.0** — Sentence-level Chinese event extraction (65 event types, 121 roles) via the no-login HuggingFace mirror; train + val only, no test.
- **CMNEE** — Chinese military news event extraction; 8 event types, 11 argument roles.
- **MAVEN** — Large general-domain event trigger detection; 168 event types, trigger-only.
- **Event Detection Dataset (Mendeley)** — English ongoing-event trigger detection over NYT economic news; one generic event type, trigger-only.
- **DocEE** — Document-level event extraction; one event per document mapped to role-typed entities and document classification. SPLITS ARE REPAIRED: upstream normal_setting overlaps itself (56 train/val, 12 train/test, 26 val/test documents, plus 84 duplicates inside one split) and the converter honours it 1:1, so dedupe_splits.py runs after conversion with precedence test > val > train. 21,842 / 2,721 / 2,744 after repair, from 21,966 / 2,748 / 2,771.
- **ChFinAnn** — Trigger-free document-level Chinese financial event extraction (5 event types); mapped to role-typed entities and multi-label event-type classification.
- **DocFEE** — Trigger-free document-level Chinese financial event extraction (9 event types); mapped to best-effort role-typed entities and multi-label event-type classification.
- **events_classification_biotech** — Multi-label biotech event-type classification; 29 labels.
- **text2json-training-data** — Schema-driven structured extraction; per-record field schemas define the output JSON structure.

## Training procedure

| Setting | Value |
|---|---|
| Trained on | 2026-08-23 |
| Duration | 17h 17m |
| Throughput | 18.2 samples/s |
| Epochs | 6 |
| Batch size | 4 (× 8 grad-accum) |
| Encoder LR | 2e-05 |
| Task-head LR | 0.0005 |
| Weight decay | 0.01 |
| Scheduler | cosine_restarts (warmup 0.05) |
| Precision | bf16 |
| Max grad norm | 1.0 |
| Best-checkpoint metric | eval_loss |
| Seed | 42 |
| Architecture | `max_len=8192`, `struct_loss=bce_posweight`, `struct_pos_weight=4.0` |

## Evaluation

### Intended use and limitations

**This is a research checkpoint, and it does not yet do the job it was built for.**

It was trained to be the first stage of an EKF disaster-tracking pipeline: read English
news wire copy and emit, per event, a trigger with its arguments bound to it. Two
pass/fail gates were fixed in the training config *before* the run, both on real AP
prose rather than on held-out corpus splits. **Both fail.**

| Gate | Bar | Result |
|---|---|---|
| 1 — trigger + >=1 bound argument on casualty-bearing Hurricane Helene windows | >= 50% | **FAIL** — 25.0% (best over the pre-registered 0.1–0.5 threshold range) |
| 2 — the span block is local: the Katrina block must contain "1,400" and not "Helene" | pass | **FAIL** — no block contains the figure at any threshold |

Below the pre-registered range the form appears but the content does not: at threshold
0.01, 86.7% of windows produce a trigger with some argument, while the Katrina sentence
binds `dead` to `"1"` (a fragment of "1,400") and `event_name` to `"Trump"` and to
`"Pennsylvania"`. Above 0.05 it returns nothing on that sentence at all.

**In-distribution it works.** On its own synthetic casualty corpus it returns the right
trigger and the right bound argument with correct spans. The gap between that and the
wire-copy result is the known failure mode of this training programme, and the cause is
most likely the mix: 72% of the new English trigger-and-argument supervision is
synthetic, against 20% human-annotated real news.

**So: do not deploy this on real news.** It is a useful baseline for the mix question
and a reasonable warm start; it is not an extractor you should trust on journalism.

### Reading the metrics below

- They are measured on **this run's own held-out test splits**, which are not the splits
  any sibling GLiNER2 model in this project was scored on. Cross-model comparisons drawn
  from these numbers alone are not valid.
- They are at this checkpoint's **calibrated threshold 0.3**, not the 0.5 used for the
  project's reference numbers.
- Note the gap between **strict** and **relaxed** argument F1 (0.214 vs 0.615). Strict
  requires the exact span; relaxed accepts overlap. The model proposes approximately
  right arguments far more often than it gets their boundaries exact, so quoting one
  without the other misrepresents it in either direction.

Decision threshold: **0.3** (calibrated against the validation set).

> **Comparing models?** This threshold was calibrated for *this* checkpoint alone. Sibling models often calibrate elsewhere, so numbers quoted at each model's own threshold are read at different operating points and are not directly comparable. Fix one threshold across models before drawing a comparison; `best/threshold_sweep.json` holds the full sweep for exactly this.

### Blind test (held-out test splits)

Micro precision / recall / F1, strict → relaxed.

| Category | Precision | Recall | F1 | Support |
|---|--:|--:|--:|--:|
| entity | 0.621 → 0.691 | 0.549 → 0.610 | 0.583 → 0.648 | 114622 |
| relation | 0.418 → 0.521 | 0.071 → 0.089 | 0.122 → 0.152 | 9949 |
| classification | 0.728 → 0.736 | 0.677 → 0.685 | 0.702 → 0.710 | 16162 |
| structure | 0.689 → 0.729 | 0.079 → 0.084 | 0.142 → 0.151 | 6692 |
| event_type | 1.000 → 1.000 | 0.915 → 0.915 | 0.956 → 0.956 | 12333 |
| event_trigger | 0.746 → 0.753 | 0.811 → 0.818 | 0.777 → 0.784 | 14713 |
| event_argument | 0.232 → 0.651 | 0.199 → 0.582 | 0.214 → 0.615 | 41785 |
| event | 0.499 → 0.739 | 0.458 → 0.695 | 0.478 → 0.716 | 68831 |

## License

**Effective license: Unverified — review required.** This model is a derivative of its base model and every training dataset, so the most restrictive term across all of them governs the whole model.

- **Commercial use:** Unverified
- **Share-alike obligation:** No
- **All licenses verified:** No
- **Base model:** mmBERT-base — see model card

**Unverified — verify the upstream terms before redistribution**
- BioRED (NLM / NCBI)
- CASIE (see source)
- CC-News real text, Haiku-4.5 annotations (see source)
- CMNEE (see source)
- DocEE (none declared (see source; not redistributable))
- DuEE 1.0 (see source)
- MAVEN (see source)
- Multi-event casualty corpus, EVENTS form (see source)
- RAMS (see source)
- WikiEvents (see source)
- bio-NER-relations (see card)
- mmBERT-base (see model card)
- synthetic_coerced (unknown) (unspecified)
- synthetic_sonnet5 (unknown) (unspecified)
- text2json-training-data (see card)

**Permissive**
- ChFinAnn (MIT)
- DocFEE (cc-by-4.0)
- Event Detection Dataset (Mendeley) (cc-by-4.0)
- events_classification_biotech (ODC-BY)
- sentence_rex (Apache-2.0)

> License strings are copied verbatim from each dataset's card/source and from `tools/train/dataset_registry.yaml`. "see card"/"see source"/"other" mean the upstream declares no clear license — treat as unverified. This summary is informational, not legal advice; confirm terms before redistribution or commercial use.

## Citation

If you use this model, please cite GLiNER2 and the underlying datasets (linked in [Training data](#training-data)).

---
_Model card generated automatically at the end of training (2026-08-23)._
