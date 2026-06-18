# Training Data

The corpora used to train GLiNER2, converted to the unified JSONL format by the
scripts under [`tools/data/`](tools/data/) (see [TRAINING.md](TRAINING.md) §2 for
the conversion commands). Sample counts below are the **actual line counts of the
generated `data/*.jsonl` splits**; type statistics are computed from the same
files. Most corpora are partitioned 80/10/10 by a seeded `SplitWriter`; the event
corpora keep their canonical train/dev/test splits.

## Summary

| Dataset | Task(s) | Train | Val† | Test | License‡ | Source |
|---|---|--:|--:|--:|---|---|
| **NER** | | | | | | |
| NuNER | NER | 790,202 | 98,373 | 98,464 | MIT | [HF](https://huggingface.co/datasets/numind/NuNER) |
| Pile-NER-definition | NER (typed + definitions) | 38,048 | 4,740 | 4,715 | see card | [HF](https://huggingface.co/datasets/Universal-NER/Pile-NER-definition) |
| GLiNER multi-task synthetic | NER (multi-task) | 10,319 | 1,276 | 1,288 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/GLINER-multi-task-synthetic-data) |
| GLiNER multilingual synthetic | NER (multilingual) | 77,259 | 9,598 | 9,749 | see card | [HF](https://huggingface.co/datasets/knowledgator/gliner-multilingual-synthetic) |
| biomed_NER | NER (biomedical) | 3,885 | 485 | 470 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/biomed_NER) |
| PubMedAbstractsNER | NER (biomedical + descriptions) | 28,051 | 3,486 | 3,450 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/PubMedAbstractsNER) |
| **Relation extraction** | | | | | | |
| sentence_rex | Relation extraction | 34,314 | 4,269 | 4,282 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/sentence_rex) |
| bio-NER-relations | NER + relations | 2,085 | 256 | 258 | see card | [HF](https://huggingface.co/datasets/knowledgator/bio-NER-relations) |
| DocRED | NER + relations (doc-level) | 83,951 | 10,421 | 10,554 | MIT | [HF](https://huggingface.co/datasets/thunlp/docred) |
| **Classification** | | | | | | |
| GLiClass v3 logic | Classification (multiple-choice) | 4,566 | 550 | 548 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/gliclass-v3-logic-dataset) |
| GLiClass v2.0-RAC | Classification (multi-label) | 439,354 | 54,718 | 55,293 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/gliclass-v2.0-RAC) |
| Scientific-text-classification | Classification (single-label) | 40,047 | 4,997 | 4,956 | see card | [HF](https://huggingface.co/datasets/knowledgator/Scientific-text-classification) |
| events_classification_biotech | Classification (multi-label) | 2,216 | 271 | 272 | ODC-BY | [HF](https://huggingface.co/datasets/knowledgator/events_classification_biotech) |
| **Structured extraction** | | | | | | |
| text2json-training-data | Schema-driven structured extraction | 7,817 | 962 | 958 | see card | [HF](https://huggingface.co/datasets/knowledgator/text2json-training-data) |
| **Event extraction** (manual download) | | | | | | |
| WikiEvents | NER + event extraction | 206 | 20 | 20 | see source | [gen-arg](https://github.com/raspberryice/gen-arg) |
| RAMS | Event extraction (trigger + args) | 7,329 | 924 | 871 | see source | [JHU](https://nlp.jhu.edu/rams/) |
| MAVEN | Event detection (trigger) | 2,913 | — | — | see source | [GitHub](https://github.com/THU-KEG/MAVEN-dataset) |
| CASIE | Event extraction (cybersecurity) | 795 | 98 | 107 | see source | [GitHub](https://github.com/Ebiquity/CASIE) |
| CMNEE | Event extraction (Chinese military) | 9,284 | 1,606 | 2,727 | see source | [GitHub](https://github.com/2086482524/CMNEE) |
| DocEE | Event extraction (doc-level) | 21,966 | 2,748 | 2,771 | see source | [GitHub](https://github.com/tongmeihan1995/docee) |
| ACE 2005 | NER + relations + events | — | — | — | LDC (LDC2006T06) | [LDC](https://catalog.ldc.upenn.edu/LDC2006T06) |
| **Total (generated)** | | **1,604,607** | **199,798** | **201,753** | | |

† Val column includes the `dev` split for WikiEvents and RAMS. MAVEN ships only a
labelled train split (dev/test labels are held out for the leaderboard). ACE 2005
is LDC-licensed and not generated here.

‡ "see card" = the HuggingFace dataset card declares no explicit license — verify
before redistribution. "see source" = manual-download corpora governed by their
original release terms.

---

## NER

### NuNER — `numind/NuNER`
Large-scale synthetic NER (~987k records) with LLM-generated entity types and
descriptions. Converter drops spans that don't appear verbatim in the text.
*Stats (120k-record sample): ~59,993 distinct entity types, avg 3.1 types/record, 100% of records carry entities.*

### Pile-NER-definition — `Universal-NER/Pile-NER-definition`
NER where each "type" is a long natural-language definition. The converter mints
synthetic per-record keys (`e_0`, `e_1`, …) and stores the definition in
`entity_descriptions`, so the model conditions on compact keys with rich text.
*Stats: up to 40 synthetic keys/record (avg 10.2), 100% carry entities; the type "vocabulary" is per-record, not a fixed schema.*

### GLiNER multi-task synthetic — `knowledgator/GLINER-multi-task-synthetic-data`
Dense multi-type synthetic NER (~10 types/record on average), open vocabulary.
*Stats: ~40,336 distinct entity types, avg 5.8 types/record, 100% carry entities.*

### GLiNER multilingual synthetic — `knowledgator/gliner-multilingual-synthetic`
Multilingual NER (German, French, Polish, …) — essential when training on
`mmBERT` so the multilingual encoder doesn't drift toward English-only.
*Stats: ~18,767 distinct entity types, avg 2.5 types/record, 100% carry entities.*

### biomed_NER — `knowledgator/biomed_NER`
Domain-specific biomedical NER with a fixed 33-class schema (CHEMICALS, DISORDER,
GENE AND GENE PRODUCTS, …).
*Stats: 33 entity types, avg 6.9 types/record, 100% carry entities.*

### PubMedAbstractsNER — `knowledgator/PubMedAbstractsNER`
~35k PubMed abstracts with ~470 UMLS-style biomedical types; descriptions are
parsed out of the label string into `entity_descriptions`.
*Stats: ~5,698 distinct entity types (open UMLS-style), avg 10.3 types/record, 100% carry entities.*

## Relation extraction

### sentence_rex — `knowledgator/sentence_rex`
Sentence-level relation extraction over Wikidata-property labels (`<e1>`/`<e2>`
markup stripped).
*Stats: 818 distinct relation types, 100% of records carry relations.*

### bio-NER-relations — `knowledgator/bio-NER-relations`
Document-level biomedical NER + relation extraction (noisy `umlsterm` entities
dropped by default).
*Stats: 48 entity types, 5 relation types, avg 2.0 entity types/record, 80.3% of records carry relations.*

### DocRED — `thunlp/docred`
Document-level NER + relation extraction. Relations use human-readable names; the
parquet `train` split merges ~3k gold-annotated docs with ~102k distant-supervised
(noisy) docs (see [TRAINING.md](TRAINING.md) for the caveat).
*Stats: 6 entity types, 96 relation types, avg 4.6 entity types/record, 99.7% of records carry relations.*

## Classification

### GLiClass v3 logic — `knowledgator/gliclass-v3-logic-dataset`
Multiple-choice classification with arbitrary per-record candidate label sets.
*Stats: 1 task, ~66,972 distinct candidate labels (open per-record candidate sets), 100% classification.*

### GLiClass v2.0-RAC — `knowledgator/gliclass-v2.0-RAC`
General-domain multi-label classification (largest classification corpus). Reuses
the v3-logic converter with `--repo` / `--task-name` overrides.
*Stats (120k-record sample): 1 task, ~214,567 distinct candidate labels (open candidate sets), 100% classification.*

### Scientific-text-classification — `knowledgator/Scientific-text-classification`
Single-label classification of scientific abstracts over 10 broad domains.
*Stats: 1 task, 10 labels (fixed vocabulary), 100% classification.*

### events_classification_biotech — `knowledgator/events_classification_biotech`
Multi-label biotech "event-type" classification (despite the name, no structured
event extraction).
*Stats: 1 task, 29 labels, 100% classification.*

## Structured extraction

### text2json-training-data — `knowledgator/text2json-training-data`
Schema-driven structured extraction; each record defines its own field names
(nested objects skipped).
*Stats: ~9,060 distinct field names (per-record schemas), avg 5.3 fields/record.*

## Event extraction

These corpora require manual or scripted download (see [TRAINING.md](TRAINING.md)
§2) and keep their canonical splits.

### WikiEvents — [gen-arg](https://github.com/raspberryice/gen-arg)
KAIROS-ontology event extraction co-trained with typed entity mentions;
auto-downloads from a public S3 bucket.
*Stats: 17 entity types, 49 event types, 57 argument roles, 94.2% of records carry events.*

### RAMS — [nlp.jhu.edu/rams](https://nlp.jhu.edu/rams/)
Multi-sentence event extraction with triggers + typed arguments.
*Stats: 139 event types, 65 argument roles, 100% of records carry events.*

### MAVEN — [THU-KEG/MAVEN-dataset](https://github.com/THU-KEG/MAVEN-dataset)
Large general-domain trigger detection (trigger-only — arguments are empty, so
only the trigger-detection path of the joint loss benefits).
*Stats: 168 event types, trigger-only (no arguments).*

### CASIE — [Ebiquity/CASIE](https://github.com/Ebiquity/CASIE)
Cybersecurity event extraction co-trained with typed entity mentions;
auto-downloads the GitHub tarball and emits a stratified 80/10/10 split.
*Stats: 21 entity types, 5 event types, 26 argument roles, 100% of records carry events.*

### CMNEE — [CMNEE](https://github.com/2086482524/CMNEE)
Chinese military news event extraction with triggers + typed arguments (manual
Google Drive download).
*Stats: 8 event types, 11 argument roles, 100% of records carry events.*

### DocEE — [tongmeihan1995/docee](https://github.com/tongmeihan1995/docee)
Largest doc-level event corpus; one event per doc, no triggers — mapped to
role-typed entities + 59-way document classification (manual Google Drive download).
*Stats: 356 role-entity types, 59 document classes, avg 4.7 types/record, 100% carry both entity and classification annotations.*

### ACE 2005 — [LDC2006T06](https://catalog.ldc.upenn.edu/LDC2006T06)
LDC-licensed; not redistributable and not generated here. Convert from your own
licensed copy via `tools/data/convert_ace2005.py` (emits a stratified 80/10/10
split covering entity, relation, and event types).

---

*Statistics were computed by scanning the generated `data/*.jsonl` train splits;
counts for NuNER and GLiClass v2.0-RAC are over a 120k-record sample (their full
train splits exceed that). Regenerate everything with `tools/data/run_all_converters.sh`.*
