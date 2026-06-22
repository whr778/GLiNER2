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
| KazNERD | NER (Kazakh, 25 types) | 47,546 | 5,884 | 5,961 | cc-by-4.0 | [HF](https://huggingface.co/datasets/yeshpanovrustem/kaznerd) |
| BC4CHEMD | NER (chemical) | 11,611 | 1,437 | 1,459 | see card | [HF](https://huggingface.co/datasets/chintagunta85/bc4chemd) |
| BC5CDR | NER (chemical + disease) | 3,126 | 394 | 395 | other | [HF](https://huggingface.co/datasets/tner/bc5cdr) |
| stockmark-jpn | NER (Japanese, 8 types) | 3,900 | 486 | 473 | cc-by-sa-3.0 | [HF](https://huggingface.co/datasets/stockmark/ner-wikipedia-dataset) |
| FiNER-ORD | NER (financial, PER/LOC/ORG) | 1,427 | 184 | 171 | **cc-by-nc-4.0** | [HF](https://huggingface.co/datasets/gtfintechlab/finer-ord) |
| KLUE-NER | NER (Korean, 6 types) | 16,782 | 2,116 | 2,104 | cc-by-sa-4.0 | [GitHub](https://github.com/KLUE-benchmark/KLUE) |
| **Relation extraction** | | | | | | |
| sentence_rex | Relation extraction | 34,314 | 4,269 | 4,282 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/sentence_rex) |
| bio-NER-relations | NER + relations | 2,085 | 256 | 258 | see card | [HF](https://huggingface.co/datasets/knowledgator/bio-NER-relations) |
| DocRED | NER + relations (doc-level) | 83,951 | 10,421 | 10,554 | MIT | [HF](https://huggingface.co/datasets/thunlp/docred) |
| KLUE-RE | NER + relations (Korean) | 26,028 | 3,237 | 3,205 | cc-by-sa-4.0 | [GitHub](https://github.com/KLUE-benchmark/KLUE) |
| BioRED | NER + relations (biomedical) | 308 | 47 | 45 | NLM / NCBI | [NCBI](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) |
| SciERC | NER + relations (scientific) | 265 | 46 | 38 | research use (AI2) | [AI2](http://nlp.cs.washington.edu/sciIE/) |
| **Classification** | | | | | | |
| GLiClass v3 logic | Classification (multiple-choice) | 4,566 | 550 | 548 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/gliclass-v3-logic-dataset) |
| GLiClass v2.0-RAC | Classification (multi-label) | 439,354 | 54,718 | 55,293 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/gliclass-v2.0-RAC) |
| Scientific-text-classification | Classification (single-label) | 40,047 | 4,997 | 4,956 | see card | [HF](https://huggingface.co/datasets/knowledgator/Scientific-text-classification) |
| events_classification_biotech | Classification (multi-label) | 2,216 | 271 | 272 | ODC-BY | [HF](https://huggingface.co/datasets/knowledgator/events_classification_biotech) |
| **Structured extraction** | | | | | | |
| text2json-training-data | Schema-driven structured extraction | 7,817 | 962 | 958 | see card | [HF](https://huggingface.co/datasets/knowledgator/text2json-training-data) |
| json_data_extraction | Schema-driven structured extraction | 378 | 55 | 50 | Apache-2.0 | [HF](https://huggingface.co/datasets/paraloq/json_data_extraction) |
| **Event extraction** (manual download) | | | | | | |
| WikiEvents | NER + event extraction | 206 | 20 | 20 | see source | [gen-arg](https://github.com/raspberryice/gen-arg) |
| RAMS | Event extraction (trigger + args) | 7,329 | 924 | 871 | see source | [JHU](https://nlp.jhu.edu/rams/) |
| MAVEN | Event detection (trigger) | 2,913 | — | — | see source | [GitHub](https://github.com/THU-KEG/MAVEN-dataset) |
| CASIE | Event extraction (cybersecurity) | 795 | 98 | 107 | see source | [GitHub](https://github.com/Ebiquity/CASIE) |
| CMNEE | Event extraction (Chinese military) | 9,284 | 1,606 | 2,727 | see source | [GitHub](https://github.com/2086482524/CMNEE) |
| LEVEN | Event detection (Chinese legal, trigger) | 5,301 | 1,230 | — | see source | [GitHub](https://github.com/thunlp/LEVEN) |
| DocEE | Event extraction (doc-level) | 21,966 | 2,748 | 2,771 | see source | [GitHub](https://github.com/tongmeihan1995/docee) |
| ACE 2005 | NER + relations + events | — | — | — | LDC (LDC2006T06) | [LDC](https://catalog.ldc.upenn.edu/LDC2006T06) |
| **Total (generated)** | | **1,721,279** | **214,914** | **215,654** | | |

† Val column includes the `dev` split for WikiEvents and RAMS. MAVEN ships only a
labelled train split (dev/test labels are held out for the leaderboard); LEVEN
ships labelled train/valid but holds out its test labels for the leaderboard.
ACE 2005 is LDC-licensed and not generated here.

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

### KazNERD — `yeshpanovrustem/kaznerd`
Kazakh NER (Wikipedia + news), 25 entity types. BIO token tags folded into
`{type: [surface]}` entities (sentences with no entities are dropped).
*Stats: 25 entity types, avg 1.6 types/record; ~84.7k mentions over 47.5k train sentences.*

### BC4CHEMD — `chintagunta85/bc4chemd`
BioCreative IV chemical NER (PubMed abstracts). Read from the parquet revision;
token/tag lengths are off by one in the source, so the converter aligns on the
common prefix.
*Stats: 1 type (CHEMICAL), avg 1.0/record; ~29k mentions over 11.6k train sentences.*

### BC5CDR — `tner/bc5cdr`
BioCreative V chemical-disease NER (the tner token-tagged version; NER only, no
relations). Bare int tags mapped via the dataset's `dataset/label.json`.
*Stats: 2 types (Chemical, Disease), avg 1.4/record; ~7.0k mentions over 3.1k train sentences.*

### stockmark-jpn — `stockmark/ner-wikipedia-dataset`
Japanese Wikipedia NER, 8 entity types (人名, 法人名, 地名, 製品名, …). Already
span-based; surfaces grouped by type, kept when verbatim in the text.
*Stats: 8 entity types, avg 1.8 types/record; ~10.3k mentions over 3.9k train sentences.*

### FiNER-ORD — `gtfintechlab/finer-ord`
Financial NER (PER/LOC/ORG) over financial news. **cc-by-nc-4.0 (non-commercial)** —
its inclusion in `mmbert-base` makes that mix non-commercial. Token-per-row source
regrouped into sentences by `(doc_idx, sent_idx)`.
*Stats: 3 types (PER, LOC, ORG), avg 1.4/record; ~2.9k mentions over 1.4k train sentences.*

### KLUE-NER — KLUE-benchmark (Korean)
Korean NER read from the canonical [KLUE GitHub](https://github.com/KLUE-benchmark/KLUE)
release (the HF loader is broken). Char-level BIO; text is the concatenated chars,
6 entity types (PS, LC, OG, DT, TI, QT).
*Stats: 6 entity types, avg 1.7 types/record; ~39.8k mentions over 16.8k train sentences.*

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

### KLUE-RE — KLUE-benchmark (Korean)
Korean relation extraction from the canonical [KLUE GitHub](https://github.com/KLUE-benchmark/KLUE)
release. Each record contributes its two typed entities plus a `{label: {head, tail}}`
relation (records labelled `no_relation` keep entities only).
*Stats: 6 entity types, 29 relation types; 70.7% of records carry a relation, ~18.4k relations over 26k train sentences.*

### BioRED — NCBI
Biomedical document-level NER + RE from the [NCBI release](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/)
(BioC.JSON). 6 entity types (Gene, Disease, Chemical, Variant, CellLine, Species);
relations link normalized entity identifiers, so head/tail are representative mentions.
*Stats: 6 entity types, 8 relation types (Association, Positive/Negative_Correlation, Bind, …), avg 3.8 entity types/doc; 99% carry relations, ~2.9k relations over 308 train docs.*

### SciERC — AI2
Scientific NER + RE over abstracts, from the [AI2 release](http://nlp.cs.washington.edu/sciIE/)
(processed JSON). 6 entity types (Task, Method, Material, Metric, OtherScientificTerm,
Generic), 7 relation types (USED-FOR, CONJUNCTION, HYPONYM-OF, …).
*Stats: 6 entity types, 7 relation types, avg 4.4 entity types/doc; 99% carry relations, ~2.4k relations over 265 train docs.*

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

### json_data_extraction — `paraloq/json_data_extraction`
Schema-driven structured extraction: each row is a `(text, JSON Schema, item)`
triple. The converter walks the extracted `item` recursively and maps every leaf
scalar / list-of-scalars to a `{field: [value]}` entity, keeping only values that
appear verbatim in the text. Small but field-diverse, and Apache-2.0 (clean license).
*Stats: ~1,634 distinct leaf field names, ~7,400 extracted values over 483 documents.*

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

### LEVEN — [thunlp/LEVEN](https://github.com/thunlp/LEVEN)
Largest Chinese legal event detection corpus; trigger-only (arguments are empty,
like MAVEN, so only the trigger-detection path of the joint loss benefits; manual
Google Drive download). The held-out test split ships candidates instead of gold
events, so only train / valid are converted.
*Stats: 108 event types, trigger-only (no arguments), ~98k trigger mentions in train.*

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
