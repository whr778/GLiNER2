# Training Data

The corpora used to train GLiNER2, converted to the unified JSONL format by the
scripts under [`tools/data/`](.) (see [TRAINING.md](../train/TRAINING.md) §2 for
the conversion commands). Sample counts below are the **actual line counts of the
generated `data/*.jsonl` splits**; type statistics are computed from the same
files. Most corpora are partitioned 80/10/10 by a seeded `SplitWriter` (or a
greedy stratified splitter for CASIE and ACE 2005); corpora that ship official
train/dev/test splits — including WikiEvents, RAMS, CMNEE, DocEE, DuEE, Re-DocRED,
the MTL-Bioinformatics-2016 corpora, and the MasakhaNER 2.0 / MasakhaNEWS
benchmarks — keep their canonical splits (noted per corpus below).

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
| MasakhaNER 2.0 | NER (20 African langs, 4 types) | 67,865 | 9,951 | 20,511 | afl-3.0 | [HF](https://huggingface.co/datasets/masakhane/masakhaner2) |
| WikiANN (PAN-X) | NER (176 langs) | _streamed_§ | _streamed_§ | _streamed_§ | see card | [HF](https://huggingface.co/datasets/unimelb-nlp/wikiann) |
| **NER (biomedical — MTL-Bioinformatics-2016)** | | | | | | |
| AnatEM | NER (anatomy) | 3,514 | 1,122 | 2,308 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| BC2GM | NER (genes) | 6,428 | 1,292 | 2,570 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| BioNLP09 | NER (proteins) | 4,711 | 1,014 | 1,700 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| BioNLP11EPI | NER (proteins) | 3,797 | 1,241 | 2,836 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| BioNLP11ID | NER (4 types) | 1,850 | 586 | 1,389 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| BioNLP13CG | NER (16 types) | 2,936 | 964 | 1,829 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| BioNLP13GE | NER (proteins) | 1,504 | 1,663 | 1,941 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| BioNLP13PC | NER (4 types) | 2,365 | 812 | 1,575 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| CRAFT | NER (6 ontologies) | 8,344 | 2,756 | 5,694 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| Ex-PTM | NER (proteins) | 857 | 279 | 1,160 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| JNLPBA | NER (5 types) | 15,150 | 1,514 | 3,202 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| NCBI-disease | NER (disease) | 2,923 | 489 | 539 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| linnaeus | NER (species) | 1,556 | 524 | 1,034 | cc-by-4.0 | [GitHub](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) |
| **Relation extraction** | | | | | | |
| sentence_rex | Relation extraction | 34,314 | 4,269 | 4,282 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/sentence_rex) |
| bio-NER-relations | NER + relations | 2,085 | 256 | 258 | see card | [HF](https://huggingface.co/datasets/knowledgator/bio-NER-relations) |
| DocRED | NER + relations (doc-level) | 83,951 | 10,421 | 10,554 | MIT | [HF](https://huggingface.co/datasets/thunlp/docred) |
| Re-DocRED | NER + relations (doc-level) | 3,053 | 500 | 500 | see card | [HF](https://huggingface.co/datasets/tonytan48/Re-DocRED) |
| KLUE-RE | NER + relations (Korean) | 26,028 | 3,237 | 3,205 | cc-by-sa-4.0 | [GitHub](https://github.com/KLUE-benchmark/KLUE) |
| BioRED | NER + relations (biomedical) | 308 | 47 | 45 | NLM / NCBI | [NCBI](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) |
| SciERC | NER + relations (scientific) | 265 | 46 | 38 | research use (AI2) | [AI2](http://nlp.cs.washington.edu/sciIE/) |
| ProfessorBob relation_extraction | Relation extraction (passage-level) | 13,926 | 1,737 | 1,728 | see card | [HF](https://huggingface.co/datasets/ProfessorBob/relation_extraction) |
| **Classification** | | | | | | |
| GLiClass v3 logic | Classification (multiple-choice) | 4,566 | 550 | 548 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/gliclass-v3-logic-dataset) |
| GLiClass v2.0-RAC | Classification (multi-label) | 439,354 | 54,718 | 55,293 | Apache-2.0 | [HF](https://huggingface.co/datasets/knowledgator/gliclass-v2.0-RAC) |
| Scientific-text-classification | Classification (single-label) | 40,047 | 4,997 | 4,956 | see card | [HF](https://huggingface.co/datasets/knowledgator/Scientific-text-classification) |
| events_classification_biotech | Classification (multi-label) | 2,216 | 271 | 272 | ODC-BY | [HF](https://huggingface.co/datasets/knowledgator/events_classification_biotech) |
| MasakhaNEWS | Classification (16 African langs, 7 topics) | 21,734 | 3,112 | 6,242 | afl-3.0 | [HF](https://huggingface.co/datasets/masakhane/masakhanews) |
| **Structured extraction** | | | | | | |
| text2json-training-data | Schema-driven structured extraction | 7,817 | 962 | 958 | see card | [HF](https://huggingface.co/datasets/knowledgator/text2json-training-data) |
| json_data_extraction | Schema-driven structured extraction | 378 | 55 | 50 | Apache-2.0 | [HF](https://huggingface.co/datasets/paraloq/json_data_extraction) |
| **Event extraction** (manual download) | | | | | | |
| WikiEvents | NER + event extraction | 206 | 20 | 20 | see source | [gen-arg](https://github.com/raspberryice/gen-arg) |
| RAMS | Event extraction (trigger + args) | 7,329 | 924 | 871 | see source | [JHU](https://nlp.jhu.edu/rams/) |
| MAVEN | Event detection (trigger) | 2,913 | — | — | see source | [GitHub](https://github.com/THU-KEG/MAVEN-dataset) |
| CASIE | Event extraction (cybersecurity) | 795 | 98 | 107 | see source | [GitHub](https://github.com/Ebiquity/CASIE) |
| CMNEE | Event extraction (Chinese military) | 9,284 | 1,606 | 2,727 | see source | [GitHub](https://github.com/2086482524/CMNEE) |
| DocEE | Event extraction (doc-level) | 21,966 | 2,748 | 2,771 | see source | [GitHub](https://github.com/tongmeihan1995/docee) |
| ChFinAnn | Event extraction (Chinese financial) | 25,632 | 3,204 | 3,204 | see source | [Doc2EDAG](https://github.com/dolphin-zs/Doc2EDAG) |
| DocFEE | Event extraction (Chinese financial) | 16,420 | 1,824 | 800 | cc-by-4.0 | [GitHub](https://github.com/tongzhou21/DocFEE) |
| DuEE 1.0 | Event extraction (Chinese) | 11,603 | 1,453 | — | see source | [LUGE](https://www.luge.ai/) |
| Mendeley-ED | Event detection (English, trigger-only) | 1,431 | 159 | 156 | cc-by-4.0 | [Mendeley](https://doi.org/10.17632/7d54rvzxkr.1) |
| ACE 2005 | NER + relations + events | — | — | — | LDC (LDC2006T06) | [LDC](https://catalog.ldc.upenn.edu/LDC2006T06) |
| **EKF disaster tracking** (synthetic) | | | | | | |
| casualty_ft | Structured extraction (single-event) | 29,198 | 1,303 | 1,038 | generated here | [`disaster_streams/`](../../datasets/disaster_streams) |
| casualty_multi | Structured extraction (multi-event) | 29,030 | 1,297 | 1,023 | generated here | [`disaster_streams/`](../../datasets/disaster_streams) |
| **Total (generated)** | | **1,933,577** | **249,880** | **276,572** | | |

† Val column includes the `dev` split for WikiEvents and RAMS. MAVEN ships only a
labelled train split (dev/test labels are held out for the leaderboard).
ACE 2005 is LDC-licensed and not generated here.

‡ "see card" = the HuggingFace dataset card declares no explicit license — verify
before redistribution. "see source" = manual-download corpora governed by their
original release terms.

§ WikiANN is **streamed at train time**, not written to `data/`, so it has no
generated split counts and is excluded from the generated total. See
[README.md](README.md) → *unimelb-nlp/wikiann* and the `data.hf_streaming` config
block.

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

### MasakhaNER 2.0 — `masakhane/masakhaner2`
Human-annotated NER over **20 African languages** (Adelani et al., EMNLP 2022),
four entity types mapped to `person`/`organization`/`location`/`date`. BIO token
tags fold into `{type: [surface]}` entities via the shared `bio_to_entities` helper
(sentences with no entities are dropped). Loaded from the datasets-server parquet
export and kept on the **official** per-language train/validation/test splits;
`--langs` selects a subset (default all 20). Per-language corpora are also written
to `data/masakhaner_<lang>.*` so a config can train on one language or a subset.
*Stats: 4 entity types, avg 1.6 types/record; ~152.4k mentions over 67.9k train sentences (20 languages combined).*

### WikiANN (PAN-X) — `unimelb-nlp/wikiann` (streamed)
Token-BIO NER over **176 languages** (Pan et al. 2017; Rahimi et al. 2019 balanced
splits), three entity types (PER/ORG/LOC → person/organization/location).
**Streamed lazily from HF at train time — never written to `data/`** — via
`tools/data/hf_stream.py` and a config `data.hf_streaming` block, so the train set
is never fully resident. Select a subset of languages or `all`; val/test are
bounded in-memory samples capped by label class (`eval_min_per_class`). See
[README.md](README.md) → *unimelb-nlp/wikiann*. Configs:
`tools/train/config/{gliner2-multi-v1,mmbert-base}-wikiann.yaml`.
*Stats: 3 entity types; silver-standard (Wikipedia-derived).*

### MTL-Bioinformatics-2016 (13 biomedical NER corpora) — [cambridgeltl/MTL-Bioinformatics-2016](https://github.com/cambridgeltl/MTL-Bioinformatics-2016)
The Crichton et al. (2017) multi-task benchmark: CoNLL BIO biomedical NER corpora,
converted by `convert_mtl_bio.py` on their **canonical** train/devel/test splits
(the `-IOB` folders; JNLPBA from its plain folder). Each corpus keeps its own entity
types — single-type (BC2GM: GENE; NCBI-disease: Disease; linnaeus: Species; AnatEM:
Anatomy; the BioNLP protein sets) through rich multi-type (BioNLP13CG: 16 types;
CRAFT: 6 ontologies CHEBI/CL/GGP/GO/SO/Taxon; JNLPBA: 5). BC4CHEMD and BC5CDR are
covered separately above; the POS corpus (GENIA-pos) and per-entity-type subset
folders are excluded. All-`O` sentences are dropped.
*Stats: 13 corpora, ~55.9k train sentences, ~243k entity mentions; CC BY 4.0.*

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
(noisy) docs (see [TRAINING.md](../train/TRAINING.md) for the caveat).
*Stats: 6 entity types, 96 relation types, avg 4.6 entity types/record, 99.7% of records carry relations.*

### Re-DocRED — `tonytan48/Re-DocRED`
Revised DocRED with corrected gold annotations (Tan et al., 2022). Unlike DocRED's
train split (which mixes ~3k gold + ~102k noisy distant-supervised docs), Re-DocRED
ships only clean gold annotations in canonical train/validation/test splits. Relation
text labels are mapped from Wikidata P-IDs using the same strings as DocRED's
`relation_text` field, so both corpora share the same 96 relation-type vocabulary.
*Stats: 6 entity types, 96 relation types, avg 4.5 entity types/record, 99.9% of records carry relations.*

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

### ProfessorBob relation_extraction — `ProfessorBob/relation_extraction`
Passage-level relation extraction over Wikidata-style properties. The source ships
one `[subject, relation, object]` triple per row; `convert_professorbob_re.py` groups
triples by passage into multi-relation records (`{label: {head, tail}}`), keeping only
triples whose surfaces appear verbatim (~47% of source triples name canonical entities
absent from the passage text and are dropped). No license is declared on the card.
*Stats: 198 relation types, ~39k relations over 17.4k passages, 100% carry relations.*

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

### MasakhaNEWS — `masakhane/masakhanews`
News-topic classification over **16 African languages** (Adelani et al.,
IJCNLP-AACL 2023). Each article's `category` becomes the true label under a single
`news topic` task; the candidate `labels` set is the union of categories over the
selected languages (7 topics: business, entertainment, health, politics, religion,
sports, technology). Input is `headline + text` by default (`--text-field` to change).
Kept on the official per-language splits; `--langs` selects a subset (default all 16),
with per-language corpora in `data/masakhanews_<lang>.*`.
*Stats: 1 task, 7 labels (fixed vocabulary), single-label, 100% classification (21.7k train articles, 16 languages combined).*

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

These corpora require manual or scripted download (see [TRAINING.md](../train/TRAINING.md)
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

### ChFinAnn — [Doc2EDAG](https://github.com/dolphin-zs/Doc2EDAG)
Large Chinese financial document-level event extraction (Zheng et al., EMNLP 2019);
trigger-free (an event is an event-type label plus a role→value table). Auto-downloads
the release zip.
*Stats: 5 event types, 24 role types; 25.6k train docs.*

### DocFEE — [tongzhou21/DocFEE](https://github.com/tongzhou21/DocFEE)
Chinese financial document-level event extraction (Chen et al., Scientific Data 2025);
trigger-free and offset-free. Auto-downloads the GitHub zip.
*Stats: 9 event types; 16.4k train docs. CC BY 4.0.*

### DuEE 1.0 — [Baidu LUGE](https://www.luge.ai/)
Sentence-level Chinese event extraction (Li et al., NLPCC 2020) with triggers + typed
argument roles. Public test labels are held out (no test split).
*Stats: 65 event types, 121 argument roles; 11.6k train sentences.*

### Mendeley Event Detection — [Mendeley 7d54rvzxkr](https://doi.org/10.17632/7d54rvzxkr.1)
English ongoing-event **trigger detection** over NYT economic/crisis news (Maisonnave
et al., 2020); word-level triggers only (no argument roles). Auto-downloads the tarball.
*Stats: trigger-only; 1.4k train sentences. CC BY 4.0.*

### ACE 2005 — [LDC2006T06](https://catalog.ldc.upenn.edu/LDC2006T06)
LDC-licensed; not redistributable and not generated here. Convert from your own
licensed copy via `tools/data/convert_ace2005.py` (emits a stratified 80/10/10
split covering entity, relation, and event types).

## EKF disaster tracking

Training data for the casualty structure model that feeds the EKF tracker. Unlike every
other corpus here it is **generated, not downloaded** — and unlike them it is built to be
*consumed by a filter*, so its ground truth is a time series rather than a label set.

The generation chain, and why each stage exists:

```
generate.py          parametric streams, seeded      free, exact ground truth
      |              dead/injured -> asymptote, missing decays, hedged noisy reports
      v
realize.py           Sonnet-5 news snippets          COSTS MONEY (--provider mock is free)
      |              one MULTI-FACT snippet per report time
      v
build_finetune_corpus.py  /  build_multievent_corpus.py    -> data/casualty_{ft,multi}.*
```

**Why synthetic at all.** The tracker needs `(t, role, value, qualifier, source)` tuples
with known truth over time. No public corpus carries a *trajectory* — real reporting gives
you one snapshot per article and no answer key for the state between them. Generating it
parametrically makes ground truth exact and free; the only paid step is turning structured
observations into realistic prose.

**Why the text is realized by a model rather than templated.** Templated text is trivially
parseable and hides the actual failure. Each report time becomes **one snippet carrying
several roles plus distractor numbers**, so extraction has to *bind* each figure to the right
role amid dates, magnitudes and competing figures. The conditioning tuple stays known truth —
the snippet states the exact digits with the hedge — so a correct extractor recovers it.

### casualty_ft — single-event
`data/casualty_ft.{train,val,test}.jsonl` · 29,198 / 1,303 / 1,038

One `casualty_report` per document. **Verified: 1.00 records/doc.** That is the corpus's
defining limitation — the count head only ever saw "1", so on a document describing several
incidents the model must blend competing figures into one forced instance. Measured
consequence on multi-event text: value binding collapses from **1.000 → 0.369**, with
**22.6%** of readings bound to the *wrong* event's number.

### casualty_multi — multi-event
`data/casualty_multi.{train,val,test}.jsonl` · 29,030 / 1,297 / 1,023

Several incidents per document, one record each. **Verified: mean 2.35 records/doc**,
distribution `{1: 1366, 2: 1425, 3: 1292, 4: 917}` over a 5,000-record sample. Single-event
documents are deliberately kept in the mix so 1.000 single-event binding cannot silently
regress while multi-event improves. Document lengths run median 132 / max 258 words, so **0%
exceed the 384-word training window** — no multi-event document is ever split across windows,
which would sever a record.

Build it from **`train` streams only**. The showcase feeds are drawn from `test`, and that
separation is the only thing keeping the evaluation uncontaminated.

### Real events — evaluation only, never training
`datasets/helene2024/`, `datasets/turkey2023/`, `datasets/venezuela_2026/`

Held-out validation, each with a `rollup.json` declaring the scope hierarchy. Helene pairs a
**Wikipedia per-state casualty table** (ground truth) with **AP prose** (feed) — deliberately
different sources, which is what makes `est_last_value` a genuine baseline. In the earlier
Türkiye–Syria run truth was read from the same sentence the extractor reads, so that baseline
scored 0.000 by construction and the filter was unmeasurable.

Sibling `datasets/disaster_streams_*` directories are alternative generations of the same
pipeline (`_docee` / `_docee250` real DocEE contexts, `_hard`, `_scaled`, `_sonnet5` realized
text, `_model` / `_model_ft` model-extracted arms), kept so ablations remain reproducible.

Conversion and training commands: [TRAINING.md](../train/TRAINING.md) §6.

---

*Statistics were computed by scanning the generated `data/*.jsonl` train splits;
counts for NuNER and GLiClass v2.0-RAC are over a 120k-record sample (their full
train splits exceed that). Regenerate everything with `tools/data/run_all_converters.sh`.*
