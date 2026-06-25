# Dataset Converters

Converters that transform public HuggingFace NER datasets into the GLiNER2
JSONL training format (`{"input": ..., "output": {"entities": {...}, "entity_descriptions": {...}}}`).

Both scripts stream from HuggingFace so they don't need to fit the dataset in RAM.
They install one dep on first run: `uv add datasets`.

## Text normalization & encoding

Every converter emits normalized, UTF-8 JSONL through one write path
(`_split.dumps_record`, used by `SplitWriter` and the standalone writers).
All reads and writes use `encoding="utf-8"`, non-ASCII is written literally
(`ensure_ascii=False`), and each record is recursively normalized so `input`
and every entity/relation/event surface stay consistent (surfaces remain
verbatim substrings of `input`). Per-string normalization (`_split.clean_text`)
does two things:

- NFKC normalization. Note this folds CJK full-width punctuation to ASCII
  (e.g. `，` to `,`); switch `NFKC` to `NFC` in `_split.clean_text` to preserve it.
- Strips stray Unicode line separators (NEL U+0085, U+2028, U+2029) to a space.
  Left in, `json.dumps` writes them literally and they fragment a JSONL record
  across physical lines for any `splitlines()`-based reader.

## Mention-type filtering

Some corpora annotate each entity mention with a *mention type* — ACE 2005 uses
`NAM` (named), `NOM` (nominal), `PRO` (pronominal). A converter that supports it
takes `--filter-config <yaml>` to keep only the types you want. **Filtering an
entity mention cascades**: any relation or event argument that referenced a
dropped mention is dropped too. Non-entity event arguments (e.g. time/value
fillers, which have no mention type) are always kept.

Config (`tools/data/config/mention_filter.yaml`):

```yaml
allow: [NAM, NOM, PRO]    # global default; omit to keep all, `[]` drops all typed
converters:               # optional per-converter overrides
  ace2005:
    allow: [NAM, NOM]     # e.g. drop pronouns for ACE 2005
```

```bash
uv run python tools/data/convert_ace2005.py \
    --input /path/to/ace_2005_td_v7/data/English \
    --out data/ace2005.jsonl \
    --filter-config tools/data/config/mention_filter.yaml
```

The converter prints how many mentions, relations, and event arguments the
filter dropped. Default (no `--filter-config`) keeps every type — byte-identical
to before. The shared plumbing is `tools/data/_mention_filter.py`; a new
converter opts in by reading its per-mention type and calling
`load_mention_filter(path, "<name>").allows(m_type)`.

> ACE 2005 support assumes raw-LDC APF: the mention type is the `TYPE` attribute
> on `<entity_mention>`, and `event_mention_argument` `REFID`s are mention-level
> (like `relation_mention_argument`). Verify against your corpus if filtering
> doesn't behave as expected.

## numind/NuNER

```bash
# Full split (with LLM-generated entity descriptions, ~1M rows)
uv run python tools/data/convert_nuner.py --split full --out data/nuner_full.jsonl

# Smaller subset for a smoke run
uv run python tools/data/convert_nuner.py --split full --out data/nuner_10k.jsonl --max-records 10000
```

Each output line carries every entity type seen for that source text, with
descriptions when they were present in the source. Spans that don't appear
verbatim in the text (the source data is LLM-generated and ~3-8% don't) are
silently dropped; the final summary line reports the count.

## Universal-NER/Pile-NER-definition

```bash
uv run python tools/data/convert_pile_ner_definition.py --out data/pile_ner.jsonl
```

Pile-NER-definition uses long natural-language definitions as entity "types".
The converter mints short synthetic keys (`e_0`, `e_1`, ...) per record and
puts the definition in `entity_descriptions`, so the model sees compact type
tokens with rich descriptions. Empty answers (negative samples) are dropped.

## knowledgator/GLINER-multi-task-synthetic-data

```bash
uv run python tools/data/convert_knowledgator_gliner.py \
    --out data/knowledgator_gliner.jsonl

# Smaller subset for a smoke run
uv run python tools/data/convert_knowledgator_gliner.py \
    --out data/knowledgator_5k.jsonl --max-records 5000
```

The source dataset stores each row as a word-tokenized text with a
"Identify the following entity classes ... Text: ..." prompt prefix and a
list of `[start_token, end_token, "\"Label\""]` spans. The converter strips
the prompt prefix (so the model trains on plain text, not the template),
re-bases each span's token indices into the body, joins body tokens into a
plain string, and unwraps the JSON quoting on labels. Each record carries
~10 entity types on average, so loss values start much higher than NuNER
records but drop fast.

## knowledgator/gliclass-v3-logic-dataset

```bash
uv run python tools/data/convert_gliclass_logic.py \
    --out data/gliclass_logic.jsonl

# Smaller subset for a smoke run
uv run python tools/data/convert_gliclass_logic.py \
    --out data/gliclass_logic_2k.jsonl --max-records 2000

# Customise the task name written into each classification record
uv run python tools/data/convert_gliclass_logic.py \
    --out data/gliclass_logic.jsonl --task-name reasoning
```

**Same converter handles `knowledgator/gliclass-v2.0-RAC`**, a sibling dataset in the GLiClass family with the identical `{text, all_labels, true_labels}` schema. RAC is general-domain (~612k rows, multi-label news topic classification with per-document custom hypothesis sets). Override `--repo` and use a different `--task-name` so the two corpora's classification tasks stay namespaced apart at train time:

```bash
uv run python tools/data/convert_gliclass_logic.py \
    --repo knowledgator/gliclass-v2.0-RAC \
    --task-name topic_classification \
    --out data/gliclass_rac.jsonl
```

This is a **classification** corpus — it trains GLiNER2's classification head, not the entity extractor. Each source row carries `text`, `true_labels`, and `all_labels`; the converter emits

```json
{"input": "<text>",
 "output": {"classifications": [
     {"task": "logic", "labels": <all_labels>, "true_label": <true_labels>}
 ]}}
```

`Classification.__post_init__` auto-sets `multi_label=True` when more than one true label is provided.

About ~27% of source rows are dropped because their `true_labels` aren't a subset of `all_labels` (NLI-style entries where the true label is a relation type like `"neutral"` and `all_labels` is a single candidate hypothesis). Those would fail GLiNER2's classification validation, so we filter them at conversion time rather than letting the trainer reject them.

Mixing this corpus with the NER corpora teaches the model classification on top of extraction; the trainer happily interleaves both record types.

## knowledgator/sentence_rex

```bash
uv run python tools/data/convert_sentence_rex.py \
    --out data/sentence_rex.jsonl

# Drop relation labels with fewer than 5 examples
uv run python tools/data/convert_sentence_rex.py \
    --out data/sentence_rex.jsonl --min-count 5
```

Sentence-level **relation extraction**, ~44k rows. Each source row marks the two relation arguments inline with `<e1>...</e1>` and `<e2>...</e2>` tags, with a single Wikidata-property label (`director`, `cast member`, `architect`, etc.). The converter strips the tags, recovers clean text, and emits one relation per record using GLiNER2's `{relation_name: {head, tail}}` shape (head = e1, tail = e2).

Vocabulary is large (~850 labels), with 92 singleton labels. `--min-count` lets you drop the tail; default 1 keeps everything.

## knowledgator/bio-NER-relations

```bash
uv run python tools/data/convert_bio_ner_relations.py \
    --out data/bio_ner_relations.jsonl

# Keep the noisy umlsterm entity bucket
uv run python tools/data/convert_bio_ner_relations.py \
    --out data/bio_ner_relations.jsonl --skip-types ''
```

Document-level biomedical **NER + relation extraction**, ~10k rows. Source rows follow a BioC-style layout with `passages`, `entities`, and `relations`. Entities have character offsets; relations reference entity IDs (`arg1_id`, `arg2_id`).

The converter joins all passage text, groups entities by type into `output.entities`, resolves each relation's args via the entity-ID lookup, and emits relations as `output.relations` with head = arg1 surface and tail = arg2 surface.

`umlsterm` is dropped by default (it accounts for ~85% of entity assignments — auto-matched UMLS concepts that are noisy for downstream training). Override with `--skip-types ''` to keep them, or `--skip-types umlsterm,Habitat` to drop multiple types.

## thunlp/docred

```bash
uv run python tools/data/convert_docred.py --out data/docred.jsonl

# Cap the distant-supervised volume (e.g. for a cleaner/smaller corpus)
uv run python tools/data/convert_docred.py --out data/docred.jsonl --max-records 20000

# Gold dev split only (998 human-annotated docs)
uv run python tools/data/convert_docred.py --out data/docred.jsonl --split validation
```

Document-level **NER + relation extraction** (DocRED). Each row has `sents` (per-sentence token lists), `vertexSet` (one coreference cluster of mentions per entity, each with `type` and `pos`), and `labels` (parallel `head`/`tail` entity indices + `relation_text`). The converter joins all tokens into one document, reconstructs each mention surface from its tokens at `pos` (so surfaces appear verbatim), groups them by the 6 entity types (`PER`, `ORG`, `LOC`, `TIME`, `NUM`, `MISC`), and emits relations under their human-readable `relation_text` (head/tail = each entity's first mention).

The original ships a dataset script that newer `datasets` won't run, so this reads the auto-converted parquet revision (`refs/convert/parquet`). Its `train` split **merges the ~3k gold annotated docs with the ~102k distant-supervised (noisy) docs** — there is no way to isolate the gold train docs. Use `--split validation` for the 998 gold dev docs, or `--max-records` to cap the distant volume. `--split test` has no relation labels.

## knowledgator/events_classification_biotech

```bash
uv run python tools/data/convert_events_biotech.py \
    --out data/events_biotech.jsonl

# Convert the test split for held-out evaluation
uv run python tools/data/convert_events_biotech.py \
    --out data/events_biotech_test.jsonl --file test.csv
```

**Despite the name, this is multi-label text classification, not structured event extraction.** Each article is tagged with 1–5 of 29 event-type categories (`funding round`, `m&a`, `alliance & partnership`, `executive statement`, etc.). There are no trigger spans or argument roles — those would require new model heads and a different dataset.

The repo ships CSVs (`train.csv`, 2,759 rows; `test.csv`) plus a legacy `events_classification_biotech.py` loader script that newer `datasets` rejects, so the converter downloads the CSV directly via `huggingface_hub` and parses it with pandas.

Input text is `Title + "\n" + Content` joined. All 29 labels are repeated as the `labels` vocabulary per record; `true_label` is the non-empty subset from columns `Label 1`...`Label 5`. `Classification.__post_init__` auto-detects multi-label.

## DocEE (Tong et al., NAACL 2022)

```bash
# Canonical normal_setting splits (recommended — matches the paper):
uv run python tools/data/convert_docee.py --no-stratify \
    --input data/docee/DocEE-en/normal_setting/train.json --out data/docee.train.jsonl
uv run python tools/data/convert_docee.py --no-stratify \
    --input data/docee/DocEE-en/normal_setting/dev.json   --out data/docee.val.jsonl
uv run python tools/data/convert_docee.py --no-stratify \
    --input data/docee/DocEE-en/normal_setting/test.json  --out data/docee.test.jsonl

# Or auto-stratify from the all-data file:
uv run python tools/data/convert_docee.py \
    --input data/docee/DocEE-en/DocEE-en.json \
    --out data/docee.jsonl
```

**Largest publicly-available document-level event extraction corpus** — 27,485 docs, 59 event types, 356 argument-role types, 180,528 argument instances. One event per document. **No triggers** are annotated, so the converter maps each doc into entities + classification by default: arguments are bucketed by their role `type` (`Husband`, `Court`, `Date`, …) into `output.entities`, and the doc-level `event_type` becomes a single classification record with the 59-type vocabulary. Pass `--emit-events` to additionally emit events records with a synthetic trigger (`[event_type]` prepended to the text); off by default.

**Manual download required**: the data ships through a Google Drive folder linked from https://github.com/tongmeihan1995/docee — no public direct URL. The easiest way is `gdown`:

```bash
mkdir -p data/docee
uv run --with gdown gdown --folder \
    'https://drive.google.com/drive/folders/1_cRnc2leAmOKT9Ma8koz6X8Ivl-_lapp' \
    -O data/docee/
```

That drops everything under `data/docee/DocEE-en/` (the all-data file at the top, plus `normal_setting/` and `cross_domain_setting/` subfolders). Each DocEE record is a 4-element list `[title, text, event_type, annotations]` where the annotation list elements are `{start, end, type, text}` dicts; the converter handles that format directly.

## CMNEE (Zhu et al., LREC-COLING 2024)

```bash
# Download once (Google Drive folder):
mkdir -p data/cmnee
uv run --with gdown gdown --folder \
    'https://drive.google.com/drive/folders/1nfKiSsu88oBeykUSYm7NGn4Q50_2GPS1' \
    -O data/cmnee/

# Then convert the canonical train/valid/test splits:
uv run python tools/data/convert_cmnee.py \
    --input data/cmnee/CMNEE/train.json --out data/cmnee.train.jsonl
uv run python tools/data/convert_cmnee.py \
    --input data/cmnee/CMNEE/valid.json --out data/cmnee.val.jsonl
uv run python tools/data/convert_cmnee.py \
    --input data/cmnee/CMNEE/test.json  --out data/cmnee.test.jsonl
```

**Chinese Military News Event Extraction**: 17,000 documents, **29,223 events**, 8 event types, 11 argument role types. Document-level, multi-event-per-document, with manual character-offset annotations for both triggers and typed arguments. Source records are clean dicts (`{id, text, event_list, coref_arguments}`) — `event_list[*]` maps directly into our `{event_type, trigger, arguments}` shape, no schema gymnastics.

This is the first Chinese corpus in the training mix; it complements `gliner_multilingual` (multilingual NER) with multilingual event supervision and adds the military domain alongside CASIE (cyber) and the general-domain English corpora. mmBERT handles Chinese natively. The `coref_arguments` field is ignored (we don't model coreference). Records with empty `event_list` (the dataset's negative samples) are dropped — the converter only emits docs that contribute event-extraction supervision.

## CASIE (Satyapanich et al., AAAI 2020)

```bash
# Auto-downloads the GitHub tarball (~24 MB), produces a stratified
# 80/10/10 train/test/val split by default:
uv run python tools/data/convert_casie.py --out data/casie.jsonl
```

Cybersecurity event corpus: 1,000 news articles annotated with 5 event subtypes (`Databreach`, `Phishing`, `Ransom`, `Vulnerability-Discover`, `Vulnerability-Patch`) and ~21 typed argument-entity types (`PII`, `Person`, `Organization`, `Device`, `Money`, `System`, etc.). Each argument carries both an event role (`Compromised-Data`, `Attacker`, `Place`, ...) **and** an entity type, so the corpus naturally feeds the entities + events co-training pattern (no relations annotated).

Stratification uses the same greedy multi-label rule (b) as `convert_ace2005.py` (shared in `tools/data/_stratify.py`). Categories combine entity types and event subtypes; on the full corpus all 26 categories appear in every split. Default prefixes event subtypes as `Cyber.<Subtype>` so they don't collide with ACE / MAVEN / RAMS / WikiEvents event types; pass `--no-prefix-event` to keep them bare. Pass `--no-stratify` for single-file output.

## WikiEvents (Li et al., NAACL 2021)

```bash
# All three canonical splits — auto-fetched from the public S3 bucket:
uv run python tools/data/convert_wikievents.py --split train --out data/wikievents.train.jsonl
uv run python tools/data/convert_wikievents.py --split dev   --out data/wikievents.dev.jsonl
uv run python tools/data/convert_wikievents.py --split test  --out data/wikievents.test.jsonl
```

Document-level event extraction (KAIROS ontology, ~50 event types and ~60 argument roles) plus typed entity mentions (PER, GPE, ORG, LOC, FAC, WEA) in the same record — perfect for **NER + event co-training**. 206 train / 20 dev / 20 test documents. The dataset's `relation_mentions` field is always empty across all splits, so the converter emits entities + events only (no relations).

Source lives in a public S3 bucket published by https://github.com/raspberryice/gen-arg, so `--split` auto-downloads via HTTPS. `--input` accepts either a URL or a local path for offline use.

## knowledgator/biomed_NER

```bash
uv run python tools/data/convert_biomed_ner.py \
    --out data/biomed_ner.jsonl
```

Biomedical NER with 35 entity classes (CHEMICALS, ACTIVITY, PHENOTYPE, FUNCTION, GROUP, DISORDER, GENE AND GENE PRODUCTS, ANATOMICAL STRUCTURE, etc.). 4,840 abstracts, averaging ~46 spans each — dense annotation.

Source rows are `{text, entities: [{start, end, class}]}` with end-exclusive character offsets, so surfaces are sliced directly from `text`. Light cleanup:

- Class names with trailing whitespace are normalised (the source has `"ORGANISMS "` and `"ORGANISMS"` as separate buckets).
- The `"Unlabelled"` class (~190 spans, no training signal) is skipped.

This is the only domain-specific (biomedical) corpus in the recipe. Mixing it with the general-domain corpora keeps the model strong on plain text while adding biomedical extraction headroom.

## knowledgator/PubMedAbstractsNER

```bash
uv run python tools/data/convert_pubmed_abstracts_ner.py \
    --out data/pubmed_abstracts_ner.jsonl
```

35k PubMed abstracts with token-level NER spans typed by UMLS-style biomedical concept categories (~470 distinct types — `Body Regions`, `Hemic and Immune Systems`, `Radiography`, `Probability`, …). The source label string bakes the type and a natural-language description together with `" - "`; this converter splits them so the type goes into the `entities` bucket and the description goes into `entity_descriptions` for the model to condition on.

The HuggingFace `datasets` loader fails on this repo (`KeyError: 'feature'`), so the converter downloads `train.json` directly via `huggingface_hub`. Token offsets are inclusive on both ends.

## knowledgator/Scientific-text-classification

```bash
uv run python tools/data/convert_scientific_text.py \
    --out data/scientific_text.jsonl

# Keep more of the long tail
uv run python tools/data/convert_scientific_text.py \
    --out data/scientific_text.jsonl --min-count 10
```

Single-label scientific-abstract classification with a **global** label vocabulary. The raw dataset is heavily skewed: 10 broad-domain labels (mathematics, quantum physics, astrophysics, computer science, statistics, etc.) each have ~5,000 examples and 28,631 fine-grained MeSH-style labels each have exactly **one** example. Records with singleton labels can't train the classification head and would inflate every record's `labels` field to ~28k entries.

The converter therefore drops labels by `--min-count` (default 2 — which naturally yields the 10 broad-domain labels covering ~50k rows, since there's a hard cliff in the distribution between rank-10 and rank-11). Raise `--min-count` to keep a broader tail, but expect rapid quality drop past the natural cliff.

The emitted records share the same `labels` vocabulary and carry `true_label` as a 1-element list.

## knowledgator/gliner-multilingual-synthetic

```bash
uv run python tools/data/convert_gliner_multilingual.py \
    --out data/gliner_multilingual.jsonl

# Smaller subset for a smoke run
uv run python tools/data/convert_gliner_multilingual.py \
    --out data/gliner_multilingual_5k.jsonl --max-records 5000
```

Same on-disk shape as the GLINER-multi-task corpus (`tokenized_text` + `[start, end, "\"label\""]` triples) but with no prompt prefix — each row is just the raw multilingual passage (German, Polish, French, etc.) and its spans. The converter joins tokens, slices each span, unwraps the JSON-quoted label, and verbatim-filters surfaces.

Pair this with `mmBERT-base` (multilingual encoder) to learn non-English extraction.

## knowledgator/text2json-training-data

```bash
uv run python tools/data/convert_text2json.py \
    --out data/text2json.jsonl

# Smaller subset for a smoke run
uv run python tools/data/convert_text2json.py \
    --out data/text2json_5k.jsonl --max-records 5000

# Pick a different JSONL file in the same repo
uv run python tools/data/convert_text2json.py \
    --file mixed_train.jsonl --out data/text2json_mixed.jsonl
```

The repo holds ~25 JSONL files with inconsistent per-shard schemas (some carry an `_augmented` column, others don't), so the converter bypasses `datasets.load_dataset` — which fails the Arrow schema cast — and downloads a single named file via `huggingface_hub` instead. Default is `augmented_train.jsonl` (12.8k rows, clean `{text, extracted}` schema). Pass `--file` to convert a different one (e.g. `sft_train.jsonl`, `mixed_train.jsonl`, `merged_clean.jsonl`).

The `extracted` payload comes in two shapes:

- **Entity list** — `{"entities": [{"entity": "...", "type": "...", "description": "..."}, ...]}`. Mapped directly to GLiNER2 NER with descriptions preserved.
- **Flat key→value** — e.g. `{"tournament_code": "ROL-2024", "winner": "Sofia Petrova", ...}`. Each top-level key becomes an entity label; the value (coerced to a string) becomes the surface.

Nested dicts and list-of-dicts values are skipped — their leaves typically don't round-trip verbatim into the source text. Surfaces that don't appear verbatim are dropped silently; ~20% of rows are dropped entirely because they're all paraphrased or generated values.

Because text2json field names are highly bespoke (each document defines its own schema), the type cardinality grows fast. Mixing this corpus with the others helps the model generalise to arbitrary, schema-driven extraction prompts at inference time.

## Output format

Both scripts produce GLiNER2 JSONL that can be passed directly to
`GLiNER2Trainer.train(train_data=...)`:

```python
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

model = GLiNER2.from_pretrained("jhu-clsp/mmBERT-base")
trainer = GLiNER2Trainer(model, TrainingConfig(output_dir="./out"))
trainer.train(train_data="data/nuner_full.jsonl")
```
