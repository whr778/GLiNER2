# Dataset Converters

Converters that transform public HuggingFace NER datasets into the GLiNER2
JSONL training format (`{"input": ..., "output": {"entities": {...}, "entity_descriptions": {...}}}`).

Both scripts stream from HuggingFace so they don't need to fit the dataset in RAM.
They install one dep on first run: `uv add datasets`.

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
