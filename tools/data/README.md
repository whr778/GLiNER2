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
