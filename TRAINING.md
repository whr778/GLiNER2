# Running GLiNER2 Training from Scratch on mmBERT

End-to-end instructions for training a custom GLiNER2 model on top of `jhu-clsp/mmBERT-small` or `jhu-clsp/mmBERT-base`, using the full **NuNER** and **Pile-NER-definition** corpora as pre-training data. Works on NVIDIA GPUs and Apple M-series (MPS); the trainer auto-selects the best available device.

---

## 1. Install

```bash
git clone <this-repo> GLiNER2 && cd GLiNER2
uv sync                       # installs torch>=2.1, transformers>=4.48, gliner, etc.
uv add datasets               # one-time, only needed for the dataset converters
```

The first training run downloads the mmBERT weights (~550 MB for small, ~1.2 GB for base) into the HuggingFace cache.

---

## 2. Convert the datasets to GLiNER2 JSONL

```bash
mkdir -p data

# NuNER `full` split — 1,000,000 rows with LLM-generated entity descriptions
uv run python tools/data/convert_nuner.py \
    --split full \
    --out data/nuner_full.jsonl

# Pile-NER-definition — ~47,671 conversations, each contributing many entity-type queries
uv run python tools/data/convert_pile_ner_definition.py \
    --out data/pile_ner_def.jsonl

# knowledgator/GLINER-multi-task-synthetic-data — multi-task synthetic NER
# (~10 entity types per record on average; prompt prefix is stripped automatically)
uv run python tools/data/convert_knowledgator_gliner.py \
    --out data/knowledgator_gliner.jsonl

# knowledgator/text2json-training-data — schema-driven structured extraction
# (each record defines its own field names; nested objects are skipped)
uv run python tools/data/convert_text2json.py \
    --out data/text2json.jsonl

# knowledgator/gliner-multilingual-synthetic — multilingual NER
# (German, Polish, French, etc.; pair with mmBERT for non-English extraction)
uv run python tools/data/convert_gliner_multilingual.py \
    --out data/gliner_multilingual.jsonl

# knowledgator/gliclass-v3-logic-dataset — classification (NOT NER)
# Trains GLiNER2's classification head; the trainer interleaves NER and
# classification records cleanly.
uv run python tools/data/convert_gliclass_logic.py \
    --out data/gliclass_logic.jsonl

# knowledgator/Scientific-text-classification — single-label classification
# of scientific abstracts (10 broad domains: math, quantum physics, ...)
uv run python tools/data/convert_scientific_text.py \
    --out data/scientific_text.jsonl

# knowledgator/biomed_NER — domain-specific biomedical NER
# (35 classes: CHEMICALS, DISORDER, GENE AND GENE PRODUCTS, ...)
uv run python tools/data/convert_biomed_ner.py \
    --out data/biomed_ner.jsonl

# knowledgator/events_classification_biotech — multi-label classification
# (29 biotech "event types"; despite the name, NO structured event extraction)
uv run python tools/data/convert_events_biotech.py \
    --out data/events_biotech.jsonl

# knowledgator/sentence_rex — sentence-level relation extraction
# (~850 Wikidata-property labels; <e1>/<e2> markup stripped)
uv run python tools/data/convert_sentence_rex.py \
    --out data/sentence_rex.jsonl

# knowledgator/bio-NER-relations — doc-level biomedical NER + RE
# (50 entity types, 48 relation types; umlsterm noise dropped by default)
uv run python tools/data/convert_bio_ner_relations.py \
    --out data/bio_ner_relations.jsonl

```

All converters stream from HuggingFace (no need to hold the dataset in RAM). Each prints a final summary line: records emitted, records dropped (for NER converters: because no span appeared verbatim; for the classification converters: because the labels couldn't form a valid classification task), and the count of distinct entity types or label counts.

Approximate output sizes after conversion:

| Source | Task | Records out | Disk |
|---|---|---:|---:|
| NuNER `full` | NER | ~990,000 | ~0.8 GB |
| Pile-NER-definition | NER | ~45,000 | ~0.2 GB |
| knowledgator/GLINER-multi-task-synthetic-data | NER | ~210,000 | ~0.4 GB |
| knowledgator/text2json-training-data | NER (extraction) | ~80,000 | ~0.2 GB |
| knowledgator/gliner-multilingual-synthetic | NER (multilingual) | ~400,000 | ~0.3 GB |
| knowledgator/gliclass-v3-logic-dataset | Classification (multiple-choice) | ~5,700 | ~10 MB |
| knowledgator/Scientific-text-classification | Classification (single-label) | ~50,000 | ~80 MB |
| knowledgator/biomed_NER | NER (biomedical, 35 classes) | ~4,800 | ~30 MB |
| knowledgator/events_classification_biotech | Classification (multi-label) | ~2,750 | ~10 MB |
| knowledgator/sentence_rex | Relation extraction | ~44,000 | ~30 MB |
| knowledgator/bio-NER-relations | Biomedical NER + RE | ~10,400 | ~80 MB |

You can pass any subset of the JSONL files to the trainer at once — they're concatenated and shuffled. Mixing all eleven is a good recipe: NuNER contributes scale and descriptions, Pile-NER contributes long natural-language type definitions, GLINER-multi-task contributes dense multi-type schemas, text2json contributes bespoke per-document field names, gliner-multilingual contributes non-English passages (essential when training on top of `mmBERT` — without it the multilingual encoder weights drift toward English-only extraction), gliclass-logic teaches multiple-choice classification with arbitrary candidate sets, Scientific-text-classification teaches single-label classification with a fixed vocabulary, biomed_NER adds domain-specific biomedical extraction, events_biotech adds multi-label business-news classification, sentence_rex introduces general-domain relation extraction, and bio-NER-relations couples biomedical NER with co-occurring relations.

```python
trainer.train(train_data=[
    "data/nuner_full.jsonl",
    "data/pile_ner_def.jsonl",
    "data/knowledgator_gliner.jsonl",
    "data/text2json.jsonl",
    "data/gliner_multilingual.jsonl",
    "data/gliclass_logic.jsonl",
    "data/scientific_text.jsonl",
    "data/biomed_ner.jsonl",
    "data/events_biotech.jsonl",
    "data/sentence_rex.jsonl",
    "data/bio_ner_relations.jsonl",
])
```

---

## 3. Pick a hardware profile

The committed training scripts use **`max_len=8192`** — mmBERT's full sequence window. ModernBERT's local-global attention keeps memory near-linear in sequence length, but activations and the optimizer state still scale with it, so batch sizes at 8192 are much smaller than at 512. The cells below assume `max_len=8192`. To regain bigger batches at the cost of context window, lower `max_len` in `TrainingConfig` (e.g. `max_len=2048` or `1024`) and roughly 2–4× each batch-size cell.

| Device | mmBERT-small (148 M params) | mmBERT-base (314 M params) |
|---|---|---|
| **Single A100/H100 80 GB** | `batch_size=16–24`, `bf16=True` | `batch_size=8–12`, `bf16=True` |
| **Single A100/4090 40–48 GB** | `batch_size=8–12`, `bf16=True` | `batch_size=4` + `gradient_accumulation_steps=2`, `bf16=True` |
| **Single 24 GB GPU (3090/4090)** | `batch_size=4–6`, `bf16=True` | `batch_size=2` + `gradient_accumulation_steps=4`, `bf16=True` |
| **Apple M-series (MPS, 32–96 GB unified)** | `batch_size=1–2`, no AMP, dev / small-slice only | `batch_size=1`, no AMP, dev / small-slice only |
| **CPU** | dev / smoke only | dev / smoke only |

Per-step wall-clock at `max_len=8192` is roughly 6–10× longer than at `max_len=512` for the same effective batch size; full-corpus runs on smaller GPUs become multi-day. Tune to your real corpus before committing to a long run.

> On MPS, mixed precision (`fp16`/`bf16`) is disabled automatically: `torch.amp.GradScaler` is CUDA-only and MPS autocast adds little speed-up for this model. The trainer logs the choice it made.
>
> For an Apple M-series dev workflow at `max_len=8192`, an epoch over the full mixed corpus is prohibitive — drop `max_len` to 1024–2048 and/or pass `--max-records 50_000` to the converters to train on a slice. Use a real GPU for the full 8192 corpus.

---

## 4. Train

> **Run the training scripts under `tools/train/` rather than piping a heredoc.** Piping the script via `python - <<PY ... PY` breaks DataLoader workers on macOS + Python 3.12+, where multiprocessing uses the **spawn** start method: workers need to re-import the main module by path, but stdin has no path. Symptom: `FileNotFoundError: '<stdin>'` followed by `BrokenPipeError`.

### 4a. mmBERT-small

Recommended baseline — runs on a single 24 GB GPU and converges in 2–3 epochs of mixed NuNER + Pile-NER-definition.

```bash
uv run python tools/train/train_mmbert_small.py
```

The script (`tools/train/train_mmbert_small.py`) calls `GLiNER2.from_encoder("jhu-clsp/mmBERT-small", ...)` — use `from_encoder` for training from scratch; `from_pretrained` is for loading saved GLiNER2 checkpoints. Edit the script in place to change hyperparameters (batch size, learning rates, epochs, `max_len`, etc.). Defaults target a 24 GB GPU; see the hardware table above for other profiles.

Checkpoints land in `out/mmbert-small/`. The `final` checkpoint is the last step; intermediate `checkpoint-<step>` directories are rotated.

### 4b. mmBERT-base

Same shape; lower learning rates, smaller batch, longer wall-clock:

```bash
uv run python tools/train/train_mmbert_base.py
```

Edit `tools/train/train_mmbert_base.py` to retune for your hardware. Defaults use `bf16=True` (prefer on Ampere+/Hopper; switch to `fp16=True` elsewhere) and `max_len=512`.

### Optional: hold out an evaluation slice

```python
from gliner2.training.data import TrainingDataset
ds = TrainingDataset.load(["data/nuner_full.jsonl", "data/pile_ner_def.jsonl"])
train_ds, val_ds, _ = ds.split(train_ratio=0.99, val_ratio=0.01, test_ratio=0.0, seed=42)
# then pass train_ds / val_ds to trainer.train(..., eval_data=val_ds) and set
# eval_strategy="steps", eval_steps=2000 in the config.
```

---

## 5. Use the trained model

```python
from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("./out/mmbert-small/final")   # or .../best
print(model.extract_entities(
    "Marie Curie discovered radium in Paris.",
    ["scientist", "element", "city"],
))
```

`GLiNER2.from_pretrained` auto-selects CUDA → MPS → CPU. Pass `map_location="cpu"` to force CPU, or `quantize=True` for fp16 inference on GPU/MPS.

---

## 6. Push to Hugging Face Hub

Once you're happy with a checkpoint, upload it so it can be loaded by anyone with `GLiNER2.from_pretrained("<username>/<repo>")`.

First, authenticate (once per machine):

```bash
uv run huggingface-cli login        # paste a write token from https://huggingface.co/settings/tokens
# or: export HF_TOKEN=hf_xxx        # non-interactive (CI, headless boxes)
```

Then push:

```bash
uv run python tools/train/push_to_hub.py \
    --checkpoint ./out/mmbert-small/final \
    --repo-id <username>/gliner2-mmbert-small \
    --private
```

Flags:

- `--checkpoint` — local path to a saved GLiNER2 checkpoint directory (`final`, `best`, or any `checkpoint-<step>`).
- `--repo-id` — target repo, created if missing.
- `--private` (default) / `--public` — repo visibility.
- `--commit-message` — optional commit message (defaults to `Upload GLiNER2 checkpoint`).

The script calls `model.save_pretrained()` into a temp directory and uploads the folder via `HfApi.upload_folder`, so the repo layout matches what `GLiNER2.from_pretrained` expects (`config.json` + `encoder_config/config.json` + `model.safetensors` + tokenizer files).

Sanity-check the upload:

```python
from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("<username>/gliner2-mmbert-small")
print(model.extract_entities("Marie Curie discovered radium in Paris.",
                             ["scientist", "element", "city"]))
```

---

## 7. Practical tips

- **Run a 50-record smoke test first** before launching a multi-day run:
  ```bash
  uv run python tools/data/convert_nuner.py --split full --out /tmp/nuner_smoke.jsonl --max-records 50
  # then train with num_epochs=1, batch_size=2, max_steps=20 to confirm the loss falls
  ```
- **Mix the two datasets in one pass.** Passing them as a list to `train_data=` interleaves them; the converter scripts already produce compatible JSONL.
- **Resume mid-run** by pointing a fresh `from_pretrained` at the latest checkpoint directory (`./out/.../checkpoint-<step>`) and starting a new trainer. The trainer always starts a new optimizer/scheduler — that's intentional, not a bug.
- **Watch the loss curve.** Healthy mmBERT-small on this data starts at ~500 (batch 1), drops below 100 in the first ~50 steps, then drifts down toward 40–60 over an epoch. Loss collapsing to ~0 means the data labels aren't reaching the loss head — stop and inspect.
- **W&B**: set `report_to_wandb=True` and `wandb_project="..."` in `TrainingConfig` to stream live metrics.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: datasets` | Converters need the HF `datasets` lib | `uv add datasets` |
| `FileNotFoundError: '<stdin>'` + `BrokenPipeError` | Running training inside a heredoc (`python - <<PY`) with `num_workers>0`; spawn workers can't re-import stdin | Save the script as a `.py` file and run `uv run python <file>.py`, or set `num_workers=0` |
| `out of memory` on CUDA | Batch too large | Halve `batch_size`, double `gradient_accumulation_steps` |
| `out of memory` on MPS | Same | Same; also try `max_len=384` and `num_workers=0` |
| Loss stuck at exactly the initial value | LR too low or all params frozen | Confirm `use_lora=False` (or pass LoRA modules); raise `task_lr` |
| Loss explodes (`NaN`/`Inf`) | LR too high or mixed precision on a fragile encoder | Drop `encoder_lr` 5x, set `fp16=False` |
| `train_metrics_history` is empty | `logging_steps` larger than total steps | Lower `logging_steps`, or run more epochs |
| Tokenizer error about unknown special tokens | mmBERT tokenizer didn't accept GLiNER2's specials | Should never happen with `transformers>=4.48` — file an issue and include the tokenizer version |
| `401 Unauthorized` from `push_to_hub.py` | No HF auth token | `uv run huggingface-cli login` (with a **write**-scope token), or `export HF_TOKEN=hf_xxx` |
| `403 Forbidden` writing to `<repo-id>` | Token lacks write access or repo is owned by someone else | Issue a new write token, or pick a repo-id you own |
