# Core Changes: `mmbert_training` vs `main`

This document covers only files that already exist on `main` and were modified
on `mmbert_training` — i.e. the shipped `gliner2` package, `pyproject.toml`,
`README.md`, `.gitignore`, and a handful of touched tests/tutorials. It does
**not** cover directories that are net-new to this branch (`tools/train/`,
`tools/data/`, `tools/train/config/`, `METRICS.md`, new test files, new
tutorial files) — those are training tooling built *on top of* the core
changes described here, not core changes themselves.

Five files inside `gliner2/training/` are new (`parallel.py`, `eta.py`,
`chunking.py`, `metrics.py`, `stopwords.py`). They aren't modifications to
existing code, but the modified `trainer.py` and `training/__init__.py`
import from them directly, so they're listed briefly for context — full
detail is out of scope for the same reason `tools/train` is.

Full diff: `git diff main...mmbert_training`.

## Summary

| File | Change (+/-) | Category |
|---|---|---|
| `gliner2/inference/schema.py` | +127/-0 | New feature: events |
| `gliner2/inference/schema_model.py` | +43/-2 | New feature: events |
| `gliner2/processor.py` | +136/-4 | New feature: events + CJK tokenization |
| `gliner2/inference/engine.py` | +185/-8 | New feature: events |
| `gliner2/api_client.py` | +133/-5 | New feature: events |
| `gliner2/training/data.py` | +185/-9 | New feature: events |
| `gliner2/model.py` | +288/-52 | Loss variants, encoder-agnostic loading, DataParallel safety |
| `gliner2/training/trainer.py` | +495/-52 | Training infra: MPS/DDP/DataParallel, checkpoint-restart, sliding window, gradient checkpointing |
| `gliner2/training/__init__.py` | +31/-0 (new file) | Public exports for the above |
| `pyproject.toml` | +23/-1 | Python floor raised, new hard dependencies |
| `README.md`, `.gitignore`, 3 tests, 4 tutorials | minor | Docs / test hardening |

---

## 1. New feature: event extraction

Adds a fourth extraction task type (alongside entities/structures/relations):
ACE-style event triggers with typed, multi-valued arguments. This is
implemented consistently across every layer of the stack — it is a
general-purpose feature, not something specific to mmBERT training.

- **`gliner2/inference/schema.py`** — `Schema.events(event_types, trigger_threshold, argument_threshold)`
  builder method. Accepts `{event_type: [role, ...]}`, the richer
  `{event_type: {"roles": [...], "description": ..., "role_descriptions": {...}}}`,
  or a list-of-dicts form. Normalizes into `self.schema["events"]` plus
  `_event_metadata` / `_event_order` / `_event_role_descriptions`, wired into
  `Schema.build()`, `Schema.from_dict()`, and `Schema.from_pydantic()`.
- **`gliner2/inference/schema_model.py`** — `SchemaInput.events` pydantic field
  and `validate_events` validator (non-empty roles, no duplicates, no blank
  strings). The "at least one section" model validator now includes `events`.
- **`gliner2/processor.py`** —
  - New `V_TOKEN = "[V]"` special token (added to `SPECIAL_TOKENS` and the
    special-ID cache).
  - New `SamplingConfig` fields `remove_events_prob` / `shuffle_event_roles`.
  - New `SchemaTransformer._process_events()`, called from
    `_encode_schemas_and_labels` alongside `_process_relations` /
    `_process_classifications`. Events are modeled as multi-field structures:
    each event type gets a schema entry `[trigger, role_1, role_2, ...]`.
    Handles two input shapes — the inference dict form (no gold spans, just
    produces the schema embedding) and the training list-of-mentions form
    (groups mentions by type, unions roles across occurrences, dedups rows).
  - `WhitespaceTokenSplitter`'s regex now tokenizes CJK characters
    individually (each Han/Hiragana/Katakana/Hangul codepoint is its own
    token) instead of falling through to the generic `\w+` word pattern,
    which previously merged CJK runs into single unsplittable tokens. This
    is unrelated to events but ships in the same file/branch.
- **`gliner2/inference/engine.py`** — `event_metadata` / `event_role_orders` /
  `event_order` threaded through the schema-metadata dict (both the `Schema`
  object path and the raw-dict path). New `_extract_events()`: for each
  predicted instance, takes the top-1 span above `trigger_threshold` for the
  trigger field and all spans above `argument_threshold` for each role field
  (roles are multi-valued). `format_results()` now detects events (via the
  `requested_events` list, or by shape — a list of dicts each with `trigger`
  and `arguments`) and buckets them under `event_extraction`, mirroring how
  relations are bucketed under `relation_extraction`. New public
  `extract_events()` / `batch_extract_events()` convenience methods.
  - Also in this file: `pin_memory=True if torch.cuda.is_available() else False`
    was simplified to `pin_memory=torch.cuda.is_available()` — a no-op
    cosmetic change, not a behavior change (the trainer's `pin_memory`
    handling, covered below, is where the actual MPS-related fix is).
- **`gliner2/api_client.py`** — `SchemaAPI.events()` (mirrors `Schema.events()`
  for the cloud-request payload) and `GLiNER2API.extract_events()` /
  `batch_extract_events()`. `has_any_task` validation now includes `"events"`.
- **`gliner2/training/data.py`** — New `EventArgument` (`role`, `entity`) and
  `Event` (`event_type`, `trigger`, `arguments`) dataclasses, structurally
  parallel to `Relation`. Both validate that surface text (trigger, each
  argument entity) appears verbatim (case-insensitive) in the parent
  example's text. `InputExample` gained an `events` field wired through
  `validate()`, `sanitize()` (drops events with missing/not-found triggers,
  drops individual arguments with missing/not-found entities, dedups
  `(role, entity)` pairs), `to_dict()`, and `from_dict()`.

## 2. Model changes (`gliner2/model.py`)

Two independent groups of changes:

**Encoder-agnostic loading (generic robustness, not mmBERT-specific):**
- `from_pretrained`'s embedding-resize-mismatch handling previously hardcoded
  the parameter name `encoder.embeddings.word_embeddings.weight` (BERT/DeBERTa
  naming). It now resolves the input-embedding parameter dynamically via
  `model.encoder.get_input_embeddings()` + a `named_parameters()` scan, so it
  works for any encoder regardless of internal naming (e.g. ModernBERT/mmBERT
  use `embeddings.tok_embeddings`).
- `tie_weights()` is now called after `resize_token_embeddings()` in `__init__`
  and again after `load_state_dict()` in `from_pretrained`, for encoders that
  tie input/output embeddings (a no-op for BERT/DeBERTa, required for
  ModernBERT/mmBERT).
- `from_pretrained` now auto-selects device (CUDA → MPS → CPU) when
  `map_location` isn't given, instead of only moving the model when
  `map_location` was explicitly passed (previously: no `map_location` meant
  the model silently stayed wherever `PreTrainedModel.__init__` put it).
- New `Extractor.from_encoder()` classmethod: bootstraps a fresh model
  (pretrained encoder weights, randomly-initialized task heads) from a raw
  HF encoder id/path, for training a GLiNER2 model from scratch on a backbone
  that was never a GLiNER2 checkpoint. `from_pretrained` remains for loading
  an existing GLiNER2 checkpoint.

**Configurable structure loss + `nn.DataParallel` safety (training infra):**
- `ExtractorConfig` gained `struct_loss` (`"bce" | "bce_posweight" | "focal" |
  "asl" | "dice" | "bce_dice"`) plus per-variant hyperparameters
  (`struct_pos_weight`, `focal_gamma/alpha`, `asl_gamma_pos/neg/clip`,
  `dice_smooth`), and `event_struct_loss` / `event_struct_pos_weight` as
  optional per-task overrides (falls back to `struct_loss` when unset).
- `compute_struct_loss` dispatches to a new `_struct_loss_term()` (per-variant
  loss term: plain BCE, positive-weighted BCE, focal, or asymmetric loss) and
  a new `_dice_struct_loss()` (soft-Dice, optionally combined with BCE via
  `"bce_dice"`; region-based, so it skips the existing random-negative-masking
  path that the other variants use).
- Loss is now tracked as four components instead of three:
  `classification_loss` / `structure_loss` / **`event_structure_loss`** /
  `count_loss`, threaded through `forward()`, `compute_losses()`, and
  `_empty_loss_dict()`. Events get their own loss variant/pos-weight
  (`event_struct_loss`/`event_struct_pos_weight`) precisely so `struct_loss`
  can be tuned for entities/relations independently of events.
- `forward()` and `_empty_loss_dict()` no longer assume
  `next(self.parameters())` succeeds — `nn.DataParallel` replicas expose no
  parameters (`replicate()` stores them as plain attributes), so both methods
  now fall back to the batch's device / fp32 when there is no parameter to
  read device/dtype from.

## 3. Training infrastructure (`gliner2/training/trainer.py`, `training/__init__.py`)

This is the largest and most mmBERT-specific file. `GLiNER2Trainer` /
`TrainingConfig` gained, in rough order of size:

- **Device support** — `_setup_device()` now supports MPS (Apple Silicon) as
  a third tier after CUDA and before CPU (mixed precision and `pin_memory`
  are force-disabled on MPS, since `GradScaler`/fp16-fp16 autocast are
  CUDA-only). It also globally disables the cuDNN fused-attention SDPA
  backend (`torch.backends.cuda.enable_cudnn_sdp(False)`) to work around a
  crash (`mha_graph.execute().is_good() == False`) hit on the long,
  variable-length sequences mmBERT/ModernBERT trains on. DDP setup now reads
  `LOCAL_RANK` from the environment (torchrun) rather than only from
  `config.local_rank`, and picks `nccl`/`gloo` based on CUDA availability.
- **Multi-GPU** — new `_setup_parallel()` wraps `self.model` for forward-pass
  use only (`self._fwd_model`); `self.model` stays the raw module for the
  optimizer/checkpointing. DDP (via torchrun) takes precedence; otherwise
  `nn.DataParallel` is used when `config.data_parallel` is set and ≥2 CUDA
  devices are visible (new config fields `data_parallel` /
  `data_parallel_device_ids`), via the new `gliner2.training.parallel` module
  (`BatchDataParallel`, `_AutocastModule` — re-opens autocast with the
  configured dtype inside each DataParallel replica thread, since
  `parallel_apply` otherwise silently drops back to CUDA-default fp16 even
  under a bf16 config). DDP is configured with `find_unused_parameters=True`
  because different task heads (entity/relation/event) aren't necessarily
  active in every batch.
- **DDP eval/early-stop coordination** — eval + early-stopping decisions now
  run on rank 0 only (`if eval_dataset and self.is_main_process`) and the
  stop decision is broadcast to all ranks via new `_sync_flag()` / `_barrier()`
  helpers, replacing the previous per-rank independent `break`.
- **Gradient checkpointing** — new `config.gradient_checkpointing` +
  `_setup_gradient_checkpointing()`, enabling the encoder's
  `gradient_checkpointing_enable()` (with `use_cache = False`) to trade
  compute for activation memory — the main lever for fitting a large encoder
  in MPS's unified memory.
- **Sliding-window chunking** — new `config.sliding_window` /
  `window_stride`. When enabled, `_prepare_data()` loads records via
  `DataLoader_Factory.load()`, chunks them into overlapping subword-token
  windows via the new `gliner2.training.chunking.chunk_records()`, then
  reshuffles deterministically before constructing `ExtractorDataset`. When
  active, word-level `max_len` truncation in `_create_dataloader()` is
  skipped (chunks are already sized in subword tokens).
- **Checkpoint restart / resume** — new `config.checkpoint_restart`
  (`None | "highest" | "last"`). `_save_training_state()` writes
  `training_state.pt` (optimizer/scheduler state dicts, epoch, global step,
  best metric, patience counter, torch/CUDA RNG state) into numbered
  `checkpoint-*` dirs only (not `best`/`final`, which stay model-only).
  `_find_resume_checkpoint()` / `_restore_training_state()` locate and reload
  it before optimizer creation in `train()`.
- **OOM handling generalized** — `except torch.cuda.OutOfMemoryError` became
  `except RuntimeError as e: if "out of memory" not in str(e).lower(): raise`,
  since MPS raises a plain `RuntimeError` for OOM, not the CUDA-specific
  exception type. `empty_cache()` is now called on whichever backend
  (`cuda`/`mps`) is active.
- **`GradScaler`/`autocast` API migration** — `torch.cuda.amp.{GradScaler,
  autocast}` (deprecated) replaced with `torch.amp.{GradScaler, autocast}`;
  `GradScaler` is now only constructed (as `GradScaler("cuda", enabled=True)`)
  when actually training in fp16 on CUDA — `self.scaler` is `None` otherwise,
  and every scaler use site (`_clip_and_step`, the training loop) now checks
  `if self.scaler is not None` instead of `if self.config.fp16`.
- **Memory management** — new `_free_memory()` (calls `gc.collect()` +
  backend `empty_cache()`) invoked at epoch boundaries and around
  `_evaluate()`, to keep the MPS/CUDA caching-allocator high-water mark from
  climbing across epochs (MPS unified memory has no OS-level OOM recovery).
- **ETA logging** — new `gliner2.training.eta` module and
  `_log_remaining_eta()`, called at the start of `_evaluate()`, projects
  remaining time from the measured steps/elapsed-time rate rather than a
  fixed warmup estimate.
- **Best-checkpoint metrics** — new `_write_eval_metrics()` persists
  `eval_metrics.json` to both `output_dir` and (when it exists) the `best/`
  checkpoint dir whenever a new best metric is reached.
- **`event_structure_loss`** threaded through `TrainingMetrics`, step
  logging, and `_evaluate()`'s aggregation, matching the model-side split
  above.
- **`training/__init__.py`** (new file) exports the public training surface:
  data classes (`Event`, `EventArgument`, `InputExample`, `Relation`,
  `Structure`, `Classification`) plus `estimate_eta`, `compute_metrics` /
  `evaluate_checkpoint` / `make_compute_metrics`, and `build_stopwords` — the
  last four pulled from new sibling modules:
  - `gliner2/training/metrics.py` — `compute_metrics`/`evaluate_checkpoint`,
    strict + relaxed (stopword-aware, normalized-overlap) scoring for
    entities, relations, classifications, and the four event sub-metrics
    (`event_type`, `event_trigger`, `event_argument`, combined `event`).
  - `gliner2/training/eta.py` — pre-training ETA estimate from a data warmup
    sample.
  - `gliner2/training/chunking.py` — subword-window chunking + per-task
    (entity/relation/event) annotation-filtering for chunks.
  - `gliner2/training/parallel.py` — `BatchDataParallel` / `_AutocastModule`
    used by `_setup_parallel()`.
  - `gliner2/training/stopwords.py` — `build_stopwords()`, used by the
    relaxed metrics.

## 4. Packaging (`pyproject.toml`)

- `requires-python` raised from `>=3.8` to `>=3.12` — **a hard breaking
  change** for any `main` user on Python 3.8–3.11.
- New **hard** (non-optional) dependencies: `datasets`, `gliner`,
  `huggingface_hub`, `langcodes[data]`, `lumi-language-id-2`, `numpy`,
  `pyyaml`, `safetensors`, `stopwordsiso`, `tqdm`. Of these, `numpy` and
  `safetensors` are plausibly core-relevant; `datasets`, `langcodes[data]`,
  `lumi-language-id-2`, `stopwordsiso`, `tqdm`, `pyyaml` are training/tooling
  dependencies (corpus loading, language ID for stopword auto-detection,
  YAML config parsing, progress bars) that a pure-inference install of
  `gliner2` does not need.
- New `[dependency-groups] dev` group (`datasets`, `pytest`,
  `pytest-timeout`) and a `pytest.ini_options` marker (`slow`) for tests that
  download models.

## 5. Minor / supporting changes

- **`.gitignore`** — ignores root-anchored `/data/` and `/out/` (local
  training inputs/outputs; anchored so `tools/data/`, the converter package,
  isn't excluded) and `/INSTRUCTIONS.md`.
- **`tests/test_backwards_compat.py`** — three numeric-equivalence tests
  gated behind `@requires_legacy_fixtures`, a `skipif` on gitignored binary
  golden files that can't be regenerated from post-migration code; they skip
  cleanly on a fresh checkout instead of failing.
- **`tests/test_relation_extraction.py`** — new `@pytest.mark.slow`
  regression test asserting `compute_metrics`/`_pred_relation_set` correctly
  parses the engine's real (tuple-based) relation output shape, guarding
  against a scoring bug that silently produced 0.0.
- **`tests/test_torch_free_import.py`** — the existing test is now skipped
  when torch *is* installed (it only makes sense in a torch-free venv);
  previously it presumably just failed in a normal dev environment.
- **`README.md`** / **tutorials 4, 7, 8, 9** — document the events feature
  and add an mmBERT training quickstart pointing at `tools/data/` converters.

---

## Refactor recommendations

The changes fall into three buckets with very different answers to "should
this become a derived class in another directory."

### A. Generic robustness fixes → keep in core, propose to `main` directly

Not mmBERT-specific and not worth isolating:
- Encoder-agnostic embedding-mismatch handling + `tie_weights()` calls in
  `model.py` (`from_pretrained`, `__init__`).
- CUDA → MPS → CPU auto-device-selection in `from_pretrained`.
- `nn.DataParallel`-safe device/dtype resolution in `forward()` /
  `_empty_loss_dict()`.
- CJK-aware tokenization in `processor.py`'s `WhitespaceTokenSplitter`.

These make the core package correct for encoders and platforms it already
claims to support (`from_pretrained` on arbitrary HF checkpoints) — they're
bug fixes, not mmBERT features, and gain nothing from living elsewhere.

### B. Event extraction → keep in core, but as its own PR/feature — not separable into a derived class

Events touch `Schema`, `SchemaInput`, `SchemaTransformer`, `GLiNER2`
(engine), `SchemaAPI`/`GLiNER2API`, and `InputExample` — five classes across
five files, each getting a new task-dispatch branch inside existing methods
(`build()`, `_encode_schemas_and_labels`, `format_results`,
`has_any_task`, `sanitize()`, ...). Subclassing any one of them to "add"
events would mean overriding most of that class's methods wholesale, since
the base methods don't currently have extension points for a new task type
— it would be a reimplementation of ~500 lines, not a thin derived class,
and it would still need every *other* class in the chain to know about the
subclass. This is a general-purpose feature (nothing here depends on
mmBERT or training infra), so the right move is to land it in `main` as its
own feature change, independent of the mmBERT work in bucket C.

### C. mmBERT training infrastructure → derived classes are viable here

This is where "derived classes in another directory" actually fits, because
most of the new behavior in `trainer.py` is already isolated into small,
overridable methods rather than inlined into `train()`. Recommendation:
create a new top-level directory `events/` (sibling to `gliner2/` and
`tools/` — not part of the `gliner2` PyPI package per
`[tool.setuptools.packages.find] include = ["gliner2", "gliner2.*"]`, and not
just a tools script), containing:

```
events/
  config.py    # EventsTrainingConfig(TrainingConfig)
  trainer.py   # EventsTrainer(GLiNER2Trainer)
```

**Overridable as-is** (already isolated as named methods on `GLiNER2Trainer`
in `main`-compatible `trainer.py`, or new methods added wholesale in this
branch — either way, a subclass can override them without touching the base
class further):

| Method | Purpose |
|---|---|
| `_setup_device()` | MPS tier, cuDNN SDPA workaround, torchrun DDP env |
| `_setup_parallel()` | DDP / `nn.DataParallel` wrapping |
| `_setup_gradient_checkpointing()` | new method entirely |
| `_prepare_data()` | sliding-window chunking |
| `_create_dataloader()` | `pin_memory` gating, `max_len` skip under chunking |
| `_save_training_state()` / `_find_resume_checkpoint()` / `_restore_training_state()` | new methods entirely |
| `_log_remaining_eta()` / `_free_memory()` | new methods entirely |
| `_write_eval_metrics()` | new method entirely |

**Needs a small hook added to the base `GLiNER2Trainer` first** — this logic
is currently inlined in the ~200-line `train()` and `_evaluate()` loops on
`mmbert_training`, not factored into its own method, so a subclass can't
override it without copy-pasting the whole loop:

| Inline logic | Suggested base-class hook |
|---|---|
| `GradScaler`/`None` construction + every `if self.scaler is not None` site | `self._make_scaler()` factory, called once in `train()` |
| `autocast(device_type=self.device.type, ...)` + call through `self._fwd_model` instead of `self.model` | Add `self._fwd_model = self.model` to the base `GLiNER2Trainer.__init__` and have `train()` call `self._fwd_model(batch)` instead of `self.model(batch)` — a no-op on `main` (base value equals `self.model`), and `EventsTrainer._setup_parallel()` overrides `self._fwd_model` when wrapping in DDP/DataParallel |
| `except RuntimeError as e: if "out of memory" not in str(e).lower(): raise` | `self._is_oom_error(e)` predicate |
| DDP `should_stop` broadcast + barrier around the eval/early-stop blocks | `self._sync_flag()` / `self._barrier()` (no-ops when not distributed — safe to add unconditionally) |
| `event_structure_loss` read from `outputs` in step logging / eval aggregation | Harmless on `main` today since `outputs.get("event_structure_loss", torch.tensor(0))` already degrades gracefully — no hook needed, this one can just be pulled into the base `TrainingMetrics` handling as-is |
| Eval forward via raw `self.model` (deliberately bypassing `self._fwd_model` since eval runs rank-0-only) | No change needed — already correct as isolated behavior once `self._fwd_model` exists |

Net: 4 small, generically-harmless hooks (`_make_scaler`, `_fwd_model` used
in `train()`, `_is_oom_error`, `_sync_flag`/`_barrier` as no-ops) added to
`main`'s `GLiNER2Trainer`, plus everything in the "overridable as-is" table,
is enough to move ~90% of the current 547-line `trainer.py` diff into
`EventsTrainer` instead of modifying the base class in place.

`config.struct_loss` / loss-variant selection in `model.py` (bucket A vs C
boundary case) — recommend keeping in core `ExtractorConfig`/`Extractor`
rather than a derived `Extractor` subclass. It's config-driven (no
mmBERT-specific code path) and `PretrainedConfig` already accepts arbitrary
kwargs, so any consumer can opt in without a subclass; the only mmBERT-tied
piece is `event_struct_loss`, which is a matter of the *event feature*
(bucket B) needing its own tunable loss, not of mmBERT training per se.
`from_encoder()` similarly belongs in core `Extractor` — it's generically
useful for bootstrapping any GLiNER2 model from a raw encoder, not mmBERT
specific, and `events/trainer.py` will call it via the inherited classmethod.

### What needs to change in `tools/train/`

`tools/train/train.py` currently imports and constructs the base classes
directly:

- `from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig`
  → change to `from events.trainer import EventsTrainer as GLiNER2Trainer,
  EventsTrainingConfig as TrainingConfig` (or update the two call sites
  directly — see below).
- `config = TrainingConfig(**cfg["training"])` (`tools/train/train.py:549`) —
  this line splats the YAML `training:` block straight into the config
  constructor. All the new fields (`gradient_checkpointing`,
  `sliding_window`, `window_stride`, `data_parallel`,
  `data_parallel_device_ids`, `checkpoint_restart`) are set here, so this
  call must target `EventsTrainingConfig`, not the base `TrainingConfig`,
  once the fields move.
- `_build_model()` (`tools/train/train.py:335-356`) calls
  `GLiNER2.from_encoder(...)` / `GLiNER2.from_pretrained(...)` directly.
  Since `from_encoder` stays on the base `Extractor`/`GLiNER2` class (bucket
  A/C-boundary decision above), this seam needs **no change** — `struct_loss`
  and friends stay core `ExtractorConfig` fields, so `model_cfg` kwargs like
  `struct_loss` keep working unmodified.
- Wherever `tools/train/train.py` instantiates the trainer (`GLiNER2Trainer(model,
  config)`), swap to `EventsTrainer(model, config)`.
- `tools/train/TRAINING.md` / `EVALUATION_PLAN.md` reference `GLiNER2Trainer`
  / `TrainingConfig` by name in prose and code samples — update those
  mentions once the import changes.
- No changes needed in `tools/data/` — none of its converters import
  `trainer.py` or `model.py`; they only produce JSONL consumed by
  `InputExample`/`Event`, which stay in core (bucket B).

### Packaging follow-up

If bucket A/B land in `main`, `requires-python>=3.12` and the training-only
hard dependencies (`datasets`, `langcodes[data]`, `lumi-language-id-2`,
`stopwordsiso`, `tqdm`, `pyyaml`) should not go with them as unconditional
`dependencies`. `pyproject.toml` already has an `[project.optional-dependencies]
local` group for `torch`/`transformers`/`peft` — follow that precedent and
add a `training` extra (`pip install gliner2[training]`) for the
corpus/language/YAML tooling deps, keeping the base `pip install gliner2`
install lean for inference-only use. The `requires-python` floor should be
raised only if `main`'s maintainers are prepared to drop 3.8–3.11 support;
that's a separate decision from the mmBERT work and shouldn't be bundled
into either refactor PR.
