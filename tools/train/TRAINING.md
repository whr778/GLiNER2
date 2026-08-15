# Training GLiNER2

This is the working guide for the **boundary architecture** — the one under active
development — and for the **EKF/MHT disaster-tracking** pipeline built on top of it. The
older span architecture still trains and still ships models; it is covered in §10.

Everything here is measured. Where a setting exists because an experiment failed without
it, the experiment is named so you can check the claim rather than trust it.

---

## 1. Which architecture, and why

| | **boundary** | span |
|---|---|---|
| how a mention is scored | start and end **endpoints**, then candidate pairs | every `(start, width)` pair |
| cost in document length | near-linear | quadratic in `max_width` |
| longest mention | unbounded | capped by `max_width` |
| events | triggers + typed role edges, decoded as records | role-typed entities |
| loads with | `AutoExtractor.from_pretrained` | `GLiNER2.from_pretrained` |
| config key | `architecture: boundary` | default |

Use **boundary** for anything document-length or event-shaped. The span head has to
enumerate `(start, width)` candidates, so a 40-token argument means `max_width >= 40` and
the candidate set explodes; the boundary head scores `L` starts and `L` ends and pairs only
the survivors. Removing the width cap is the point.

> **Load a boundary checkpoint with `AutoExtractor`, never `GLiNER2`.** `GLiNER2` *is* the
> span class. A boundary checkpoint fails on `config.max_width` long after the download
> succeeded, so the traceback reads like a bad repo id when it is nothing of the sort.

```python
from gliner2 import AutoExtractor
model = AutoExtractor.from_pretrained("./out/joint-boundary-mmbert-137k/best")
```

---

## 2. Install

```bash
uv sync --extra local
```

The `local` extra carries `torch`, `transformers` and `peft`. Two pins move **together** and
you cannot bump one alone:

```
transformers>=5.6,<5.7     # in the [local] extra
kernels>=0.12,<0.13        # core
```

`uv pip install -e .` (without `[local]`) resolves transformers 5.13 against the core
`kernels` pin. FlashAttention 2 then never hooks, mmBERT runs ~11× slower, and bf16 goes
non-finite around step 50. Set `GLINER2_STRICT_ATTN=1` so the fallback raises instead of
silently degrading, and confirm the encoder reports `kernels-community/flash-attn2`.

On a fresh GPU box: system Python is often 3.10 and this project needs >=3.12
(`uv venv --python 3.12`), and current drivers reject the default cu130 wheel — pin
`torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`.

---

## 3. Data: convert, then prove the splits are clean

### 3a. Convert

Converters live in `tools/data/` and all share the same split flags. See
[`tools/data/README.md`](../data/README.md) for the full corpus list and licences.

```bash
uv run python tools/data/convert_nuner.py --split full --out data/nuner_full.jsonl
uv run python tools/data/convert_text2json.py --emit structures --out data/text2json.jsonl
```

Each writes `<base>.train.jsonl`, `.val.jsonl`, `.test.jsonl` (default 80/10/10, seed 42).

**Splits are grouped by document.** `SplitWriter` routes on a normalized hash of
`record["input"]`, so a source that emits one document several times — text2json emits one
document up to 10 times with 8 different extraction schemas — keeps all its rows in one
split. This is the default and you should not turn it off; `group=None` restores the old
per-row routing and exists only as an escape hatch.

That default is recent, and the corpora predating it leaked badly: text2json's val was
**99.0%** contained in its train, gliclass_logic 38.3%, knowledgator_gliner 27.1%,
events_biotech 21.6%, klue_re 17.3%. **Regenerate a corpus before trusting its eval.**

### 3b. Prove it

```bash
# every corpus: within-split duplicates, cross-corpus overlap
uv run python tools/data/check_leakage.py --pattern 'data/*.jsonl'

# the real gate: does THIS config's aggregated train/val/test overlap?
uv run python tools/data/check_leakage.py --config tools/train/config/joint-boundary-mmbert-137k.yaml
```

The `--config` form is the one that matters and it exits non-zero on contamination. Checking
corpora one at a time is not sufficient: a training mix pools many corpora, and corpus A's
train can hold a document sitting in corpus B's test — neither file overlaps itself, yet the
blind test is contaminated. The report attributes every overlap to the file pair responsible.

### 3c. The trainer gates it anyway

`tools/train/train.py` runs the same check before a single step and repairs it:

```
[split hygiene] REPAIRED
    train    135087 records  (dropped 1093 exact duplicate(s))
    val        5818 records  (dropped 26 exact duplicate(s))
    test      15371 records  (dropped 115 exact duplicate(s))
    removed 21 document(s) from train: also present in test
    removed 746 document(s) from train: also present in val
    removed 22 document(s) from val: also present in test
```

Two rules, and conflating them destroys real supervision:

- **within** a split, only an exact repeat — same text **and** same target — is dropped;
- **across** splits, any shared **text** is contamination whatever the targets say.

**Test is authoritative and never modified.** Val yields to test, train yields to both, so
removal always comes out of the lower-priority split and the blind set stays exactly what
the corpus declared.

`training.split_hygiene:` selects `drop` (default), `raise`, `warn` or `off`. Use `warn` to
reproduce a pre-gate run unchanged.

---

## 4. The boundary recipe

Two stages. Train a base on a large mixed corpus, then warm-start the downstream from it.

### Stage 1 — the base

```bash
uv run torchrun --standalone --nproc_per_node=2 tools/train/train.py \
  --config tools/train/config/joint-boundary-mmbert-137k.yaml
```

`batch_size` is **per GPU** and accumulation is halved so the effective batch stays 32 on one
GPU or two. Moving to a single GPU without restoring accumulation silently halves your batch.

The curve `{10k, 40k, 100k, 137k}` exists because base training volume is the variable under
test — see [`JOINT_IE_SCALING.md`](../events_working_papers/JOINT_IE_SCALING.md) §4.

### Stage 2 — warm-start the downstream

```bash
uv run torchrun --standalone --nproc_per_node=2 tools/train/train.py \
  --config tools/train/config/joint-boundary-rams-137k.yaml
```

`model.pretrained` points at `./out/joint-boundary-mmbert-137k/best` and `architecture:
boundary` is checked against the checkpoint — warm-starting a span checkpoint raises rather
than silently training the wrong architecture.

**Warm-starting forgets.** Training only on the added task destroys the ones you are not
training: the casualty fine-tune used a narrow homogeneous schema with zero replay and
afterwards returned a digit when asked for a `location`. Mix 5–10% of the original
distribution back in — `build_warmstart_mix.py` samples replay across **both** old task
families, because taking only events preserves one head and starves the other.

### What the configs encode, and why

| Setting | Reason |
|---|---|
| `struct_loss: bce_posweight`, `struct_pos_weight: 4.0` | Bootstraps the argument head. Plain `bce` left argument recall near **0** on fresh heads; focal **collapsed** it. |
| `bf16: true` + FlashAttention 2 | On mmBERT, **sdpa + bf16 produces NaN** (reproduced on 1×H100 at step 15). FA2 is also 11× faster. |
| `attn_implementation` unset on the `pretrained` path | It is applied to `model.config` *after* the encoder is built — a silent no-op. FA2 is inherited from the base checkpoint. |
| `metric_for_best: eval_event_argument_strict_micro_f1` | Arguments are what head-init moves; selecting on loss hides it. |

### Overrides on the `pretrained` path

`model.boundary_head:` keys are merged into the loaded checkpoint's settings and
`boundary_settings` is rebuilt — **including the head's own reference**, which is a separate
object built in `BoundaryHead.__init__`. Rebuilding only `model.boundary_settings` left every
knob the head reads through `self.settings` pinned at its checkpoint value, which produced a
treatment arm identical to its control. Structural keys (`enable_records`, `candidate_pool`,
`boundary_dim`, …) raise instead of being applied, because the modules they size are already
built.

---

## 5. Loss balance: measure before you tune

The boundary loss decomposes by **mechanism** (start / end / pair / inside / soft-IoU /
rerank / proposal / abstention / count), not by task. Every task's supervision is comingled
into one scalar, so "is the event signal too small?" is not answerable by reading the loss.

### 5a. Look at where the gradient actually goes

```bash
uv run python tools/train/probe_task_losses.py \
  --config tools/train/config/warmstart-natural.yaml \
  --checkpoint out/.../final --batches 100 --gold-injection 0.25
```

It runs the **training** path — train mode, the training collator, the scheduled gold
injection — and splits every query-typed term into per-task contributions that sum back to
the scalar the optimizer sees (reconciliation is reported; expect ~1e-7).

Measured on a converged warm-start checkpoint:

| task | share of the training gradient |
|---|--:|
| entities | 77.2% |
| json_structures | 11.5% |
| **events** | **6.6%** |
| relations | 2.5% |

Entities dominate. That is the imbalance, and it is not visible without the buckets.

### 5b. Three levers, and their reach

```yaml
model:
  boundary_head:
    task_loss_weights:        {entities: 1.0, relations: 1.0, events: 2.0, json_structures: 1.0}
    task_loss_weight_scope:   all      # "span" (default) | "all"
    task_pos_weights:         {events: 8.0}
    report_task_losses:       true     # diagnostic only, no gradient effect
```

- **`task_loss_weights`** scales a task's whole term — magnitude only.
- **`task_loss_weight_scope`** decides what that reaches. `span` (default) is start/end/pair
  = **18.5%** of the loss; `all` adds inside, soft-IoU, rerank, proposal, abstention and
  count for **94.3%**. A dose sweep at `span` reach was null on every metric precisely
  because `w=4` moved events from 6.6% to 10.6% of the gradient while three quarters of it
  sat untouched.
- **`task_pos_weights`** scales **positives against negatives inside** a task's queries —
  direction, not magnitude. Applies to start/end/pair/inside. Not soft-IoU (fractional
  targets, so "the positive term" is undefined) and **ignored** by the
  `asymmetric_focal` marginal path, which never calls `_safe_bce`.

Dose a `pos_weight` from the measurement, not a hunch: `k` multiplies a task's contribution
by `(k*pos + neg) / (pos + neg)`, and the probe prints `pos` and `neg`. The regime matters —
at **initialization** the event positive fraction is 0.052 (balance at `k≈18`), but at
convergence it is 0.562, where `k>1` does not correct an imbalance so much as create the
opposite one.

**Defaults are exactly inert.** An all-ones weight map takes the original code path, not a
numerically-equivalent one, so an arm-to-arm difference is the treatment and not the
plumbing.

---

## 6. EKF/MHT disaster tracking

**The EKF has no learned parameters.** `est_ekf` is a censoring-aware random-walk smoother
whose smoothing strength is a hand-set constant (`q_rel = 0.20`), with fixed source/qualifier
fusion weights. There is no fitting step and no checkpoint. What you *train* is the
**casualty structure model** whose observations the filter consumes; everything else is
generation, configuration and measurement.

Full stage map, including what is off by default:
[`PIPELINES.md`](../events_working_papers/PIPELINES.md) §2.

### Step 1 — generate synthetic streams (free, seeded, exact ground truth)

```bash
uv run python datasets/disaster_streams/generate.py \
  --out datasets/disaster_streams --n-train 400 --n-val 60 --n-test 60 --seed 42
```

Each stream is one disaster whose true state evolves (dead/injured approach an asymptote,
missing decays) observed by noisy, *hedged* reports. Parametric and deterministic, so ground
truth is exact and it costs nothing.

### Step 2 — realize the streams as news text (this costs money)

```bash
uv run python datasets/disaster_streams/realize.py --split train --provider anthropic \
  --model claude-sonnet-5 --estimate          # price it first
uv run python datasets/disaster_streams/realize.py --split train \
  --provider anthropic --model claude-sonnet-5 --out datasets/disaster_streams_sonnet5
```

`--provider mock` spends nothing and is the smoke-test path. Observations are grouped into
one multi-fact snippet per report, so extraction has to bind each figure to the right role
amid competing numbers.

### Step 3 — build the structure corpus

```bash
uv run python datasets/disaster_streams/build_multievent_corpus.py \
  --data datasets/disaster_streams_sonnet5 --split train \
  --out data/casualty_multi.train.jsonl --max-interference 3 --record-mode natural
```

**Build from `train` streams only** — the showcase feeds come from `test`, and that
separation is what keeps the evaluation uncontaminated.

Use the multi-event build unless you have a reason not to. The single-event corpus has
exactly one record in all 31,539 documents, so the count head only ever saw "1"; on a
multi-incident document value binding collapses from **1.000 to 0.369**, with **22.6%** of
readings bound to the wrong event's number.

### Step 4 — fine-tune the structure model

```bash
uv run python tools/train/train.py --config tools/train/config/casualty-multievent.yaml
```

`casualty-finetune.yaml` is the single-event recipe that closed ~75% of the gap to the
structured ceiling; `casualty-multievent.yaml` holds it fixed and changes **only the
corpus**. Note `fp16: true` — the sdpa+bf16 defect is a ModernBERT problem and does not
apply to DeBERTa-v3.

### Steps 5–8 — real feed, scope hierarchy, run, score

```bash
uv run python tools/ekf_showcase/harvest_helene_gt.py --out datasets/helene2024/ground_truth.json
uv run python tools/ekf_showcase/build_helene_feed.py --max-articles 120

uv run python tools/ekf_showcase/run_pipeline.py \
  --feed datasets/helene2024/_cache/feed.jsonl \
  --truth datasets/helene2024/ground_truth.json \
  --out datasets/helene2024/_cache/tracked.json \
  --casualty-model ./out/casualty-multievent/best \
  --record-mode natural --associate record \
  --rollup datasets/helene2024/rollup.json --event-year 2024 \
  --window article --device cpu

uv run python tools/ekf_showcase/score_helene.py \
  --tracked datasets/helene2024/_cache/tracked.json \
  --truth datasets/helene2024/ground_truth.json --role dead
```

Ground truth and feed are **deliberately different sources** (Wikipedia's per-state casualty
table vs AP prose). That is what makes `est_last_value` a real baseline rather than an
oracle: in the Türkiye–Syria run truth was read from the same sentence the extractor reads,
so the baseline scored 0.000 by construction and the filter was unmeasurable.

**Everything from the temporal filter down is OFF unless you pass it** — `--event-year`,
`--rollup`, `--record-mode`, `--associate`. The defaults reproduce the older numbers, so a
run that omits them is not the current pipeline.

- `--associate record` takes the location from the record's own field; `envelope` falls back
  to nearest-location-by-character-distance.
- `--event-year 2024` drops figures dated before the event — Izmit 1999's 17,500 was being
  bound as a 2023 observation 15 times.
- `--device cpu` is usually right: **MPS is 3–4× slower** here, from per-op sync overhead.

### The known weak point

Association is the research blocker, **not** the filter. Proximity, GPE tags, record-location
and admin rollup have all been tried; the scope gate was the first thing that moved it
(per-state nRMSE 5.247 → 0.591, against a random-removal control at 4.427). **No MHT is
built** — `PIPELINES.md` §4 has the measurement showing the oracle headroom does not
currently justify one.

---

## 7. Evaluate

```bash
uv run python tools/train/eval.py --config <config> --checkpoint out/<run>/best --split test
```

Three rules that have each changed a conclusion in this project:

1. **Sweep the threshold per checkpoint, then compare at matched thresholds.**
   `metric_sweep: true` selects each checkpoint at *its own* best threshold — right for
   shipping one model, **wrong for a curve**. Re-DocRED alternated 0.1/0.3 and looked
   non-monotonic until split by threshold. `test_metrics.json` does not record the
   threshold; pull `best/threshold_sweep.json`.
2. **Judge deltas against a measured noise floor, not a guess.** Re-running an identical
   control on a second seed gave relation strict **±0.041** on `mix_natural`, four times the
   0.01 that had been assumed — large enough to void a published-looking result.
3. **Report per capability.** A single aggregate hides which capability moved which way.

Structures are **not scored by the blind test**: `_schema_from_gold` builds no schema for
`json_structures`, so structure-only records are skipped — 35.1% of `mix_natural`'s val. Use
`tools/train/probe_records.py` for structure quality.

---

## 8. Push to the Hub

```bash
uv run huggingface-cli login          # write-scope token, once per machine
# or: export HF_TOKEN=hf_xxx          # headless; never pass a token in argv

uv run python tools/train/push_to_hub.py \
  --checkpoint ./out/joint-boundary-mmbert-137k/best \
  --repo-id <username>/gliner2-joint-boundary-mmbert-137k \
  --private
```

`--private` is the default. The repo layout matches what `from_pretrained` expects
(`config.json` + `encoder_config/config.json` + `model.safetensors` + tokenizer files). A
`MODEL_CARD.md` is generated at the end of training with the datasets actually used, their
licences, and an effective-licence determination from `dataset_registry.yaml`.

Sanity-check with `AutoExtractor`, not `GLiNER2`, for boundary checkpoints.

---

## 9. Hardware

The base configs train at `max_len=4096` with `sliding_window: true`; mmBERT's positional cap
is 8192. Local-global attention keeps memory near-linear in sequence length, but activations
and optimizer state still scale with it.

| Device | mmBERT-small (148 M) | mmBERT-base (314 M) |
|---|---|---|
| **A100/H100 80 GB** | `batch_size=16–24`, bf16 | `batch_size=8–12`, bf16 |
| **A100/4090 40–48 GB** | `batch_size=8–12`, bf16 | `batch_size=4` + accum 2, bf16 |
| **24 GB (3090/4090)** | `batch_size=4–6`, bf16 | `batch_size=2` + accum 4, bf16 |
| **Apple MPS** | `batch_size=1–2`, no AMP, dev only | `batch_size=1`, no AMP, dev only |
| **CPU** | smoke tests only | smoke tests only |

On MPS, mixed precision is disabled automatically (`GradScaler` is CUDA-only) and the trainer
logs the choice. MPS also hits a Metal assertion with relations in the batch on bf16 mmBERT —
use CPU for local debugging.

---

## 10. The span architecture

Still supported and still shipping models (`fastino/gliner2-base-v1` and the
`gliner2-*-v1-*` configs). Train it the same way, minus `architecture: boundary`:

```bash
uv run python tools/train/train.py --config tools/train/config/mmbert-small.yaml
```

```python
from gliner2 import GLiNER2
model = GLiNER2.from_pretrained("./out/mmbert-small/best")
model.extract_entities("Marie Curie discovered radium in Paris.", ["scientist", "element", "city"])
```

`max_width` is a span-head field and must **not** appear in a boundary config. Everything in
§3 (data hygiene) and §7 (evaluation) applies to both architectures.

---

## 11. Tips

- **Smoke-test 50 records first** before a multi-day run: `--max-records 50`, `num_epochs=1`,
  `batch_size=2`, and confirm the loss falls.
- **Watch the loss curve.** Healthy mmBERT-small starts ~500 (batch 1), drops below 100 in
  ~50 steps, drifts to 40–60 over an epoch. Collapsing to ~0 means labels are not reaching
  the loss head — stop and inspect.
- **Resume** by pointing `model.pretrained` at `./out/<run>/checkpoint-epoch-<N>` (or
  `best`/`final`). The trainer starts a fresh optimizer and scheduler; that is intentional.
- **Run scripts from files, not heredocs.** `python - <<PY` breaks DataLoader workers under
  the spawn start method: workers re-import the main module by path and stdin has none.
- **W&B**: `report_to_wandb=True` + `wandb_project="..."` in `TrainingConfig`.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch`/`transformers` | `local` extra not installed | `uv sync --extra local` |
| bf16 loss goes non-finite ~step 50, and training is ~11× slow | FA2 never hooked; transformers resolved outside the pinned range | `uv pip install -e ".[local]"`, verify `kernels-community/flash-attn2`, set `GLINER2_STRICT_ATTN=1` |
| `config.max_width` error loading a checkpoint | Loaded a boundary checkpoint with `GLiNER2` | Use `AutoExtractor.from_pretrained` |
| A treatment arm scores identically to its control | A `boundary_head` override was dropped | Confirm the value on `model.boundary_head.settings`, not just `model.config` |
| CUDA `Error 802: system not yet initialized`, but `nvidia-smi` is healthy | Host-side fault (no NVSwitch, `GPU Fabric GUID: N/A`) | Not fixable in-guest. Terminate and relaunch, ideally another region |
| Blind-test scores look too good | train/test share documents | `check_leakage.py --config <cfg>`; regenerate the corpora it names |
| Eval returns `{}` and checkpoint selection falls back to loss | Every record produced an empty schema | Trigger-only corpora need role-less event types kept; structures are not scored at all (§7) |
| `FileNotFoundError: '<stdin>'` + `BrokenPipeError` | Training launched from a heredoc with `num_workers>0` | Save to a `.py` file, or `num_workers=0` |
| `out of memory` | Batch too large | Halve `batch_size`, double `gradient_accumulation_steps` |
| `401`/`403` from `push_to_hub.py` | Missing or read-only HF token | Write-scope token via `huggingface-cli login` or `HF_TOKEN` |
