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
`kernels` pin, which breaks FlashAttention 2 — see below for what that costs and how to
catch it.

### `kernels-community/flash-attn2`: two failures with the same name

That string appears in both, and they need opposite fixes. **Check which one you have
before changing anything.**

**On CUDA — FA2 silently never hooks.** transformers reads `flash_attention_2` as the
*pip* package, whose prebuilt wheels stop at cp313/torch2.9, so on this stack it is always
rejected; the kernel that actually loads is served from the Hub registry under the repo id
`kernels-community/flash-attn2`, which needs `kernels` at a version compatible with the
installed transformers. When that pin slips the loader degrades to sdpa with one warning in
a log full of them, and on a bf16 ModernBERT encoder **sdpa is a correctness failure, not a
slowdown**: ~11× slower and non-finite losses around step 50. Fix by installing the extra
that pins both together, and by making the degrade loud:

```bash
uv sync --extra local          # NOT `uv pip install -e .`, which resolves transformers 5.13
export GLINER2_STRICT_ATTN=1   # turn the silent fallback into a load-time error
```

**Do not `uv add flash-attn`.** The pip package is not the path this stack uses and is
not in the lockfile: its prebuilt wheels stop at cp313/torch2.9, so on Python 3.12 +
torch 2.11 there is no wheel to match and the install falls back to a source build that
needs `nvcc` (and cannot work on a Mac at all). FA2 comes from the Hub registry via
`kernels`. The confusion is that both are called the same thing — transformers reads the
string `flash_attention_2` as *the pip package*, fails to find it, and only then retries
under the Hub repo id, which is the attempt that succeeds.

**Off CUDA (Mac, CPU box) — `KeyError: 'kernels-community/flash-attn2'` on the first
forward.** Loading a checkpoint that was *trained* with FA2 — every mmBERT checkpoint here
stores `attn_implementation: flash_attention_2`. Off CUDA transformers accepts the plain
name at load, normalizes it to the hub repo id, and nothing raises until the first forward
looks the kernel up and does not find it. Construction alone looks healthy. Fixed in
`8af9c5f`: off CUDA both spellings now degrade to sdpa *at load*. On an older checkout,
pass a config whose `attn_implementation` is `sdpa`.

**Verify what you actually got — the model config reports the REQUEST, not the result:**

```python
m = AutoExtractor.from_pretrained(ckpt, map_location="cpu")
m.config.attn_implementation           # 'flash_attention_2'  <- what the checkpoint asked for
m.encoder.config._attn_implementation  # 'sdpa'               <- what actually loaded
```

Reading only the first is how a run gets recorded as "FA2" while training on sdpa.

On **MPS**, sdpa is swapped for Metal FlashAttention automatically whenever the device
resolves to mps (`mps-flash-attn`, a darwin-only dependency). Nothing to configure; if the
package is missing you get a warning and stock sdpa.

On a fresh GPU box: system Python is often 3.10 and this project needs >=3.12
(`uv venv --python 3.12`), and current drivers reject the default cu130 wheel — pin
`torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`.

---

## 3. Data: convert, then prove the splits are clean

### 3a. Rebuilding `data/` from scratch

`data/` is gitignored and holds ~12 GB that converters, bought annotation and several
label-unification passes produced. Restore it from the private Hub mirrors; only convert
what is not mirrored.

```bash
# what would be pulled, and what CANNOT be
uv run python tools/data/restore_from_hf.py --all --dry-run

# just one run's data, or everything
uv run python tools/data/restore_from_hf.py --config tools/train/config/joint-boundary-mmbert-137k.yaml
uv run python tools/data/restore_from_hf.py --all
```

516 files are mirrored and the 137k config restores 36/36. Every `--all` run prints an
UNRECOVERABLE list; that is a backup gap to close, not noise. Files already present are
skipped, so this is safe to re-run — `--force` re-downloads.

**To rebuild from source instead of the Hub, do not hand-run 36 converters.**
`tools/data/run_all_converters.sh` runs every one of them in the right order, logs each
step, applies the zh→en label map, and refuses to finish if Chinese labels or contaminated
splits survive:

```bash
tools/data/run_all_converters.sh          # log: /Volumes/Development/tmp/converters.log
```

It builds BASE corpora only — it does not build the derived corpora and never runs the
`annotate_*` scripts, which cost money. Its exact invocations and build order are the
reference for the per-corpus commands below.

**What the Hub cannot give you, and what rebuilds it:**

| corpus | rebuild with |
| --- | --- |
| `anatem`, `bionlp09/11epi/11id/13cg/13ge/13pc`, `ex_ptm` | `convert_mtl_bio.py --dataset <AnatEM\|BioNLP09\|Ex-PTM\|…>` |
| `bc5cdr` | `convert_hf_token_ner.py --repo tner/bc5cdr --revision refs/convert/parquet` |
| `ace2005` | `convert_ace2005.py` — LDC licence, source not redistributable |
| `klue_ner` / `scierc` / `paraloq_json` / `stockmark_jpn` | `convert_klue.py` / `convert_scierc.py` / `convert_paraloq_json.py` / `convert_stockmark_ner.py` |
| `maven_ner` | `events_to_entities.py`, derived from `maven` |
| `wikiann` | never stored — streamed at train time |

**A fresh conversion re-emits the ORIGINAL labels, and that is the trap.**
`convert_duee.py` and `convert_docfee.py` produce Chinese entity keys, event types, roles
and classification MENUS — exactly the state the unification removed. A corpus rebuilt
from source and dropped into a config trains a label space the base never learned:

```bash
uv run python tools/data/apply_label_map.py --map tools/data/label_map_zh_all.json \
  data/duee.*.jsonl data/docfee.*.jsonl data/text2json.*.jsonl
```

Then verify the three ways CLAUDE.md requires: zero variant clusters, zero label uses
lost, and the map CLOSED (no target is itself a key).

**Order matters — every derived corpus is built FROM the base ones, so a stale base
propagates silently:**

1. base corpora — restore, or the converters in 3b
2. label map applied (above)
3. slices — `build_scaling_mix.py` → `data/scaling/`, `build_joint_scaling_mix.py` → `data/scaling_joint/`
4. mixes and replay — `build_warmstart_mix.py`, `build_137k_replay.py`,
   `build_turkish_dose_mix.py` (`mix_natural`, `tr_dose*`), `build_loc_control.py`,
   `build_zh_multitask_mix.py`
5. `check_leakage.py --config <yaml>` before any of it reaches a run

**Restoring by basename alone is unsafe and the tool refuses to.**
`data/scaling_joint/chfinann.val.jsonl` is a 150-record slice while `whr778/chfinann`
holds the 3,204-record val under that same name. Subdirectories resolve through
`jsonl_dirs` in `dataset_registry.yaml`; an unregistered one refuses rather than fetching
the parent's file. Parent repos also still carry stale copies of the slices at their root
— `whr778/docfee` serves a `docfee.j100k.test.jsonl` with 1,983 Chinese labels — so slice
files are only ever taken from the directory mirror.

### 3b. Convert

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

### 3b-2. Buying annotation: select candidates, then submit

Adding a language means buying labels. The cost is per DOCUMENT and the value is per
POSITIVE, so **the filter's precision is the price** — see
[`tools/data/notes/ANNOTATION_ECONOMICS.md`](../data/notes/ANNOTATION_ECONOMICS.md) for
the full method and the measured table.

**Step 1 — build the candidate pool.** Never submit a raw corpus. Only the *ambiguous*
region is worth paying for: text with no casualty cue is a free negative and must never
reach the API.

```bash
# Turkish: 18 of 31 parquet shards, chosen for outlet spread
uv run python tools/data/build_turkish_pool.py \
  --out /Volumes/Development/data/turkish_pool.jsonl \
  --exclude data/turkish_gate/gate_ann_tr_heldout_full.jsonl   # never re-buy what you own

# Simplified Chinese: streams shaowenchen/news_zh, filters cue + script
uv run python tools/data/build_chinese_candidates.py --limit 5000
```

Three things every candidate builder must do, each of which has cost real money here:

- **Cue-filter.** The cue selects what gets *adjudicated*; the adjudication, not the
  regex, assigns the label. Keep it broad — a narrow cue pre-judges the cases you are
  paying to resolve.
- **Exclude what you already own**, by normalised group key. An eval document that
  re-enters the training pool is the cross-set contamination the split rule forbids.
- **Filter script/language explicitly**, rather than assuming. `news_zh` is 93.93%
  Simplified and 0.03% Traditional; that 0.03% would still train characters the
  deployment never sees.

**Step 2 — price it before you buy.** Measure purity on a labelled sample of the POOL,
not of a similar corpus:

```bash
uv run python tools/data/gate_purity_curve.py --model whr778/gliner2-gate2-mmbert-tr
```

This exists because the first Turkish estimate was wrong by 3x: it used a base rate from a
different source (42.3% vs the real 25.1%) and an *assumed* gate purity of 75% that
measured 37.3%. Composing a free regex with the model gate reached 78.8% purity — better
than either alone, and it halves what the model must score.

**Step 3 — submit as a batch.** Always `--batch` (50% cheaper). Two annotators:

```bash
# stage-0 gate labels: current_toll / historical_toll / exposure_only / no_toll
uv run python tools/data/annotate_gate.py --batch \
  --corpora data/chinese_gate/zh_candidates.jsonl \
  --cue "$(uv run python -c 'import sys;sys.path.insert(0,"tools/data");from build_chinese_candidates import CUE;print(CUE.pattern)')" \
  --out data/chinese_gate/zh_gate_sample

# stage-2 field-level records: location / dead / injured / missing
uv run python tools/data/annotate_casualty.py --batch \
  --corpora data/turkish_gate/cas_candidates_top38k.jsonl \
  --out data/turkish_gate/cas_ann_tr
```

**RECORD THE BATCH ID IMMEDIATELY**, in
[`tools/data/notes/TURKISH_BATCHES.md`](../data/notes/TURKISH_BATCHES.md). A killed poller
does not lose the money — the batch completes server-side — but only if the id survives:

```bash
uv run python tools/data/annotate_casualty.py --fetch-batch msgbatch_... --out <same-out>
```

Resubmitting pays twice for identical output. `data/` is gitignored, so an id written
there is not durable; commit it under `tools/`.

**Step 4 — verify before training.** `annotate_casualty.py` enforces the two contracts
that make a record trainable and prints the drop rate: values must be VERBATIM substrings
(the boundary head locates fields as spans, so a paraphrase trains nothing and reports no
error) and counts must be bare numerals keeping the source's own scale words, so Turkish
`29 bin 313` survives. Then check split hygiene against everything you already own —
0 overlap, 0 duplicates — before the data reaches a config.

### 3c. Prove it

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

### 3d. The trainer gates it anyway

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

## 3e. One label space, shared by a base and everything warm-started from it

A label is an INPUT to GLiNER2 at inference, so two spellings of one concept are two
different queries. `CompanyName` (chfinann) and `Company Name` (docfee) are one field of
one taxonomy written twice, and a model warm-started from a base that learned the first
will be asked for the second.

Unify in the CONFIG, never by rewriting corpora:

```yaml
labels_file: labels/unified.yaml     # resolved relative to the config
```

`tools/train/config/labels/unified.yaml` is GENERATED. Regenerate after any data change:

```bash
uv run python tools/train/build_label_maps.py <configs and/or jsonl...> \
    --canonical tools/train/config/joint-boundary-mmbert-137k.yaml \
    --out tools/train/config/labels/unified.yaml
```

`--canonical` names the corpora whose spellings WIN, and it is not optional for a base
with downstream models: without it, cc_news_haiku45's 427,160 lowercase `location`
outvoted the base and flipped the canonical spelling to one the base never learned.

**Five categories**, each with `rollup` / `separator` / `map`: `entities`, `relations`,
`events` (types AND argument roles), `classifications` (menu AND answers), and
`structures` -- json_structures names AND field names, carrying `record_metadata` keys and
`anchor` with the rename. Without the last one the EKF surface was the only label surface
with no lever, and a missed `anchor` decodes to `{}` in silence.

### Four ways this goes wrong quietly

- **An EMPTY inline `labels:` block OVERRIDES `labels_file`.** 21 of the 27 warm-started
  configs carried `{rollup: false, map: {}}`; wiring the file without deleting the block
  is a no-op that looks done. `tests/test_train_configs.py` pins that every config
  warm-started from the base resolves the SAME maps.
- **The map must be CLOSED.** It is applied ONCE, so `LOC: Location` beside
  `Location: location` leaves both spellings alive. `build_label_maps` refuses to emit a
  map whose target is itself a key.
- **Roll-up runs BEFORE the map**, so a map key containing the separator can never fire.
  Pinned by a test.
- **A missing input is refused, not skipped.** The map has twice been built from fewer
  corpora than intended -- once because zsh does not word-split an unquoted `$VAR`, once
  because a file list did not survive a reboot -- and both times it looked like a clean
  run while silently dropping a corpus.

### What must NOT be merged

Prove two labels mean the same thing by reading the surfaces they tag. Known keeps:
`GPE` != `Location` (Thailand vs the Indian Ocean), `NORP` != `Organization`
(nationalities), `FAC` != `Location` (buildings), redocred's `TIME` holds dates, and
docee's `Target` (who an attack hit) != bio_ner_relations' `target` (a kinase substrate).
Only TAXONOMY corpora may drive the map: `mix_natural` is 58% placeholder labels
(`e_0` x30,034) and `zh_multitask` is 70% singletons.

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

Six rules that have each changed a conclusion in this project:

1. **Sweep the threshold per checkpoint, then compare at matched thresholds.**
   `metric_sweep: true` selects each checkpoint at *its own* best threshold — right for
   shipping one model, **wrong for a curve**. Re-DocRED alternated 0.1/0.3 and looked
   non-monotonic until split by threshold. `test_metrics.json` does not record the
   threshold; pull `best/threshold_sweep.json`.
2. **Judge deltas against a measured noise floor, not a guess.** Re-running an identical
   control on a second seed gave relation strict **±0.041** on `mix_natural`, four times the
   0.01 that had been assumed — large enough to void a published-looking result.
3. **Report per capability.** A single aggregate hides which capability moved which way.
4. **A flat aggregate is not a null result — but rule 1 still applies to the paired test.**
   gate2 v2 scored `relevance` 0.8341 against v1's 0.8368 and was written off. Scored on
   identical rows stratified toward the hard classes it wins 37 to 17, exact McNemar
   p = 0.0091 — and that result does **not** survive: with each model at the threshold its
   own validation split chooses, the score is **18 / 18, p = 1.0000**. The models are
   indistinguishable; the p = 0.0091 was measuring the gap between two *operating points*
   that a shared argmax happened to put in different places. Paired testing is necessary and
   not sufficient — pair the rows *and* match the thresholds, or the test just launders a
   calibration difference into a capability claim.
5. **Give every class enough rows to have an opinion.** That same gate's `exposure_only`
   accuracy was recorded at 0.250 from a 16-row sample; on all 72 rows in the split it is
   0.431. Part of the gap being chased did not exist. Take *every* row of the scarce classes
   and cap only the plentiful ones.
6. **`argmax` is an operating point, not a neutral default, and a saturated softmax makes
   it a bad one.** The same gate needs threshold **0.998** to sit at its stated recall bar;
   0.5 falls deep inside its positive region. Moving there takes false positives on
   exposure text (`220,000 victims have been served meals`) from **0.556 to 0.097** and
   overall accuracy from 0.719 to 0.847, at 0.70 recall — no retraining. Every number
   recorded for this gate before that, including a 21.5% Turkish FP rate, was measured at
   the broken default. Choose the threshold on **val**, then report test once.

`tools/ekf_showcase/gate_perclass.py --sweep` does 4, 5 and 6 for the relevance gate — stratified
sampling, paired scoring, exact McNemar (the normal approximation reports p = 1.32 at three
discordant pairs each way). Its per-class p-values are labelled exploratory in the output:
five uncorrected comparisons are not five results.

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

**`batch_size` is PER-GPU, so one config is two different recipes.** This is the single
most expensive gotcha in this file: the same YAML launched two ways trains two different
models, and only the launch command differs.

    1 GPU : batch 4 x accum 4       = effective 16, 42,730 steps
    2 GPUs: batch 4 x accum 4 x 2   = effective 32, 21,365 steps  <- HALF the updates

Measured on joint-boundary-mmbert-137k. The 2-GPU launch was read as a label-space
regression for a day: relation strict F1 fell 0.098, with precision nearly doubling
(0.202 -> 0.383) while recall collapsed (0.212 -> 0.064) -- the signature of an
under-trained head, since relations are the minority task at 26.98% of the mix. Rebuilding
at the 1-GPU recipe recovered +0.098 of it.

Under `torchrun --nproc_per_node=N`, divide `gradient_accumulation_steps` by N to keep the
effective batch, or accept an off-curve model and re-baseline everything downstream.

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

For **inference** the MPS verdict is workload-dependent, and both directions are measured:
a 2-label mmBERT classification runs **~28% faster** on MPS than CPU (188 vs 265 ms/row)
once warm, after a ~380 ms/row first round of Metal shader warm-up that one-shot callers
pay and batch jobs do not; a 139-type event decode is **3–4× slower** on MPS, because that
path is many tiny ops with CPU/GPU syncs. `mps-flash-attn` replaces SDPA automatically
whenever the device resolves to MPS. Measure the workload; neither device wins by default.

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
| bf16 loss goes non-finite ~step 50, and training is ~11× slow | FA2 never hooked; transformers resolved outside the pinned range | `uv sync --extra local`, then check `m.encoder.config._attn_implementation` (§2) and set `GLINER2_STRICT_ATTN=1` |
| `KeyError: 'kernels-community/flash-attn2'` on the first extract, off CUDA | The checkpoint config stores `attn_implementation: flash_attention_2`; transformers accepts it there, normalizes it to the hub repo id, and only the forward discovers no kernel is registered | Fixed in `8af9c5f`: off CUDA both spellings degrade to sdpa at load. On an older checkout, pass a config with `attn_implementation: sdpa` |
| `config.max_width` error loading a checkpoint | Loaded a boundary checkpoint with `GLiNER2` | Use `AutoExtractor.from_pretrained` |
| A treatment arm scores identically to its control | A `boundary_head` override was dropped | Confirm the value on `model.boundary_head.settings`, not just `model.config` |
| CUDA `Error 802: system not yet initialized`, but `nvidia-smi` is healthy | Host-side fault (no NVSwitch, `GPU Fabric GUID: N/A`) | Not fixable in-guest. Terminate and relaunch, ideally another region |
| Blind-test scores look too good | train/test share documents | `check_leakage.py --config <cfg>`; regenerate the corpora it names |
| Eval returns `{}` and checkpoint selection falls back to loss | Every record produced an empty schema | Trigger-only corpora need role-less event types kept; structures are not scored at all (§7) |
| `FileNotFoundError: '<stdin>'` + `BrokenPipeError` | Training launched from a heredoc with `num_workers>0` | Save to a `.py` file, or `num_workers=0` |
| `out of memory` | Batch too large | Halve `batch_size`, double `gradient_accumulation_steps` |
| `401`/`403` from `push_to_hub.py` | Missing or read-only HF token | Write-scope token via `huggingface-cli login` or `HF_TOKEN` |
