# Head-Init Data-Scaling Curve (mmBERT) — Experiment Spec (decide before spending)

Status: experiment spec (not yet run). Date: 2026-08-04.
Goal: measure, not guess, how much structure/argument data it takes to warm
**mmBERT-base**'s fresh from-encoder heads into a usable downstream event model —
turning the reasoned bracket in [`HEAD_INIT_DATA_SCALE.md`](HEAD_INIT_DATA_SCALE.md)
into a curve on the encoder we actually want to ship (multilingual, native 8192
long-context; the `mmbert_training` branch target). Companion to
[`FASTINO_GLINER2_TRAINING.md`](FASTINO_GLINER2_TRAINING.md) and PAPER.md §10.6/§10.7.

---

## 1. Bottom line

Train mmBERT-base `from_encoder` on a structure/argument corpus at three sizes
(**~10K / ~40K / ~100-120K**), fine-tune each on RAMS under the fixed
`mmbert-base-rams` recipe, and plot **RAMS argument-strict F1 vs Stage-A corpus
size**. Unlike the DeBERTa curve, mmBERT has **only the lower anchor measured** —
there is no fastino-scale warm-start on mmBERT (fastino is DeBERTa-v3), so the upper
end is what we are trying to discover, and DeBERTa's 254K point is a *cross-encoder
reference*, not an mmBERT endpoint.

| Stage-A size N | Stage-A source | RAMS arg-strict F1 | Status |
|--:|---|--:|---|
| **0** | none (fresh heads → RAMS) | **0.050** | measured — `mmbert-base-rams` (§10.6) |
| ~96K (confounded) | broad NER-heavy, **2 epochs** | 0.028 | measured — combined base (§10.7); see §2 |
| ~10K | structure/argument corpus | ? | to run |
| ~40K | structure/argument corpus | ? | to run |
| ~100-120K | structure/argument corpus | ? | to run |
| *254,334 (DeBERTa)* | *fastino `gliner2-base-v1`* | *0.462* | *cross-encoder reference only* |

The measured mmBERT N=0 is **0.050** (entity 0.964, trigger 0.611 — the encoder and
trigger head are fine; only the **argument** head is at the floor). The question is
whether head-warming lifts that argument floor on mmBERT, and at what N.

**Cheapest informative version: $0 generation + ~$40-60 of A100 compute** by
assembling Stage A from the multilingual event corpora already on disk (Section 3).

## 2. Why mmBERT, and what the combined experiment already told us

- **mmBERT is the production target** (multilingual, native whole-document 8192 — no
  windowing, no global decoder; PAPER §9.5). The DeBERTa curve was the clean
  scientific control (shared encoder with fastino); this is the *applied* question —
  does the head-init recipe transfer to the encoder we want to ship.
- **We already ran a confounded version.** The combined base (PAPER §10.7) trained
  mmBERT `from_encoder` on ~96K records, and its own RAMS argument head came out at
  **0.028 — below the 0.050 RAMS-only floor.** But that point is confounded three
  ways: only **2 epochs** (vs 15), an **NER-heavy** corpus (GLiNER multilingual +
  multi-task NER dominate — *not* structure/argument-dense), and `eval_loss`
  checkpoint selection. This curve removes all three: structure/argument-dense
  corpus, proper epochs, argument-strict selection, nested sizes.
- **Prediction: the knee sits higher than DeBERTa's.** A cold/multilingual encoder
  needs more head-warming data ([`HEAD_INIT_DATA_SCALE.md`](HEAD_INIT_DATA_SCALE.md)
  §3.3), so where DeBERTa's knee was predicted ~40-60K, mmBERT's is predicted
  **~60-100K+** — and the confounded 0.028 point is a hint the argument head may be
  genuinely harder to warm on mmBERT than on DeBERTa. That is exactly what the curve
  resolves.

## 3. Stage-A corpus sourcing (the real decision)

Corpus must be **structure/argument-dense** (trigger→argument span-attribute shape),
not NER-heavy — it has to exercise the span + count + occurrence-ID heads
([`HEAD_INIT_DATA_SCALE.md`](HEAD_INIT_DATA_SCALE.md) §4). **RAMS and WikiEvents are
excluded from Stage A** (downstream targets — including them leaks). Sizes are
**nested** (10K ⊂ 40K ⊂ 120K) so the curve isolates scale, not composition.

### Option B — assemble from on-disk event corpora ($0 generation) **[recommended]**
mmBERT is **multilingual**, so — unlike the DeBERTa curve — the Chinese event sets on
disk are usable, and the on-disk pool reaches the top point for free:

| Corpus | Train records | Lang | Type |
|---|--:|---|---|
| chfinann | 25,632 | zh | financial events |
| docee | 21,966 | en | document-level events |
| docfee | 16,420 | zh | financial events |
| duee | 11,603 | zh | events |
| cmnee | 9,284 | zh | events |
| text2json | 7,817 | en | structured extraction |
| maven | 2,913 | en | event detection |
| events_biotech | 2,216 | en | events |
| mendeley_ed | 1,431 | en | event detection |
| casie | 798 | en | cyber events |
| **total** | **~100K** | mixed | reaches all three points |

Pro: near-zero cost, immediate, and multilingual coverage suits mmBERT. Con:
composition shifts across nested cuts unless the mix ratio is held fixed when
subsampling (sample each corpus proportionally at 10K and 40K, not head-of-file).

### Option A — generate homogeneous synthetic (cleaner control, costs money)
Our `synthetic_sonnet5` shape is right but only ~1.5K exists. Generate a 120K pool,
nested-subsample (cost is the 120K top point only; `tools/data/synthetic/COST_BREAKDOWN.md`, batch):

| Teacher | $/1k (batch) | 120K pool |
|---|--:|--:|
| gpt-4.1-nano | $0.58 | **~$70** |
| gpt-4o | $14.50 | ~$1,740 |
| claude-sonnet-5 | $14.10 | ~$1,690 |

A cheaper teacher shifts the whole curve down (ceiling caveat) — read knee *location*,
not absolute F1, on a nano-labeled curve.

**Recommendation:** run Option B first (proportional multilingual mix, $0 generation).
If the knee is ambiguous or composition drift muddies it, spend ~$70 on a nano 120K
pool (Option A) for a clean homogeneous rerun.

## 4. Configs

Two configs per point (N ∈ {10K, 40K, 120K}); the N=0 endpoint reuses `mmbert-base-rams.yaml`.

**Stage A — `scaling-mmbert-N.yaml`** (clone of `mmbert-base-combined.yaml`, fixing its confounds):
- `model.encoder: jhu-clsp/mmBERT-base`, `max_len 8192`, `struct_loss bce_posweight`,
  `struct_pos_weight 4.0` (mirror `mmbert-base-rams`).
- `from_encoder`; `encoder_lr 2e-5 / task_lr 5e-4`; `sliding_window false`,
  training `max_len 2048` (holds any RAMS/event doc + schema; native long context,
  **no global_decode**); `bf16`; `gradient_checkpointing true`.
- **Memory:** mmBERT-base at 2048 OOM'd the 40GB A100 at batch 8 → `batch_size 4`,
  `gradient_accumulation_steps 8` (eff batch 32). **A100, not A10.**
- `num_epochs: 5` (fastino-like; the combined base's 2 was too few — a confound to
  fix); raise toward ~8-10 for the 10K point so small-N gets enough passes.
- `metric_for_best eval_loss` (multi-task base), `early_stopping false`.
- `data.corpora`: the size-N list (Option B event corpora truncated to N, or Option A
  synthetic dir); `event_files: {}` (RAMS/WikiEvents excluded — leakage).

**Stage B — `scaling-mmbert-N-rams.yaml`** (clone of `mmbert-base-rams.yaml`, the recipe behind the 0.050 endpoint):
- swap `model.encoder` → `model.pretrained: ./out/scaling-mmbert-N/best` (warm-start
  the Stage-A heads instead of fresh — the same `from_pretrained`-on-local-checkpoint
  pattern the combined WikiEvents fine-tune used).
- Everything else identical to `mmbert-base-rams`: `encoder_lr 2e-5 / task_lr 5e-4`
  (or lower encoder_lr 1e-5 for warm-start), `num_epochs 15`,
  `metric_for_best eval_event_argument_strict_micro_f1`, `metric_sweep true`,
  `sliding_window false`, `max_len 2048`, no `global_decode`, `event_files.rams`.
- `output_dir: ./out/scaling-mmbert-N-rams`; read
  `test_metrics.json → eval_event_argument_strict_micro_f1`.

N=0 needs no new run: `mmbert-base-rams.yaml` = 0.050.

## 5. Cost + compute

| Item | Cost |
|---|--:|
| Generation — Option B (assemble on-disk, multilingual) | **$0** |
| Generation — Option A nano 120K (if needed) | ~$70 |
| Compute — 3 points × (Stage A + Stage B), **A100 @ $1.99/hr** | **~$40-60** (~20-30 GPU-hr) |

mmBERT-base (336M, heavy attention at 2048) needs the A100 the DeBERTa curve did not:
the combined base (~96K × 2 epochs) was ~5h on an A100. Stage A at 120K × 5 epochs ≈
the long pole (~12-15h); 40K/10K are shorter; each Stage-B RAMS fine-tune ~1-2h. On
one A100, the whole curve is ~a day wall-clock; on 2-4 A100s, a few hours.
**Recommended path (Option B + compute): ~$40-60, no generation spend.**

## 6. Decision rule (what the curve buys)

- **Argument F1 climbs clearly above 0.050 with N, knee ~60-100K:** head-warming
  transfers to mmBERT — build the multilingual head-init curriculum at the knee size,
  then fine-tune. This is the win condition for a shippable multilingual event model.
- **Flat at ~0.05 across all N (like the confounded 0.028 point, now deconfounded):**
  mmBERT's fresh argument head does **not** warm from this data scale — either it needs
  fastino's 254K+ on mmBERT (expensive; would need its own upper-anchor run) or the
  DeBERTa-v3 warm-start path (fastino) is the pragmatic route and mmBERT-from-encoder
  is a dead end for arguments. Either way, a decisive negative before big spend.
- **Climbing but still low at 120K:** knee is beyond 120K — budget toward 254K-scale
  multilingual generation, or reconsider the encoder.

Note: without a measured mmBERT upper anchor, the curve reads **shape** (does warming
help, and where does it knee), not absolute ceiling. If shape is promising, a single
fastino-scale mmBERT run would fix the top of the curve — but that is a separate,
larger spend, deliberately out of scope for this "decide before spending" probe.

## 7. Related
- [`HEAD_INIT_DATA_SCALE.md`](HEAD_INIT_DATA_SCALE.md) — the estimate this measures (mmBERT knee predicted higher, §3.3).
- [`FASTINO_GLINER2_TRAINING.md`](FASTINO_GLINER2_TRAINING.md) — the DeBERTa 254K reference recipe.
- PAPER.md §10.6 (mmBERT N=0 = 0.050; DeBERTa reference band 0.042→0.462), §10.7 (the confounded ~96K combined point = 0.028).
- Configs to clone: `mmbert-base-combined.yaml` (Stage A), `mmbert-base-rams.yaml` (Stage B + N=0 endpoint).
- Cost basis: `tools/data/synthetic/COST_BREAKDOWN.md`.
