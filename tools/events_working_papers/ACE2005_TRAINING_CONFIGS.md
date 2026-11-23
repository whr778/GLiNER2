# ACE 2005 Training Configs: Architecture, Diffs, and Execution Order

**Status:** prepared 2026-11-23 alongside the six configs in `tools/train/config/ace2005/`. Every setting below was verified against the live training/loss/eval code (`tools/train/train.py`, `gliner2/configuration.py`, `gliner2/training/trainer.py`, `gliner2/models/boundary/{model,records}.py`) and traced against `whr778/gliner2-eb23a/config.json`.

**Standing measurement rules:**
- Score the zero-shot `whr778/gliner2-eb23a` control on the identical splits before comparing any fine-tuned arm.
- Pick checkpoints and thresholds on **validation**; score the blind **test** split once per arm.
- Quote the measured noise floor beside every delta: **`±0.02` strict micro-F1** for single runs (**`±0.041`** on relations).
- Record every finished run — including nulls and negatives — in `EXPERIMENT_CATALOG.md`.

---

## 1. Why ACE 2005 Needs Specific Settings on `eb23a`

ACE 2005 (LDC2006T06, English + Arabic + Chinese) was held out of `whr778/gliner2-eb23a` completely. Tracing ACE's ontology through `labels/unified-full-v4.yaml` against `eb23a`'s shipped `default_schema` and prior audits establishes four boundary conditions:

1. **Ontology overlap with `eb23a`:**
   - **Entities (7):** `PER -> Person`, `ORG -> Organization`, `GPE -> GPE`, `LOC -> Location`, `FAC -> Facility`, `VEH -> VEH`, `WEA -> WEA`.
   - **Relations (6):** `ART`, `Gen_Aff`, `ORG_Aff`, `Part_Whole`, `Per_Soc`, `Phys` — **0 of 6** seen in `eb23a`.
   - **Event types (33):** **26 of 33** seen in `eb23a`; **7 unseen** (`Contact.Phone_Write`, `Justice.Arrest_Jail`, `Justice.Convict`, `Justice.Execute`, `Justice.Extradite`, `Justice.Pardon`, `Justice.Release_Parole`).
   - **Event roles (35):** **26 of 35** seen in `eb23a`; **9 unseen** (`Org` plus the 8 ACE temporal subtypes `Time_Within`, `Time_Starting`, `Time_Ending`, `Time_Before`, `Time_After`, `Time_Holds`, `Time_At_Beginning`, `Time_At_End`).
2. **Language & alignment gap (measured 2026-11-18):**
   - `eb23a` trained on **0 Arabic base documents** (step-0 trigger max prob `AR=0.0101` vs `EN=0.9971`, `ZH=0.8224`).
   - Under `WhitespaceTokenSplitter` (post-NFKC): `ace2005_zho` has `5 / 30,661` (`0.02%`) unaligned spans, while `ace2005_ara` has **`6,051 / 26,999` (`22.41%`)** unaligned spans (`SUBTOKEN 4,208`, `RUNON 1,608`, `EXTENDABLE 235`) from Arabic proclitics (`و`/`ف`/`ب`/`ل`/`س`) and bound `PRO` suffixes.
3. **Prior empirical baselines (`EXPERIMENT_CATALOG.md`):**
   - **Rank-16 LoRA (`ace2005-lora-eb23a`, 2026-11-18):** NEGATIVE on Arabic (`event_argument` strict micro-F1 `0.0000`, `event_trigger` `0.0040` `@0.3`). Full-rank adaptation (`encoder_lr: 2.0e-5`, `task_lr: 2.0e-4`) is the active path.
   - **Full-rank `ace2005-eb23a` r2 (2026-11-19, post-`17390bd` with `expand_argument_coref: false`):** lifted blind-test `event_argument` strict micro-F1 from `< 0.01` to **`eng 0.2090`**, **`zho 0.1859`**, **`ara 0.0347`** global (`per_head ara 0.0603`, `--model-schema 0.0532`).
   - **Paired `wikievents-coref` A/B (2026-11-19):** `coref_clusters: ignore` tied `enable` on per-head gold-menu `event_argument` strict micro-F1 (`0.2552` vs `0.2548`, `+0.0004`, inside `±0.02`) while lifting global `@0.7` `event_argument` strict micro-F1 **`+0.0689`** (`0.1863 -> 0.2552`) and doubling `--model-schema` `event_argument` strict micro-F1 **`+0.0631`** (`0.0615 -> 0.1246`).
4. **Small step count on full ACE:** ~1,300 train documents across EN + AR + ZH at effective batch `4 * 4 = 16` is only **~81 steps/epoch (~488 steps over 6 epochs)**, whereas `eb23a`'s inherited anneal/warmup horizons were sized for an 82,930-step from-scratch run.

---

## 2. Shared Correctness Fixes (Present in All Six Configs)

Every config under `tools/train/config/ace2005/` now enforces these non-experimental correctness invariants:

| Setting | Value | Why it is required |
|---|---|---|
| `model.boundary_head.record_coref_link_weight` | `0.0` | `eb23a` inherits `record_coref_link: true` and `record_coref_link_weight: 0.3`. Both `convert_ace2005.py` (default `merge_event_mentions=False`) and `adapt_ace2005_office.py` emit **one trigger per event record**. In `_coref_link_loss` (`records.py:2065-2090`), positives require `owner[i] == owner[j]`, so `pos` is always empty (`0` positives) while `neg` scores every same-type trigger pair (including coreferent `<event>` triggers) plus false triggers as `0.0`. Because `inst_states` is not detached (`records.py:934`), leaving weight `0.3` pushes pure all-negative gradients into trigger states and the encoder while driving `TriggerLink` logits to `-inf`. |
| `training.expand_argument_coref` | `false` | Keeps event role supervision on local annotated argument spans while `MentionLink` still trains on `coref_clusters` (when `coref_clusters: enable`). Expanding LDC `<entity>` chains onto roles inflates ~80% of role targets into distant mentions/pronouns and collapses strict `event_argument` F1 to `< 0.01`. |
| `model.boundary_head.record_argument_coref_copy` / `_boost` | `none` / `0.0` | Overrides `eb23a`'s inherited `canonical` copy and `0.25` boost so decode does not inject unannotated coreferent argument copies against local ACE gold. |
| `training.ignore_missing_entities` + `training.suppress_orphaned_queries` | `true` / `true` | Pairs with `RecordFieldTarget.has_unaligned` (`516a92f`) for event roles: when all Arabic mentions of an entity type in a document fail whitespace alignment, the entity query is suppressed instead of training as a false-negative ABSENT query (`boundary_preprocessing.py:673-677`). |
| `training.negative_labels_per_dim` | `{entities: 7, events: 33}` | Covers ACE's full 7-entity / 33-event ontology (`relations` omitted per the 2026-10-19 `relneg` A/B), matching `--model-schema` inference. Paired with `negative_gold_full_roles: true` and `schema_dropout: false`. |
| `model.boundary_head.report_task_losses` | `true` | Zero gradient effect. Logs per-task span loss contributions (`entities`, `events`, `relations`) alongside `record_field_loss` so loss-balance decisions are measured rather than guessed. |
| `eval.calibration_menu` / `eval.model_schema` / `training.default_schema_mode` | `model_schema` / `true` / `dataset` | Scopes the checkpoint's shipped `default_schema` to ACE's ontology, calibrates global + per-head + decoupled `event_argument` thresholds under `model_schema`, and reports both gold-menu and `--model-schema` blind-test breakdowns by language (`eng`, `ara`, `zho`). |
| `data.office_ace` | `true` | Adapts office ACE `entity_facts`, `relation_facts`, `event_facts`, and `clusters` into GLiNER2 offset records (with `trigger_id`-keyed multi-trigger splitting and NFKC/invisible-char offset rebasing); passes already-GLiNER2-shaped files through unchanged. |

---

## 3. Config-by-Config Reference

All six configs share `model.pretrained: whr778/gliner2-eb23a`, `model.architecture: boundary`, `labels_file: labels/unified-full-v4.yaml`, `data.office_ace: true`, and the three corpora `data/ace2005_{eng,ara,zho}`.

| Config | Role | `coref_clusters` | `relation_loss_weight` | `task_loss_weights` (`scope`) | `soft_iou_anneal` / `consistency_warmup` | `max_neg_queries` | `gold_injection` (`hold_frac`) | `epochs` / `eff_bs` | `metric_for_best` |
|---|---|---|---:|---|---|---:|---|---|---|
| `ace2005-eb23a.yaml` | **Control (full-rank baseline)** | `enable` (default) | `2.0` | `None` (`span`) | `20000` / `2000` (inherited) | `64` | `0.75 -> 0.25` (`0.15`) | `6` / `16` | `eval_overall_strict_head_macro_f1` |
| `ace2005-coref-enable.yaml` | **Arm 1 control** (explicit `enable`) | `enable` | `2.0` | `None` (`span`) | `20000` / `2000` | `64` | `0.75 -> 0.25` (`0.15`) | `6` / `16` | `eval_overall_strict_head_macro_f1` |
| `ace2005-coref-ignore.yaml` | **Arm 1 treatment** (drop `MentionLink` + coref decode) | **`ignore`** | `2.0` | `None` (`span`) | `20000` / `2000` | `64` | `0.75 -> 0.25` (`0.15`) | `6` / `16` | `eval_overall_strict_head_macro_f1` |
| `ace2005-eb23a-schedule.yaml` | **Arm 2 treatment** (warm-start schedule & step budget) | `enable` | `2.0` | `None` (`span`) | **`0` / `0`** | **`128`** | **`0.75 -> 0.0` (`0.10`)** | **`8` / `8`** | `eval_overall_strict_head_macro_f1` |
| `ace2005-eb23a-events.yaml` | **Arm 3 treatment** (event loss balance & selection) | `enable` | **`0.5`** | **`{events: 2.0}` (`all`)** | `20000` / `2000` | `64` | `0.75 -> 0.25` (`0.15`) | `6` / `16` | **`eval_event_argument_strict_micro_f1`** |
| `ace2005-lora-eb23a.yaml` | **Superseded LoRA** (`r=16`, `alpha=32`, LR `1e-4 / 3e-4`) | `enable` | `2.0` | `None` (`span`) | `20000` / `2000` | `64` | `0.75 -> 0.25` (`0.15`) | `8` / `16` | `eval_overall_strict_head_macro_f1` |

### Exact Single-Axis Control-to-Treatment Diffs

Each A/B treatment arm differs from its control on **one experimental axis** (plus `output_dir` and `experiment_name`):

1. **Arm 1 — Coref (`ace2005-coref-enable.yaml` -> `ace2005-coref-ignore.yaml`):**
   - `training.coref_clusters`: `'enable'` -> `'ignore'`
   - **What it tests:** Whether dropping `MentionLink` training (`entity_coref_link_weight: 0.3 -> 0.0`, freezing `mention_link`, dropping coref-only chunks) and disabling inherited `entity_coref_threshold: 0.5` reproduces the `wikievents-coref` `--model-schema` `event_argument` gain (`+0.0631`) on ACE 2005 when `expand_argument_coref: false` is already fixed.
   - *Note:* `ace2005-coref-enable.yaml` is functionally identical to `ace2005-eb23a.yaml` (`coref_clusters` defaults to `"enable"`), so if you already run `ace2005-eb23a.yaml` on the same commit and splits, you do not need to run `ace2005-coref-enable.yaml` separately.

2. **Arm 2 — Warm-Start Schedule (`ace2005-eb23a.yaml` -> `ace2005-eb23a-schedule.yaml`):**
   - `model.boundary_head.soft_iou_anneal_steps`: inherited `20000` -> `0` (immediately annealed to exact boundary supervision, matching where `eb23a` finished rather than holding soft-IoU at ~0.98 across a ~500-step fine-tune).
   - `model.boundary_head.consistency_warmup_steps`: inherited `2000` -> `0` (full consistency weight `1.0` from step 0 rather than staying `< 0.25` throughout the run).
   - `model.boundary_head.max_negative_queries_per_batch`: inherited `64` -> `128` (stops `candidate_pair_loss` from discarding most of the injected event/role negative queries in a 4-doc micro-batch).
   - `training.gold_injection_end`: `0.25` -> `0.0`, `training.gold_injection_hold_frac`: default `0.15` -> `0.10` (warms cold Arabic triggers early, then finishes at realistic inference `injection=0.0`).
   - `training.gradient_accumulation_steps`: `4` -> `2`, `training.num_epochs`: `6` -> `8` (effective batch `8`, lifting optimizer steps from ~488 to ~1,300).
   - **What it tests:** Whether fixing the 83k-step from-scratch schedule mismatch and doubling update frequency lifts strict `event_trigger` and `event_argument` micro-F1 (especially on cold-start Arabic).

3. **Arm 3 — Event Loss Balance & Selection (`ace2005-eb23a.yaml` -> `ace2005-eb23a-events.yaml`):**
   - `model.boundary_head.relation_loss_weight`: `2.0` -> `0.5` (prevents `relation_keep_gold_pairs: true` + `2.0` weight from crowding out event gradients, as seen in `eb22c-relfix-trace`).
   - `model.boundary_head.task_loss_weights`: `None` -> `{events: 2.0}` with `task_loss_weight_scope: all` (reaches all 9 query-typed span losses: start, end, pair, inside, soft_iou, rerank, proposal, abstention, count).
   - `training.metric_for_best`: `eval_overall_strict_head_macro_f1` -> `eval_event_argument_strict_micro_f1` (in 5-head macro, `event_argument` carries only 20% weight against `entity`, `relation`, `event_trigger`, and gold-menu `event_type`).
   - **What it tests:** Whether re-weighting the shared boundary head toward events and selecting directly on `event_argument` improves event/argument F1 without collapsing entities or relations.

4. **Superseded Reference (`ace2005-lora-eb23a.yaml`):**
   - Retained only as a synced, event-safe LoRA reference (fixed `expand_argument_coref: false`, `record_coref_link_weight: 0.0`, `copy: none`, `boost: 0.0`, `entities: 7 / events: 33`, `calibration_menu: model_schema`, `ignore_missing_entities` / `suppress_orphaned_queries`). Do not use as the primary Arabic/multilingual ACE path unless re-testing higher rank or English/Chinese-only adaptation.

---

## 4. Suggested Build & Execution Order

Run in sequence (or Arms 1–3 in parallel once Step 1 establishes the control), holding the commit hash, data splits, and `whr778/gliner2-eb23a` base checkpoint identical across all arms:

### Step 0 — Data Hygiene & Zero-Shot Control (Mandatory Before Training)
```bash
# 1. Verify zero cross-split leakage across train / val / test
uv run python tools/data/check_leakage.py \
  --config tools/train/config/ace2005/ace2005-eb23a.yaml

# 2. Score the zero-shot eb23a control on val (both gold-menu and --model-schema)
uv run python tools/train/eval.py \
  --checkpoint whr778/gliner2-eb23a \
  --config tools/train/config/ace2005/ace2005-eb23a.yaml \
  --split val --model-schema
```
*(Optional diagnostic on licensed ACE splits: run `tools/data/measure_argument_inside_trigger.py data/ace2005_eng data/ace2005_ara data/ace2005_zho` to record ACE's placed-argument inside-trigger share alongside `ARGUMENT_INSIDE_TRIGGER.md`.)*

### Step 1 — Cleaned Full-Rank Control (`ace2005-eb23a.yaml`)
```bash
uv run python tools/train/train.py \
  --config tools/train/config/ace2005/ace2005-eb23a.yaml
```
**What to verify in the log (gates that must fire):**
- `[argument-coref] coref_clusters=enable | expand_argument_coref=False`
- Zero `coref link:` training lines (`record_coref_link_weight=0.0` skips `_coref_link_loss`).
- `report_task_losses: true` prints `entities_*` vs `events_*` span loss shares alongside `record_field_loss` and `relation_loss`.
- Record val & blind-test `event_type`, `event_trigger`, `event_argument`, `entity`, and `relation` strict micro-F1 (overall and per-language `eng`/`ara`/`zho`, both gold-menu and `--model-schema`).

### Step 2 — Arm 1: Coref (`ace2005-coref-ignore.yaml`)
```bash
uv run python tools/train/train.py \
  --config tools/train/config/ace2005/ace2005-coref-ignore.yaml
```
- Compare directly against Step 1 (`ace2005-eb23a.yaml`), or run `ace2005-coref-enable.yaml` beside it if executed on a later commit.
- **Decision rule:** Keep `coref_clusters: ignore` if `--model-schema` or per-head `event_argument` strict micro-F1 beats control by `> +0.02` without hurting `entity`/`relation` beyond the floor (and entity coref output is not needed from this model).

### Step 3 — Arm 2: Warm-Start Schedule (`ace2005-eb23a-schedule.yaml`)
```bash
uv run python tools/train/train.py \
  --config tools/train/config/ace2005/ace2005-eb23a-schedule.yaml
```
- Compare against Step 1 on **validation** first (check whether `event_trigger` and `event_argument` strict micro-F1 rise across `eng`/`ara`/`zho` and whether `ara` escapes the `~0.03–0.06` floor as `gold_injection` anneals to `0.0`).
- **Decision rule:** Keep the schedule axis if val strict `event_argument` or `event_trigger` clears `+0.02` (especially on `ara` and `--model-schema`).

### Step 4 — Arm 3: Event Loss Balance & Selection (`ace2005-eb23a-events.yaml`)
```bash
uv run python tools/train/train.py \
  --config tools/train/config/ace2005/ace2005-eb23a-events.yaml
```
- Compare against Step 1 using the Step 1 `report_task_losses` numbers to interpret whether events were actually starved by `relation_loss_weight: 2.0` and dense entities.
- **Decision rule:** Keep if `event_argument` strict micro-F1 beats control by `> +0.02` while `relation` stays within `±0.041` (or within your acceptable relation trade-off).

### Step 5 — Compose Winners & Score Blind Test Once
- Combine **only** the axes from Steps 2–4 that cleared their validation noise floor into a single composed config (e.g., `ace2005-eb23a-v2.yaml`), validate once, score the blind test once, and log every arm in `tools/events_working_papers/EXPERIMENT_CATALOG.md`.

---

## 5. Known Ceilings Outside Config Tuning

Three structural ceilings remain even after all four steps above:

1. **Arabic subtokenization (`22.41%` unaligned gold):** `WhitespaceTokenSplitter` cannot match spans inside Arabic proclitics (`و`/`ف`/`ب`/`ل`/`س`) or attached pronouns (`ه`/`ها`/`هم`). `suppress_orphaned_queries: true` + `RecordFieldTarget.has_unaligned` prevent those spans from poisoning training as false negatives, but they remain unreachable during training and strict-span eval unless Arabic clitic segmentation is introduced in the tokenizer/splitter layer.
2. **Label-map warm-start gap on `Org` (and `Time_*`):** `eb23a` learned `Organization` on all 8 org-bearing ACE event types (`Business.*_Org`, `Personnel.*`), whereas `labels/unified-full-v4.yaml` maps `ORG -> Organization` only under `entities`, leaving ACE's event role `Org` unmapped (unseen in `eb23a`). `unified-full-v4.yaml` is pinned by `tests/training/test_label_drop.py::test_v4_is_v3_plus_the_pinned_drops_only` (`v3` + relation drops only), so adding `Org: Organization` to `events.map` should be done in a `v5` label file via `tools/train/build_unified_full.py` for the next base run. Map `Time-* -> Time` only if downstream evaluation does not distinguish the 8 ACE temporal role subtypes.
3. **Single-trigger vs coreferent-event records:** `record_coref_link_weight: 0.0` is correct for default per-`<event_mention>` ACE records (OneIE comparability). If a future run switches to `convert_ace2005.py --merge-event-mentions` (multi-trigger event clusters), restore `record_coref_link_weight: 0.3` and evaluate `event_cluster` / `event_cluster_argument` metrics alongside strict per-trigger metrics.
