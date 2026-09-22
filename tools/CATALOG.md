# The tools catalog

Every executable under `tools/` — **239 scripts** — and what each one is for. Compiled
2026-09-22 from the tools' own docstrings.

**How to read it.** The tools divide into two kinds, and the division matters more than the
directory they live in:

- **Instruments** MEASURE something and are allowed to refuse. They exist because this
  programme's expensive mistakes have almost all been *silent* — supervision that trained
  nothing, a gate that could not fail, a delta inside a noise floor nobody quoted.
  Section 1 lists them by the question they answer, not by filename.
- **Production tools** build, convert, repair, publish or launch. Sections 2–7.

Everything runs through `uv`:

```bash
uv run python tools/<dir>/<tool>.py --help
```

**Two standing rules this catalog exists to serve.** A gate must be able to FAIL — before
trusting any check, ask what it would print if the thing it guards were broken. And a delta
is not a result without a noise floor beside it. Measured floors:
`event_argument` seed sd **0.0009–0.0020**, `entity` **0.0139**, relations **±0.041**.

| directory | scripts | what lives there |
|---|---:|---|
| [`tools/data`](#2-toolsdata--corpus-construction) | 98 | converters, annotators, repairs, label space, Hub mirrors |
| [`tools/ekf_showcase`](#5-toolsekf_showcase--textnormaltracking) | 55 | the disaster-tracking research line end to end |
| [`tools/train`](#3-toolstrain--training-probing-scoring) | 45 | training, probes, sweeps, scoring, model cards |
| [`tools/lambda`](#4-toolslambda--gpu-runs-that-stop-themselves) | 24 | GPU runs that provision, publish and terminate |
| [`tools/data/synthetic`](#21-toolsdatasynthetic) | 8 | LLM generation of base-training data |
| [`tools/prototypes`](#6-toolsprototypes--architecture-spikes) | 4 | architecture spikes on toy harnesses |
| [`tools`](#7-tools-root) | 3 | inference CLI and import-surface diffing |

---

## 1. Instruments — the things that measure

### 1a. Does the gold actually reach the model?

| instrument | the question, and what it found |
|---|---|
| `data/measure_surface_alignment.py` | **How much gold never aligns, and why.** A mention is supervision only if its surface aligns to the TOKENIZED text; a miss decodes to `(-1,-1)` and is skipped with no error. `surface in text` is the WRONG test — on docee it says 100.00% where the real path says 99.19%. Mix-wide **1.34%** never aligns; worst `bio_ner_relations` **4.77%**. Splits the loss by what would fix it: EXTENDABLE (repair), SUBTOKEN (**never** repair — changes the referent), RUNON, TOKENIZATION, ABSTRACTIVE. `--fail-over` gates a launch. |
| `train/probe_absent_queries.py` | **How many queries carry no gold** — the positive class of `abstention_loss`. Its headline `0` was once a hardcoded print; it now measures from `targets.mention_mask` and reads **0.19%** (2 of 1,079), concentrated in chfinann. |
| `data/audit_corpora.py` | Structural validity over every corpus in `data/`. STRUCT flags any structure with no `record_metadata` — which is silently undecodable on the boundary path. |
| `data/check_leakage.py` | Does any corpus share input text with another, or across its own splits? The standing contamination gate. |
| `data/event_multiplicity.py` | Prices event-instance multiplicity against the KEY a decoder uses to address instances — the quantitative case for `event_records`. |
| `train/size_gold_capacity.py` | Sizes `max_gold_per_query` from a config's OWN corpora, rather than a default that silently truncates. |

### 1b. Will this config train at all? (pre-launch gates)

| instrument | what it refuses |
|---|---|
| `train/check_corpora_fetchable.py` | A box whose config names a corpus the box cannot fetch. |
| `train/check_label_menus.py` | A config whose classification MENUS disagree between corpora. Resolves files through the trainer's own `_split_files`/`_event_split`, so `event_files`-only corpora are scanned. **Caveat printed by the tool:** it reads the LOCAL tree, and a fresh box fetches from HF — eb17-best trained docee at 59 labels against 60 elsewhere while this gate read 60/60/60 on disk. |
| `train/check_hf_write.py` | A run that would spend hours earning something it cannot push. Only a real upload proves write access. |
| `data/check_augment_alignment.py` | An augmentation that changed the gold. |
| `data/check_gate_corpus.py` | A gate corpus separable WITHOUT reading the words. |

### 1c. What is the model actually doing?

| probe | the question |
|---|---|
| `train/probe_records.py` | Instance separation, value binding, and field FILL, scored separately. |
| `train/probe_event_multiinstance.py` | Does this checkpoint emit more than one event instance of the same type per document? |
| `train/probe_event_type_fp.py` | Event-type FALSE POSITIVES, which the blind test cannot see. |
| `train/probe_argument_recall.py` | Is low argument recall undertraining, or a missing mechanism? |
| `train/probe_beam_disagreement.py` | Is the joint beam INERT, or a coin flip? |
| `train/probe_task_losses.py` | Where the boundary loss actually goes, per task. |
| `train/probe_query_negatives.py` | Is there a query-axis hard negative to mine, or would a loss have nothing to bite on? |
| `train/probe_predicted_entity_types.py` | Does the model's OWN predicted type carry signal about argument correctness? |
| `train/cuda_attn_probe.py` | What attention path actually loads on CUDA, and does it produce NaN. |
| `train/debug_shared_pool_nan.py` | Localizes the non-finite that kills `candidate_pool: shared`. |

### 1d. Measure the ceiling BEFORE spending on the treatment

| instrument | what it prices |
|---|---|
| `train/measure_margin_ceiling.py` | How much the typed margin could move at all. Refuted the Option-2 arm before it was bought. |
| `train/measure_logit_scale.py` | Per-corpus listwise logit scale, and the typed margin's `k`. |
| `train/measure_corpus_familiarity.py` | How well the BASE encoder already knows each corpus's (language, domain) cell. |
| `train/base_model_perplexity.py` | Per-language pseudo-perplexity, before committing supervised data. |
| `train/score_gist_guides.py` | Can any available guide tell "33 killed" from "10,000 evacuated"? |
| `ekf_showcase/span_giou_headroom.py` | Does a GIoU-shaped span target have anything to bite on? (Measured: essentially not.) |
| `ekf_showcase/reject_headroom.py` | Can a LEARNED reject beat the ratio gate? Separability first, build second. |
| `data/gate_purity_curve.py` | What purity the gate can deliver on a pool, and what it costs. |

### 1e. Is the delta real?

| instrument | what it enforces |
|---|---|
| `train/compare_runs.py` | Two runs' test metrics **with the noise floor made explicit**. |
| `train/compare_capabilities.py` | Per-capability diff between two blind-test metric files. |
| `train/sweep_record_thresholds.py` | The record head across its own decode thresholds. |
| `train/sweep_preservation.py` | Several checkpoints on ONE held-out set, each at its own best threshold. |
| `train/sweep_entity_typed_arguments.py` | Can predicted entity types improve argument precision? |
| `data/compare_label_distributions.py` | Generated corpora against a real-text reference. |

---

## 2. `tools/data` — corpus construction

**36 converters** (`convert_*.py`), one per source corpus: ace2005, bio_ner_relations,
biomed_ner, biored, casie, chfinann, cmnee, docee, docfee, docred, duee, events_biotech,
finer_ord, gliclass_logic, gliner_multilingual, hf_token_ner, klue, knowledgator_gliner,
masakhaner, masakhanews, maven, mendeley_ed, mtl_bio, nuner, paraloq_json,
pile_ner_definition, professorbob_re, pubmed_abstracts_ner, rams, redocred,
scientific_text, scierc, sentence_rex, stockmark_ner, text2json, wikievents.
`run_all_converters.sh` runs every one except ACE 2005 (licence).

> **Converter rule.** Every record write goes through `_split.dumps_record` — NFKC plus
> stray line-separator stripping — never raw `json.dumps`. Shared helpers: `_split.py`
> (partitioning), `_stratify.py` (greedy multi-label stratified split), `_mention_filter.py`.

**Builders** — `build_gate_corpus.py` (all text real), `build_negative_pools.py` (the labels
a document must learn to reject), `build_role_type_map.py`, `build_turkish_pool.py` /
`_candidates` / `_eval`, `build_chinese_candidates.py`, `build_zh_general_pool.py`,
`build_english_casualty_candidates.py`, `build_duplicate_control.py`.

**Annotators (paid, Haiku)** — `annotate_casualty.py`, `annotate_gate.py`,
`annotate_event_type.py`, `annotate_event_entities.py`, `annotate_entities_zh.py`,
`annotate_long_news.py`, `annotate_multitask.py`.

**Repairs** — `repair_turkish_surfaces.py` (lemmatised gold, **refusing unsafe repairs**),
`repair_casualty_anchors.py`, `repair_contradicted_negatives.py`, `dedupe_splits.py`,
`stamp_record_metadata.py`, `stamp_field_cardinality.py`, `interleave_splits.py`.

**Label space** — `build_label_maps.py` (in `train/`), `apply_label_map.py`,
`translate_labels.py`, `unify_docee_menus.py`, `unify_classification_labels.py`,
`split_withdrawal_collision.py`.

**Hub mirrors** — `restore_from_hf.py` rebuilds `data/` on a fresh box; `push_corpus.py`,
`push_corpus_hf.py`, `push_dir_hf.py`, `push_repaired_corpora.py`, `push_scaling_slices.py`,
`publish_when_ready.sh`. All repos PRIVATE.

### 2.1 `tools/data/synthetic`

`generate.py`, `prompts.py`, `providers.py`, `schema_spec.py`, `validate.py` (strict record
construction), `cost.py` (token-and-price model), `ab_generation.sh`.

---

## 3. `tools/train` — training, probing, scoring

`train.py` (1,670 lines) is the entry point. Around it: `eval.py`, `finalize_run.py`
(re-runs the post-training tail), `save_or_die.py` (gets a checkpoint off the machine or
refuses to let it die quietly), `push_to_hub.py`, `model_card.py`, `backfill_schema.py`,
`refresh_structure_metrics.py`, `update_dose_cards.py`, `precompute_guide_scores.py`,
`make_nan_probe_configs.py`.

**Mix builders** — `build_137k_replay.py`, `build_scaling_mix.py`,
`build_joint_scaling_mix.py`, `build_warmstart_mix.py`, `build_turkish_dose_mix.py`,
`build_zh_multitask_mix.py`, `build_casualty_multilingual.py`, `build_loc_control.py`.

Probes, sweeps, gates and measures are in [section 1](#1-instruments--the-things-that-measure).

---

## 4. `tools/lambda` — GPU runs that stop themselves

> **Every job carries three independent stops**: a `timeout` on the job
> (`JOB_TIMEOUT`, which stops `train.py` and still lets the runner publish), a detached
> hard-deadline watchdog (`HARD_DEADLINE`), and terminate on the normal path.
> `idle_guard.sh` is stop 4: the box came up and NO JOB EVER STARTED.

**Plumbing** — `launch_when_available.sh` (polls for capacity, then provisions;
**pin the card**, or an A/B silently lands on two different GPUs), `provision.sh`,
`provision_box.sh`, `bootstrap_box.sh` (proves each step), `box_run.sh`, `_publish.sh`
(retry 6× and verify against the Hub's own file list — a clean return is not proof).

**Experiment runners** — `event_base_run.sh`, `absneg_ab_launch.sh`, `roles_ab_launch.sh`,
`pool_ab.sh`, `cardinality_ab.sh`, `decode_arms.sh`, `negatives_ab.sh`,
`negatives_verdict.sh`, `dose_sweep.sh`, `dose_curve_box.sh`,
`event_threshold_sweep.sh`, `base_reference.sh`, `rescore_blind_test.sh`,
`throughput_smoke.sh`, `pool_nan_debug.sh`, `pool_fp32_probe.sh`, `score_pool.py`.

---

## 5. `tools/ekf_showcase` — text→tracking

`run_pipeline.py` (1,301 lines) is the whole research line in one command: news feed →
EKF-tracked casualty timeline. `README.md` there is the entry point.

**Feeds and ground truth** — `build_helene_feed.py`, `build_turkey_feed{,_tr,_zh}.py`,
`build_aegean_feed.py`, `harvest_helene_gt.py`, `harvest_turkey_gt.py`,
`harvest_aegean_gt.py`, `make_demo_feed.py`, `make_docee_feed.py`,
`build_real_comparison.py`, `turkey_truth_jsonl.py`.

**Gates** — `scope_gate.py` (688 lines: is this figure the place's own toll, the national
total, or another event's?), `scope_gate_test.py`, `gate_threshold_sweep.py`,
`gate_perclass.py` (scored PAIRED), `language_gate.py`, `gate_turkish_fp.py`,
`gate_turkish_heldout.py` (did it learn Turkish or TRT Haber's register?),
`gate_translation_ablation.py`, `gate56_composition.py`, `two_sided_gate_sweep.py`,
`viterbi_gate_sweep.py`, `benchmark_gate.py`, `frontend_gates.py`.

**Filter/association research** — `mht_associate.py`, `imm_gate_sweep.py`,
`robust_filter_sweep.py`, `vector_state_test.py`, `revision_test.py`,
`revision_state_test.py`, `hmm_collapse_test.py`, `aegean_collapse_test.py`,
`turkey_collapse_check.py`, `stream_ceiling.py` (method error vs source coverage).

**Extraction probes** — `binding_accuracy.py`, `event_binding_probe.py`,
`extractor_language_probe.py`, `false_positive_rate.py`, `energy_probe.py`,
`framing_experiment.py`, `bullet_premise_test.py`, `record_threshold_probe.py`,
`record_threshold_sweep.py`, `scope_label_probe.py`, `stage1_transfer_probe.py`,
`turkish_dose_probe.py`, `spatial_anchor.py`.

**Scoring** — `score_helene.py`, `score_turkey.py`, `rescore_helene.py`,
`rescore_recorded_run.py`.

---

## 6. `tools/prototypes` — architecture spikes

`ffn_variants.py` (plain GELU, GeGLU, fixed partial gating), `activation_variants.py`,
`lr_ladder.py` (does partial gating survive an LR that destabilises GeGLU?),
`divergence.py` (classify an MLM loss trace as trained / diverged / flat).

---

## 7. `tools` root

`infer.py` — command-line inference, with optional document-level global decoding.
`import_surface.py` / `compare_surface.py` — snapshot and diff a package's public import
surface, for verifying a refactor removed only what it meant to.

---

## Where the findings live

This catalog says what each tool *is*. What each one *measured* is in
[`events_working_papers/EXPERIMENT_CATALOG.md`](events_working_papers/EXPERIMENT_CATALOG.md)
(chronological, newest first, including the nulls and negatives — they are the majority),
with open work in [`events_working_papers/TODO.md`](events_working_papers/TODO.md) and the
training recipe in [`train/TRAINING.md`](train/TRAINING.md).
