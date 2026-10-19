# The tools catalog

Every executable under `tools/` — **301 scripts** — and what each one is for. Compiled
2026-09-22 from the tools' own docstrings; the 49 scripts added since were catalogued
2026-10-05, including the new `tools/derive`, and 24 more (missed or new) on 2026-10-14, and 18 more on 2026-10-17 (coreference, menu poison, common menus).

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
| [`tools/data`](#2-toolsdata--corpus-construction) | 108 | converters, annotators, repairs, label space, Hub mirrors |
| [`tools/train`](#3-toolstrain--training-probing-scoring) | 73 | training, probes, sweeps, scoring, model cards |
| [`tools/ekf_showcase`](#5-toolsekf_showcase--textnormaltracking) | 55 | the disaster-tracking research line end to end |
| [`tools/lambda`](#4-toolslambda--gpu-runs-that-stop-themselves) | 42 | GPU runs that provision, publish and terminate |
| [`tools/derive`](#3a-toolsderive--derive-a-fine-tuning-config-from-measurement) | 8 | measure a corpus against a base and write the config |
| [`tools/data/synthetic`](#21-toolsdatasynthetic) | 7 | LLM generation of base-training data |
| [`tools/prototypes`](#6-toolsprototypes--architecture-spikes) | 4 | architecture spikes on toy harnesses |
| [`tools`](#7-tools-root) | 3 | inference CLI and import-surface diffing |

Counted at each directory's top level (`tools/train/pr_sweep_results/pr_sweep.py` is in
`tools/train`); together they make the 300.

---

## 1. Instruments — the things that measure

### 1a. Does the gold actually reach the model?

**`tools/train/probe_anchor_gate.py`** — the share of gold records that train NOTHING because the model never proposed their anchor. `compute_group_loss` resolves the gold anchor against the model's OWN candidate spans and skips the record on a miss; events anchor on the TRIGGER, so a missed trigger discards that instance's entire argument supervision. Refuses to report if the gate was never reached.

| instrument | the question, and what it found |
|---|---|
| `data/measure_surface_alignment.py` | **How much gold never aligns, and why.** A mention is supervision only if its surface aligns to the TOKENIZED text; a miss decodes to `(-1,-1)` and is skipped with no error. `surface in text` is the WRONG test — on docee it says 100.00% where the real path says 99.19%. Mix-wide **1.34%** never aligns; worst `bio_ner_relations` **4.77%**. Splits the loss by what would fix it: EXTENDABLE (repair), SUBTOKEN (**never** repair — changes the referent), RUNON, TOKENIZATION, ABSTRACTIVE. `--fail-over` gates a launch. |
| `train/probe_absent_queries.py` | **How many queries carry no gold** — the positive class of `abstention_loss`. Its headline `0` was once a hardcoded print; it now measures from `targets.mention_mask` and reads **0.19%** (2 of 1,079), concentrated in chfinann. |
| `data/audit_corpora.py` | Structural validity over every corpus in `data/`. STRUCT flags any structure with no `record_metadata` — which is silently undecodable on the boundary path. |
| `data/check_leakage.py` | Does any corpus share input text with another, or across its own splits? The standing contamination gate. |
| `data/near_dup_eval.py` | **Near-duplicates exact-match dedup misses**: eval docs whose 8-word shingles sit >= `--threshold` (0.2) inside one training doc or an earlier eval split. Found WikiEvents test #10 92% inside a train doc (exact dedup: 0). False-fires on shared footers (DocEE's ABC footer). `dedupe_splits.py --near-dup` acts on it. |
| `data/event_multiplicity.py` | Prices event-instance multiplicity against the KEY a decoder uses to address instances — the quantitative case for `event_records`. |
| `train/size_gold_capacity.py` | Sizes `max_gold_per_query` from a config's OWN corpora, rather than a default that silently truncates. |
| `train/trace_sample_life.py` | **The life of one training sample**: every stage on the REAL pipeline, printed, ending in its gradient. The trace behind `events_working_papers/LIFE_OF_A_SAMPLE.md`. |
| `train/measure_record_supervision.py` | How many trigger candidates the record head actually SUPERVISES, counted on real training batches. |
| `train/trace_anchor_raise.py` | Reproduces a record-spec raise on the REAL data path with no GPU and no training (eb17-best and its smoke both died in a DataLoader worker). |
| `data/measure_event_normalisation.py` | Is the event gold trigger-keyed or event-keyed? The two measurements that price design C (normalised events). |
| `data/trace_absent_event_negatives.py` | Traces absent-event negatives exactly as training injects them and tests whether "absent" is true; `data/adjudicate_absent_events.py` has Claude adjudicate a RANDOM sample of them. |
| `data/measure_target_structure.py` | **What does each corpus ask the model to learn?** Per event and relation corpus in a config, through the real collator: argument inflation (each occurrence a separate required list-role value), same-type pairing load and shared arguments, relation pair inflation `all` vs `nearest`, pairing ambiguity, null-like labels. Prints the TRAINING_DATA.md section "Target structure of event and relation corpora". Complements `measure_occurrence_ambiguity.py` (raw placements) with the collated targets. |
| `data/measure_occurrence_ambiguity.py` | Per event corpus: how many places training marks each gold trigger/argument string (every occurrence, `latent_all`), events per document and document length -- why whole-document event loss falls more slowly (`EVENT_ARGUMENT_DIAGNOSIS.md` s4l, TODO #26). |
| `data/measure_casualty_gold_coverage.py` | Per paragraph of casualty_docee: covered by gold, and if not, why (collision vs unique toll). `data/trace_casualty_gold_gap.py` traces each paragraph to its source snippet's ground truth -- the root cause that `rebuild_casualty_docee.py` fixed. |

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
| `train/probe_trigger_misses.py` | **Where do gold triggers go missing?** Every gold trigger split into found / below the gate / wrong type / never decoded, per corpus and language, on train.py's own validation list. |
| `train/trace_argument_evidence.py` | Do a trigger's ARGUMENTS vouch for it? Argument evidence of gold vs false trigger candidates ("the King of Sweden"). Measured ~0.5 AUC on eb18, which put the existence head on hold. |
| `train/measure_assign_decomposition.py` | Is trigger -> argument assignment a JUNCTION or ROLE FIT? Splits the assign logit into P (pair) + R (role); per-column and per-row AUCs. Found role fit with a weak join (per-column 0.631 on eb18), which motivated the junction layer. |
| `train/measure_junction_det.py` | **Is the junction overfitting?** Ownership (per-column AUC, top-1) and the DET of the argument gate (scorer and decode views), TRAIN vs VAL per corpus family, on any checkpoint. See `METRICS.md`, "Trigger -> argument link diagnostics". |
| `train/measure_argument_key_gap.py` | Where does argument recall live? The same predictions scored WITH and WITHOUT the trigger key (`--classify` splits wrong-trigger arguments into boundary / sibling / false trigger). |
| `train/probe_candidate_coverage.py` | Why emission stops scaling with the window: gold COVERAGE by candidates under `candidate_budget`. |
| `train/probe_object_score_separation.py` | Does the object score already separate good instances from spurious ones? `train/reconcile_trigger_ratio.py` explains why this probe and the blind test disagree (0.96:1 vs 3.78:1). |
| `train/reliability_curve.py` | Is the model under-confident, or is a low threshold genuinely trading precision away? The question a threshold sweep cannot answer. |
| `train/measure_gradient_shares.py` | Which loss terms drive the update: gradient share per term on real training micro-batches. |
| `train/measure_absent_dilution.py` | How much ABSENT queries dilute the boundary loss, on real training batches with no model. Its `load_records(split=)` is the shared sampler the other measure tools use. |
| `train/measure_pos_neg_ratio.py` | The per-task negative:positive ratio the boundary BCE actually sees, before anyone sets `pos_weight`. |
| `train/eval_decode_pooling.py` | Decode-time pooling of same-type instances on boundary-variant triggers, rescored. Tested 2026-10-04: NEGATIVE (F1 -0.001 to -0.004). |
| `train/probe_label_order.py` | **Does the model read LABELS or their POSITION?** Same model, records and widened menu; only one dimension's label order changes (`--dimension entities|events|relations`). Found the entity position leak: 0.000 absent-first vs 0.646 gold-first (MENU_SPEC s9). |
| `train/measure_coref_separability.py` | Baseline for coreference: do trigger-instance states separate coreferent mention pairs from DIFFERENT same-type events? |
| `train/overfit_relations.py` | **Can the relation head fit a FEW training records?** Trains a checkpoint on N relation records of one corpus through the real dataset/negatives/collator/loss and decodes them every K steps; logs gold vs positive-labelled pairs and pos/neg logits. `--relation-gold-pairs`, `--labels-file`, `--relation-biaffine` (diagnostic: zero-init biaffine scorer on a warm checkpoint). Found eb21's relation head could not fit inflated labels; nearest + v4 + biaffine fixed it (TODO #40). Run on CUDA with `lambda/overfit_box.sh`. |
| `train/overfit_events.py` | The same overfit test for the EVENT record head, scored with the in-run `compute_metrics` (gold menu, global decode) on the same records. `lambda/overfit_box.sh SCRIPT=train/overfit_events.py`. |
| `train/menu_poison_check.py` | **Which menu labels POISON the rest?** Runs a checkpoint on VAL under the gold menu, the full menu, and the full menu minus each event role / type (`--entities`: each entity label), counting events fired AND exact type+trigger hits; suspects are confirmed greedily so side effects drop out. Found CASIE's `Damage_Amount` (menu 0 -> 32 hits vs gold 27). Examples: `train/INFER.md` "Common menus and poison labels". |
| `train/probe_poison_mechanism.py` | **Where does a poison label act?** Trigger max p per event type under menu variants (role removed / original / one role added), and `--swap`: text states of one arm with query states of the other. Found the trigger QUERY of the listing type, not the text. |
| `train/probe_menu_axes.py` | Which menu AXIS silences events: per-doc menus varying one axis at a time (extra types, extra entities, role order, extra roles, one role). The probe that isolated `Damage_Amount`. |
| `train/coref_readout.py` | Entity-coreference readout: pair AUC of a checkpoint's mention link on VAL docs, beside string match and cosine, per corpus (pronoun pairs separately). |
| `train/measure_link_by_task.py` | Which TASKS does the junction's column loss train? Counted per task type on real training batches. |

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
| `train/coref_auc_floor.py` | Document-bootstrap noise floor for `coref_readout`'s pair AUC, PAIRED across arms (same gold mentions, same pairs); CIs for named contrasts. corefC's news gain +0.127 [+0.088, +0.179]. |
| `train/compare_capabilities.py` | Per-capability diff between two blind-test metric files. |
| `train/sweep_record_thresholds.py` | The record head across its own decode thresholds. |
| `train/sweep_preservation.py` | Several checkpoints on ONE held-out set, each at its own best threshold. |
| `train/sweep_entity_typed_arguments.py` | Can predicted entity types improve argument precision? |
| `data/compare_label_distributions.py` | Generated corpora against a real-text reference. |
| `data/compare_annotators.py` | Agreement between two annotators (or two runs) on the SAME documents -- the annotation noise floor -- and the real price of one batch. |
| `train/sweep_menu_threshold.py` | A checkpoint's decision threshold picked under an application MENU on val, the blind test scored ONCE; `--threshold T --test-menus ...` scores a menu-size ladder. |
| `train/sweep_record_anchor_threshold.py` | Sweeps the RECORD decode gates on VALIDATION; records decode at their own gates, not the span threshold. |
| `train/pr_sweep_results/pr_sweep.py` | Precision / recall over a threshold grid for eb17-best: pooled, CASIE val and CMNEE val separately. Results beside it. |

---

## 2. `tools/data` — corpus construction

**38 converters** (`convert_*.py`), one per source corpus: ace2005, bio_ner_relations,
biomed_ner, biored, casie, chfinann, cmnee, corefud (any CorefUD CoNLL-U corpus, e.g. GUM, LitBank), docee, docfee, docred, duee, events_biotech,
finer_ord, gap (GAP pronoun coreference), gliclass_logic, gliner_multilingual, hf_token_ner, klue, knowledgator_gliner,
masakhaner, masakhanews, maven, mendeley_ed, mtl_bio, nuner, paraloq_json,
pile_ner_definition, professorbob_re, pubmed_abstracts_ner, rams, redocred,
scientific_text, scierc, sentence_rex, stockmark_ner, text2json, wikievents.
`run_all_converters.sh` runs every one except ACE 2005 (licence).

> **Converter rule.** Every record write goes through `_split.dumps_record` — NFKC plus
> stray line-separator stripping — never raw `json.dumps`. Shared helpers: `_split.py`
> (partitioning), `_stratify.py` (greedy multi-label stratified split), `_mention_filter.py`.

**Builders** — `build_gate_corpus.py` (all text real), `build_negative_pools.py` (the labels
a document must learn to reject), `build_role_type_map.py`, `build_turkish_pool.py`,
`build_turkish_candidates.py`, `build_turkish_eval.py`, `build_chinese_candidates.py`, `build_zh_general_pool.py`,
`build_english_casualty_candidates.py`, `build_duplicate_control.py`.

**Annotators (paid, Haiku)** — `annotate_casualty.py`, `annotate_gate.py`,
`annotate_event_type.py`, `annotate_event_entities.py`, `annotate_entities_zh.py`,
`annotate_long_news.py`, `annotate_multitask.py`. Coreference: `annotate_coref.py` (buys entity coref clusters over NUMBERED candidate mentions, batch + recover + voted eval draws; the `*_coref_llm` corpora) and `coref_llm_pilot.py` (its pilot: can an LLM cluster news mentions well enough to train the link on).

**Repairs** — `repair_turkish_surfaces.py` (lemmatised gold, **refusing unsafe repairs**),
`repair_casualty_anchors.py`, `repair_contradicted_negatives.py`, `dedupe_splits.py` (`--near-dup T` also drops train docs containing a later split's doc), `add_coref_clusters.py` (adds gold `coref_clusters` IN PLACE, every other byte unchanged),
`stamp_record_metadata.py`, `stamp_field_cardinality.py`, `interleave_splits.py`,
`rebuild_casualty_docee.py` (v2, with the `_standalone` fix the v1 build predated),
`merge_rams_passages.py` (one record per passage; RAMS stores one event per example).

**Slices** — `build_capped_slice.py`: a seeded slice of whole documents, capped by
event-argument gold, holding LLM-annotated data at a fixed real : synthetic ratio.

**Collectors and screens** — `fetch_cc_news.py` (real English news from `vblagoje/cc_news` for annotation),
`harvest_long_ccnews.py` (the long tail the collector threw away), `screen_ccnews_candidates.py` (free regex,
then the extractor), `prefilter_pool.py` (the free numeric-toll regex before the GPU),
`normalize_gate_source.py` (one `source` value per outlet for the gate corpus's balance).

**Self-consistency gold** — `vote_annotations.py` (k runs of the SAME documents -> one voted record; a
classification is voted only across runs asked the same menu), `repair_voted_classifications.py` (re-votes
an existing voted set's classifications from its raw runs, changing nothing else).

**Derived corpora** — `augment_baseword.py` (every surface word, text and labels, reduced to its base form),
`events_to_entities.py` (trigger-only events as typed spans), `roles_to_entities.py` (event ARGUMENT roles
as an entity corpus; Option 4), `merge_entity_types.py` (purchased entity TYPES attached as a typing signal
only), `merge_turkish_event.py` (the Turkish TYPE and ENTITY passes in DocEE's shape), `prepare_docee_zh.py`
(DocEE-zh into DocEE-en's label space), `split_rams_test.py` (rams.test into val and blind-test halves).

**Streaming** — `hf_stream.py`: HuggingFace token-NER datasets into training WITHOUT disk (`data.hf_streaming`).

**Label space** — `build_unified_full.py` (in `train/`; writes `labels/unified-full.yaml` over
every TAXONOMY corpus in `data/`, the base's spellings canonical -- what eb18/eb19 use),
`build_label_maps.py` (in `train/`; the older `labels/unified.yaml`), `apply_label_map.py`,
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
`make_nan_probe_configs.py`, `backfill_model_config.py` (adds `label_map` +
`inference_defaults` to a checkpoint published before 2026-10-01; re-run when a sweep moves
the threshold), `build_unified_full.py` (the label map, see section 2).
How to run a trained checkpoint is in [`train/INFER.md`](train/INFER.md).

**Common inference menus** -- `build_schema.py` (one app menu from corpora's TRAIN gold, for `infer.py --schema-json` and `score_predictions --menu app:`; reports val/test labels it does not offer; `--drop` removes poison labels; outputs in `train/config/schemas/`).

**A/B config generators** (write both arms from a base so they differ only where intended) -- `make_corefB_configs.py`, `make_corefC_configs.py`, `make_rolesym_configs.py` (`negative_gold_full_roles`, the menu-poison fix), `make_relneg_configs.py` (relation negatives, NEGATIVE). **Base config generators** -- `make_eb22_config.py` (eb22 option A, a warm start; crashed, superseded), `make_eb22b_config.py` (eb22 as option B: from scratch with fastino gliner2.5-multi-v1's structural relation/span flags, TODO #40 fixes).

**Mix builders** — `build_137k_replay.py`, `build_scaling_mix.py`,
`build_joint_scaling_mix.py`, `build_warmstart_mix.py`, `build_turkish_dose_mix.py`,
`build_zh_multitask_mix.py`, `build_casualty_multilingual.py`, `build_loc_control.py`.

Probes, sweeps, gates and measures are in [section 1](#1-instruments--the-things-that-measure).

---

## 3a. `tools/derive` — derive a fine-tuning config from measurement

Point it at a pre-split corpus and a pretrained checkpoint; it measures what a fine-tuning
config needs and writes the evidence, one run per (corpus, base) pair. `README.md` there is
the entry point. `derive_config.py` drives 7 stages (`--gpu` adds 4–6 on the VALIDATION split):
`model_probe.py` (describe a base from its config.json alone -- span vs boundary, label map),
`corpus_probe.py` (per-head gold and label measurements, composed from existing tools),
`data_health.py` (each finding with severity, evidence and the fixing command; a BLOCK stops
stage 7), `label_review.py` (the corpus's labels against what the base was trained to read),
`labels_file.py` (the base's label map plus the new corpus), `gpu_stages.py`
(reachability, zero-shot baseline, operating points), `emit_config.py` (stage 7: the config,
every value annotated with its source). `lambda/derive_config_gpu_test.sh` runs stages 1–6
on GPU against three structurally different bases.

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

**Base runs and their smokes** — `eb18_smoke.sh`, `eb19_smoke.sh` + `eb19_smoke_launch.sh`
(the recipe from scratch before the base is bought: loss health from the progress bar, the
in-run junction gate, the label-negatives dose and in-loss negative queries, speed, memory; ANY base config:
`CFG=tools/train/config/base/eb21.yaml bash tools/lambda/eb19_smoke_launch.sh`), `eb19_launch.sh`, `eb21_launch.sh` (the eb21 base, one box, full schedule). Launchers pin `EXPECT_COMMIT`, and
`provision_box.sh` refuses a box at any other commit. Per-epoch checkpoints (`PUSH_EPOCHS=1`)
are pushed AFTER `train.py` exits, not as each epoch ends -- to read an epoch mid-run, copy it
off the box.

**Fast A/Bs** — `fast_ab_launch.sh` (any family: `EXP=<family> ARM=<arm>`, one box per arm,
configs `train/config/ab/<EXP>-<ARM>.yaml`), `p2fast_ab_launch.sh` (its first instance).

**Menu jobs** — `menu_threshold_launch.sh` + `menu_threshold_job.sh` (re-pick each ARM's threshold under
an app menu on val, publish per arm; `ARMS` comma-separated), `menu_probe_job.sh` (the menu-size ladder).
`pull_epoch_det.sh` copies each `checkpoint-epoch-N` off a running base and measures its junction DET mid-run.

**One-off CUDA checks** (publish nothing; results come back by scp; setup duplicated from `run_tests_cuda.sh`) -- `menu_read_box.sh` (a VAL menu read, `sweep_menu_threshold --val-only`, for several checkpoints: 13 min per model on an A10 vs hours on the laptop CPU), `overfit_box.sh` (the overfit traces above, per corpus, `SCRIPT` selects relations or events), `coref_cuda_check.sh` (entity-coref val sweep on CUDA with memory logged; same cut-off as CPU), `full_menu_timing.sh` (seconds per doc for the full-menu blind test; both runs OOMed in the relation decode, TODO #37).

**Operating-point jobs** — `sweep_and_rescore_eb17.sh` (record-gate sweep on validation),
`blind_test_record_anchor.sh` (the blind test ONCE at the picked point),
`anchor_sweep_wide.sh` (does the optimum move once the proposal cap is lifted?),
`width_threshold_arms.sh`, `window_overlap_arms.sh` (where window/overlap CAN matter:
`cc_news_long`), `rescore_eb17_eb18.sh` (the first like-for-like eb17 vs eb18 delta).

**`run_tests_cuda.sh`** — runs the TEST SUITE on a real CUDA box, then terminates it. `TESTS="<node ids>"` narrows it and keeps the tracebacks.
`resolve_device` picks cuda -> mps -> cpu, so a laptop run never takes the cuda branch:
two known device-specific facts (MPS `torch.gather` on int64 corrupting above 2^24, and
the DDP test's skip guard being unable to fire on Darwin) are invisible locally in both
directions. Needs no `$HF_TOKEN`, because it publishes nothing. Does the cu128 swap itself (`--reinstall-package torch`, the load-bearing flag) and REFUSES to score anything if `torch.cuda.is_available()` is still False -- without that it once ran the whole suite with CUDA dead and reported 2,814 passed.

---

## 5. `tools/ekf_showcase` — text→tracking

`run_pipeline.py` (1,301 lines) is the whole research line in one command: news feed →
EKF-tracked casualty timeline. `README.md` there is the entry point.

**Feeds and ground truth** — `build_helene_feed.py`, `build_turkey_feed.py`, `build_turkey_feed_tr.py`, `build_turkey_feed_zh.py`,
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

`train/label_style_v3.py` (via `build_unified_full.py --v3`) — derive `labels/unified-full-v3.yaml`:
Title_Snake over every training label, from the reviewed `events_working_papers/label_style/*.tsv`; writes the
`style:` block training and inference apply; refuses blank decisions and failed gates.
`train/config/menus/` — application menus (MENU_SPEC.md): a curated schema plus the corpora annotated
exhaustively against it (`news55.json`). Used by `eval.menu` / `eval.calibration_menu`, `eval.py --menu`,
`score_predictions.py --menu` and `infer.py --schema-json`.
`train/score_predictions.py` — score an `infer.py --output` file exactly as the blind test scores
(same scorer, filter, duplicate drop, language buckets; `gold_mapped` when present); `--card` prints the
model card's tables. Warns when predictions were not made with `--gold-schema`.
`infer.py` — command-line inference, with optional document-level global decoding. Loads
boundary checkpoints (`AutoExtractor`; `GLiNER2` is the span class), takes the checkpoint's own
schema (`--model-schema --tasks ...`), and applies its `inference_defaults` and `label_map`
like the viewer. `--merge-coreferent link --coref-threshold` merges coreferent trigger mentions
on checkpoints trained with the link (eb20+). Examples and cautions in [`train/INFER.md`](train/INFER.md).
`train/calibrate_coref_threshold.py` — pick a trained checkpoint's coreference merge threshold on val
(grid 0.5-0.95) and, with `--write`, store it in its config.json only if the merges pass the gates
(>= 20 merges, precision >= 0.8, fusion < 5%, cluster + argument F1 >= off, and on the one-trigger
`--control` corpora no F1 loss and fusion < 5%). train.py runs the same sweep when
`eval.coref_calibration` is set, and scores the blind test again with the merge off if it ships.
`import_surface.py` / `compare_surface.py` — snapshot and diff a package's public import
surface, for verifying a refactor removed only what it meant to.

---

## Where the findings live

This catalog says what each tool *is*. What each one *measured* is in
[`events_working_papers/EXPERIMENT_CATALOG.md`](events_working_papers/EXPERIMENT_CATALOG.md)
(chronological, newest first, including the nulls and negatives — they are the majority),
with open work in [`events_working_papers/TODO.md`](events_working_papers/TODO.md) and the
training recipe in [`train/TRAINING.md`](train/TRAINING.md).
