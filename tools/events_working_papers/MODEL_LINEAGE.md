# Model Lineage — every training config, in training order, mapped to the research

Status: index. Date: 2026-08-17. Covers all 93 files in `tools/train/config/`.

Every model we trained is a chain: some base checkpoint, then one or more training stages,
each stage a YAML in `tools/train/config/`. This document walks those stages in the order
they run and, for each config, names the working paper that motivates or reports it. Read a
row bottom-up and you get a model's provenance; read a stage top-down and you get every arm
we ran at that stage.

Scope note: [[PIPELINES]] maps the *inference* path (text in, records out). This maps the
*training* path. They do not overlap.

## Terminology — "warm start" means three different things here

The configs use one phrase for three distinct operations. The stage numbering below keys off
the mechanism, not the word:

| usage | mechanism | where |
|---|---|---|
| **cold start** | `model.encoder:` — `from_encoder`, fresh GLiNER2 heads on a raw encoder | Stage 2 |
| **warm start (external)** | `model.pretrained: fastino/...` — continues heads Fastino trained | Stage 4 |
| **warm start (intermediate stage)** | `model.pretrained: ./out/...` — continues heads *we* trained, as a deliberate mid-training stage | Stage 3 |

`architecture: boundary` marks the boundary head; its absence means the older span head. The
two are not interchangeable — warm-starting across them raises `ArchitectureMismatchError`.

## The chains

```
  A. Fastino span line (the workhorse)
     fastino/gliner2-{base,large,multi}-v1  --[Stage 4]-->  ~30 task fine-tunes
                                            --[Stage 5]-->  casualty-* (EKF extractor)

  B. mmBERT span line (head-init question)
     jhu-clsp/mmBERT-{small,base}  --[Stage 1]-->  loss sweep
                                   --[Stage 2]-->  mmbert-base-rams  --[Stage 4]--> mmbert-base-wikievents
                                   --[Stage 2]-->  scaling-mmbert-{10k,40k,100k}

  C. mmBERT boundary line (Paper 2 — the main experimental spine)
     jhu-clsp/mmBERT-base
        --[Stage 2]--> joint-boundary-mmbert-{10k,40k,100k,137k}
                          |                         |
                          |                         +--[Stage 3]--> warmstart-natural(-*), warmstart-anchorless,
                          |                         |                joint-boundary-warmstart-struct
                          |                         |                    |
                          |                         |                    +--[Stage 4]--> rams-clean-{b,c}
                          |                         +--[Stage 4]--> rams-clean-a-base137k
                          +--[Stage 4]--> joint-boundary-{rams,redocred}-{10k,40k,100k,137k}
                                                     |
                                                     +--[Stage 4]--> {maven,casie}-tier2-{control,eventrecords}
```

Chain C is the one the program rests on: [[JOINT_IE_SCALING]] is its design, and Stage 2 → 4
is the 12-arm head-init curve.

---

## Stage 0 — Bases we did not train

No config; these are the roots every chain starts from.

| base | what it is | working paper |
|---|---|---|
| `fastino/gliner2-base-v1` | DeBERTa-v3-base, 254K examples, span + structure heads, **no event or relation head** | [[FASTINO_GLINER2_TRAINING]] §1, §3 |
| `fastino/gliner2-large-v1` | the large counterpart | [[FASTINO_GLINER2_TRAINING]] |
| `fastino/gliner2-multi-v1` | mDeBERTa-v3 multilingual counterpart | [[FASTINO_GLINER2_TRAINING]] |
| `microsoft/deberta-v3-base` | raw encoder, 512 positions → 384-word window | [[PAPER_0_FOUNDATION]] §9.5 |
| `jhu-clsp/mmBERT-base` / `-small` | raw encoder, 8192 positions — whole documents in one pass | [[PAPER_0_FOUNDATION]] §9.5 |

Why it matters downstream: base-v1 has no event head, so every event capability in Stage 4 is
being learned from a head that was never trained on events. That is the head-init bottleneck
the whole program circles ([[FASTINO_GLINER2_TRAINING]] §4-5).

## Stage 1 — Struct-loss selection (cold start, small scale)

Which structure loss, decided before spending on scale. All `from_encoder` on mmBERT.

| config | init | data | working paper | role |
|---|---|---|---|---|
| `mmbert-small.yaml` | mmBERT-small | ACE 2005 + stockmark_jpn, klue_re | [[PAPER_0_FOUNDATION]] §6 | focal baseline |
| `mmbert-small-focal.yaml` | mmBERT-small | WikiEvents | [[PAPER_0_FOUNDATION]] §6, §10.3 | focal arm |
| `mmbert-small-bce.yaml` | mmBERT-small | WikiEvents | [[PAPER_0_FOUNDATION]] §10.3 | plain BCE arm |
| `mmbert-small-bce-posweight.yaml` | mmBERT-small | WikiEvents | [[PAPER_0_FOUNDATION]] §10.3 | BCE + pos-weight — **the variant since adopted as the project default** (47 configs set it) |
| `mmbert-small-dice.yaml` | mmBERT-small | WikiEvents | [[PAPER_0_FOUNDATION]] §10.3 | soft-Dice arm |
| `mmbert-small-bce-dice.yaml` | mmBERT-small | WikiEvents | [[PAPER_0_FOUNDATION]] §10.3 | BCE + soft-Dice arm |
| `mmbert-small-asl.yaml` | mmBERT-small | WikiEvents | [[PAPER_0_FOUNDATION]] §10.3 | asymmetric-loss arm |
| `mmbert-base.yaml` | mmBERT-base | 25 corpora + 7 event corpora | [[PAPER_0_FOUNDATION]] §6, §7 | scale-up of the chosen loss; the broadest span-arch mix we ran |

`bce_posweight` with `pos_weight 8` is what the from-encoder configs downstream inherit.
Note the provenance gap: adoption is documented, the *comparison* is not —
[[PAPER_0_FOUNDATION]] §10.3's table is still `TBD` in every cell, so these seven arms are a
sweep that was run and acted on but never written up. [[EVENT_LOSS_PLAN]] is the
implementation record for separating event loss out of `structure_loss`.

## Stage 2 — Cold start (`from_encoder`, fresh heads)

The head-init question: can fresh heads be warmed from data alone, and how much data does it
take? Two architecture arms.

### 2a. Span architecture

| config | init | data | working paper | role |
|---|---|---|---|---|
| `mmbert-base-rams.yaml` | mmBERT-base | RAMS | [[PAPER_0_FOUNDATION]] §10.6, §9.5 | the long-context from-encoder reference; pushed as `whr778/mmbert-base-rams` |
| `deberta-base-fromenc-rams.yaml` | deberta-v3-base | RAMS | [[PAPER_0_FOUNDATION]] §10.6 | **encoder-isolation control** — same recipe, short context, separates encoder effect from head-init effect |
| `mmbert-base-combined.yaml` | mmBERT-base | synthetic + GLiNER multilingual + multi-task NER + RAMS (~96K, 2 epochs) | [[PAPER_0_FOUNDATION]] §10.7 | the **treatment base** of the broad-data head-init A/B — **a negative result**; its WikiEvents fine-tune is the treatment arm in Stage 4b |
| `deberta-base-fromenc-synthetic.yaml` | deberta-v3-base | synthetic_sonnet5_1k | [[PAPER_0_FOUNDATION]] §7, §10.6 | can synthetic data alone teach all five tasks from scratch |
| `scaling-mmbert-10k.yaml` | mmBERT-base | 10 event corpora, 10K nested subsample | [[SCALING_CURVE_EXPERIMENT]] §3-4, [[HEAD_INIT_DATA_SCALE]] §4 | span data-scaling curve |
| `scaling-mmbert-40k.yaml` | mmBERT-base | same pool, 40K | [[SCALING_CURVE_EXPERIMENT]] | span data-scaling curve |
| `scaling-mmbert-100k.yaml` | mmBERT-base | same pool, ~100K (full) | [[SCALING_CURVE_EXPERIMENT]] | span data-scaling curve; arg F1 0.050 / 0.115 / 0.158 across the three points |
| `mmbert-base-masakhaner.yaml` | mmBERT-base | MasakhaNER 2.0 (20 African langs) | [[PAPER_0_FOUNDATION]] §7 | multilingual from-encoder NER |
| `mmbert-base-masakhanews.yaml` | mmBERT-base | MasakhaNEWS (16 langs) | [[PAPER_0_FOUNDATION]] §7 | multilingual from-encoder classification |
| `mmbert-base-wikiann.yaml` | mmBERT-base | WikiANN, **HF-streamed** (no disk) | [[PAPER_0_FOUNDATION]] §7 | multilingual NER at 176-language scale; step-bounded because a stream has no length |

The curve here is what [[HEAD_INIT_DATA_SCALE]] §2 predicted a bracket for and
[[PAPER_0_FOUNDATION]] §10.8 reports.

### 2b. Boundary architecture — the joint_ie bases

| config | init | data | working paper | role |
|---|---|---|---|---|
| `joint-boundary-mmbert-10k.yaml` | mmBERT-base | 3 relation + 10 event corpora, 10K total | [[JOINT_IE_SCALING]] §3b, §4 | boundary cold-start base |
| `joint-boundary-mmbert-40k.yaml` | mmBERT-base | same, 40K | [[JOINT_IE_SCALING]] §4 | boundary cold-start base |
| `joint-boundary-mmbert-100k.yaml` | mmBERT-base | same, 100K | [[JOINT_IE_SCALING]] §4 | boundary cold-start base |
| `joint-boundary-mmbert-137k.yaml` | mmBERT-base | same, 136,787 (full) | [[JOINT_IE_SCALING]] §4 | **the base everything in Stages 3-4 hangs off**; pushed as `whr778/gliner2-joint-boundary-mmbert-137k` |

Deliberately separate output paths from `scaling-mmbert-*`: these do not overwrite the span
runs, so the two architectures stay comparable ([[JOINT_IE_SCALING]] §3c).

## Stage 3 — Intermediate warm-start stage (boundary)

Between the 137K base and the downstream task: does a broad multi-task stage help? All init
from `./out/joint-boundary-mmbert-137k/best`, all `architecture: boundary`.

| config | data | working paper | role |
|---|---|---|---|
| `joint-boundary-warmstart-struct.yaml` | `warmstart_mix` + 12 event corpora (val/test only), 30% replay | [[JOINT_IE_SCALING]] §7 | adds STRUCTURE + NER; exists because mmbert-137k cannot do `[C]` record extraction at all |
| `warmstart-natural.yaml` | `mix_natural` | [[JOINT_IE_SCALING]] §7 | **record-mode A/B, arm A** — `natural`; the control recipe for all of Phase 3 |
| `warmstart-anchorless.yaml` | `mix_anchorless` | [[JOINT_IE_SCALING]] §7 | **arm B** — `anchorless`; byte-identical mixture bar `record_metadata`. Learns nothing |
| `warmstart-natural-seed43.yaml` | `mix_natural` | [[EVENT_LOSS_PHASE3_PLAN]] §4 | **noise floor** — control recipe at a second seed; any \|delta\| below this gap is unreadable |
| `warmstart-natural-evw05.yaml` | `mix_natural` | [[EVENT_LOSS_PHASE3_PLAN]] §4 | run 1 dose-response, `task_loss_weights` events=0.5 |
| `warmstart-natural-evw20.yaml` | `mix_natural` | [[EVENT_LOSS_PHASE3_PLAN]] §4 | run 1, events=2.0 |
| `warmstart-natural-evw40.yaml` | `mix_natural` | [[EVENT_LOSS_PHASE3_PLAN]] §4 | run 1, events=4.0 |
| `warmstart-natural-evwide2.yaml` | `mix_natural` | [[EVENT_LOSS_PHASE3_PLAN]] §10 | run 2 extended-reach flat weight, w=2.0 → 12.5% of gradient |
| `warmstart-natural-evwide4.yaml` | `mix_natural` | [[EVENT_LOSS_PHASE3_PLAN]] §10 | run 2, w=4.0 → 22.2% of gradient |
| `warmstart-natural-evpw08.yaml` | `mix_natural` | [[EVENT_LOSS_PHASE3_PLAN]] §10 | run 2 per-task **pos_weight** 8.0 → 12.2% of gradient |
| `warmstart-natural-evpw16.yaml` | `mix_natural` | [[EVENT_LOSS_PHASE3_PLAN]] §10 | run 2, pos_weight 16.0 → 17.8% |
| `warmstart-natural-evpw32.yaml` | `mix_natural` | [[EVENT_LOSS_PHASE3_PLAN]] §10 | run 2, pos_weight 32.0 → 27.1% |
| `warmstart-natural-gist.yaml` | `mix_natural` | [[EKF_MHT_DESIGN]] §27.4-27.9 | GIST guide-embedding arm — the query-axis hard-negative test |

The `evw*` / `evpw*` arms differ from `warmstart-natural.yaml` in exactly one field group
(the loss weights) plus `output_dir` and `experiment_name`. They are generated, not
hand-edited. Run 1's flat weight was a **null lever**; §10 explains why and computes run 2's
doses rather than guessing them.

## Stage 4 — Task training (the downstream fine-tune)

### 4a. On the boundary bases — Paper 2's curve and its Tier 2 follow-ups

| config | init | data | working paper | role |
|---|---|---|---|---|
| `joint-boundary-rams.yaml` | 137k base | RAMS | [[JOINT_IE_SCALING]] §4 | **template** — the per-base copies below are generated from it |
| `joint-boundary-rams-10k.yaml` | 10k base | RAMS | [[JOINT_IE_SCALING]] §4 | event downstream, curve point |
| `joint-boundary-rams-40k.yaml` | 40k base | RAMS | [[JOINT_IE_SCALING]] §4 | event downstream, curve point |
| `joint-boundary-rams-100k.yaml` | 100k base | RAMS | [[JOINT_IE_SCALING]] §4 | event downstream, curve point |
| `joint-boundary-rams-137k.yaml` | 137k base | RAMS | [[JOINT_IE_SCALING]] §4 | event downstream; pushed as `whr778/gliner2-joint-boundary-rams-137k` |
| `joint-boundary-redocred.yaml` | 137k base | Re-DocRED | [[JOINT_IE_SCALING]] §4 | **template** for the relation downstream |
| `joint-boundary-redocred-10k.yaml` | 10k base | Re-DocRED | [[JOINT_IE_SCALING]] §4 | relation downstream, curve point |
| `joint-boundary-redocred-40k.yaml` | 40k base | Re-DocRED | [[JOINT_IE_SCALING]] §4 | relation downstream, curve point |
| `joint-boundary-redocred-100k.yaml` | 100k base | Re-DocRED | [[JOINT_IE_SCALING]] §4 | relation downstream, curve point |
| `joint-boundary-redocred-137k.yaml` | 137k base | Re-DocRED | [[JOINT_IE_SCALING]] §4 | relation downstream, curve point |
| `rams-clean-a-base137k.yaml` | 137k base | RAMS | [[JOINT_IE_SCALING]] §4 | **CONTROL** — the published 137K recipe re-run on current code, after ~10 commits touched the loss path |
| `rams-clean-b-warmstart.yaml` | `whr778/gliner2-warmstart-natural-clean` | RAMS | [[JOINT_IE_SCALING]] §4, [[EVENT_LOSS_PHASE3_PLAN]] | **TREATMENT** — does routing through the Stage 3 `mix_natural` stage help the event downstream |
| `rams-clean-c-evwide2.yaml` | `whr778/gliner2-warmstart-natural-evwide2-clean` | RAMS | [[EVENT_LOSS_PHASE3_PLAN]] §10 | as B, but the Stage 3 arm carried the event-weighted loss. Read against B, not A |
| `maven-tier2-control.yaml` | `whr778/gliner2-joint-boundary-rams-137k` | MAVEN | [[JOINT_IE_SCALING]] Tier 2 | does the RECORD head recover instances the mention path cannot express — control arm |
| `maven-tier2-eventrecords.yaml` | same | MAVEN | [[JOINT_IE_SCALING]] Tier 2 | treatment arm. MAVEN over CASIE: 12x the instance supervision |
| `casie-tier2-control.yaml` | same | CASIE | [[JOINT_IE_SCALING]] Tier 2 | control arm on the multi-instance-dense corpus (94.5% of docs repeat an event type) |
| `casie-tier2-eventrecords.yaml` | same | CASIE | [[JOINT_IE_SCALING]] Tier 2 | treatment arm |

The 12 curve arms are Stage 2 × Stage 4 (4 bases × {RAMS, Re-DocRED} plus the base's own
metrics). All must be read at a **matched threshold** — [[JOINT_IE_SCALING]] §4b records the
defect that hid this once already.

### 4b. On the Fastino span bases — English events

| config | init | data | working paper | role |
|---|---|---|---|---|
| `gliner2-base-v1-rams.yaml` | base-v1 | RAMS | [[PAPER_0_FOUNDATION]] §10.4 | warm-start event reference; the head-init contrast to `mmbert-base-rams` |
| `gliner2-base-v1-casie.yaml` | base-v1 | CASIE | [[PAPER_0_FOUNDATION]] §10.4 | event sweep |
| `gliner2-base-v1-docee.yaml` | base-v1 | DocEE | [[PAPER_0_FOUNDATION]] §10.4 | event sweep |
| `gliner2-base-v1-wikievents.yaml` | base-v1 | WikiEvents | [[PAPER_0_FOUNDATION]] §10.1 | the blind-test headline run |
| `gliner2-base-v1-events-english.yaml` | base-v1 | CASIE + DocEE + MAVEN + RAMS + WikiEvents | [[PAPER_0_FOUNDATION]] §10.4 | joint English-event training |
| `gliner2-large-v1-rams.yaml` | large-v1 | RAMS | [[PAPER_0_FOUNDATION]] §10.4 | large-model sweep |
| `gliner2-large-v1-casie.yaml` | large-v1 | CASIE | [[PAPER_0_FOUNDATION]] §10.4 | large-model sweep |
| `gliner2-large-v1-docee.yaml` | large-v1 | DocEE | [[PAPER_0_FOUNDATION]] §10.4 | large-model sweep |
| `gliner2-large-v1-maven.yaml` | large-v1 | MAVEN (event form) | [[PAPER_0_FOUNDATION]] §10.4 | large-model sweep |
| `gliner2-large-v1-wikievents.yaml` | large-v1 | WikiEvents | [[PAPER_0_FOUNDATION]] §10.4 | large-model sweep. **Header comment says multi-v1; the `pretrained:` field says large-v1 — trust the field** |
| `gliner2-large-v1-events-english.yaml` | large-v1 | 5 English event corpora | [[PAPER_0_FOUNDATION]] §10.4 | joint English-event training, large |
| `gliner2-multi-wikievents.yaml` | multi-v1 | WikiEvents | [[PAPER_0_FOUNDATION]] §10.1 | SOTA-comparison probe on WikiEvents |
| `mmbert-base-wikievents.yaml` | `whr778/mmbert-base-rams` | WikiEvents | [[PAPER_0_FOUNDATION]] §10.6-10.7 | **the one span-line two-stage chain**: 206 train docs is too few for fresh heads, so it inherits argument competence from the RAMS mmBERT. **This file served both §10.7 arms** — as checked in it is the *control* (RAMS base); the *treatment* run re-pointed `pretrained:` at `whr778/mmbert-base-combined`. The treatment's parent is not recoverable from the file, only from §10.7 |

### 4c. On the Fastino span bases — Chinese and multilingual

| config | init | data | working paper | role |
|---|---|---|---|---|
| `gliner2-multi-v1-cmnee.yaml` | multi-v1 | CMNEE (Chinese military news) | [[PAPER_0_FOUNDATION]] §10.4 | Chinese doc-level events |
| `gliner2-multi-v1-duee.yaml` | multi-v1 | DuEE 1.0 | [[PAPER_0_FOUNDATION]] §10.4 | full trigger + typed-argument Chinese events |
| `gliner2-multi-v1-chfinann.yaml` | multi-v1 | ChFinAnn | [[PAPER_0_FOUNDATION]] §10.4 | **trigger-free** — converter emits role-typed entities + multi-label classification, so the metric is classification F1 |
| `gliner2-multi-v1-docfee.yaml` | multi-v1 | DocFEE | [[PAPER_0_FOUNDATION]] §10.4 | trigger-free and offset-free; same shape as ChFinAnn |
| `gliner2-multi-v1-events-chinese.yaml` | multi-v1 | CMNEE | [[PAPER_0_FOUNDATION]] §10.4 | joint Chinese-event training |
| `gliner2-multi-v1-events-all.yaml` | multi-v1 | 6 corpora, English + Chinese | [[PAPER_0_FOUNDATION]] §10.4 | the widest event mix on the span architecture |
| `gliner2-multi-v1-masakhaner.yaml` | multi-v1 | MasakhaNER 2.0 | [[PAPER_0_FOUNDATION]] §7 | warm-start counterpart to `mmbert-base-masakhaner` |
| `gliner2-multi-v1-masakhanews.yaml` | multi-v1 | MasakhaNEWS | [[PAPER_0_FOUNDATION]] §7 | warm-start counterpart to `mmbert-base-masakhanews` |
| `gliner2-multi-v1-wikiann.yaml` | multi-v1 | WikiANN, HF-streamed | [[PAPER_0_FOUNDATION]] §7 | warm-start counterpart to `mmbert-base-wikiann` |

### 4d. On the Fastino span bases — relations, NER, synthetic

| config | init | data | working paper | role |
|---|---|---|---|---|
| `gliner2-base-v1-redocred.yaml` | base-v1 | RE-DocRED | [[PAPER_0_FOUNDATION]] §10.5, [[RE_DIFFERENCES]] | relation extraction; selected on relation F1, not loss |
| `gliner2-large-v1-redocred.yaml` | large-v1 | RE-DocRED | [[PAPER_0_FOUNDATION]] §10.5 | relation extraction, large |
| `gliner2-large-v1-docred.yaml` | large-v1 | DocRED (original annotation) | [[PAPER_0_FOUNDATION]] §10.5 | the un-re-annotated contrast to RE-DocRED |
| `gliner2-base-v1-biomed-ner.yaml` | base-v1 | 13 MTL-Bioinformatics-2016 corpora | [[PAPER_0_FOUNDATION]] §7 | NER is what this head was trained for, so warm start is the strongest default |
| `gliner2-base-v1-maven.yaml` | base-v1 | `data/maven_ner` | [[PAPER_0_FOUNDATION]] §10.4 | MAVEN as **NER**, not as events — unlike `gliner2-large-v1-maven`. Only a train split is on disk |
| `gliner2-base-v1-mendeley-ed.yaml` | base-v1 | Mendeley Event Detection | [[PAPER_0_FOUNDATION]] §10.4 | one generic event type, no arguments; metric is trigger F1 |
| `gliner2-base-v1-synthetic.yaml` | base-v1 | synthetic_sonnet5_1k | [[PAPER_0_FOUNDATION]] §7 | can our synthetic data teach all five tasks — **in-distribution gain only**; see the eval caveat below |
| `gliner2-base-v1-synthetic-haiku5k.yaml` | base-v1 | synthetic_haiku45_5k | [[PAPER_0_FOUNDATION]] §7 | the 5K Haiku-generated counterpart |
| `rams-baseword.yaml` | base-v1 | RAMS, lemma/base-word form | [[TODO]] | lemmatized-argument arm; **provisional** — unswept threshold |
| `rams-duplicate-control.yaml` | base-v1 | RAMS, `train.duplicate_control` | [[TODO]] | the matched control for `rams-baseword` |

## Stage 5 — Application training (the EKF extractor)

Downstream of Paper 0's models, upstream of Paper 1's tracker. All init from
`fastino/gliner2-base-v1`.

| config | data | working paper | role |
|---|---|---|---|
| `casualty-finetune.yaml` | `casualty_ft` (Sonnet-5 realized) | [[EKF_MHT_DESIGN]] §19-20 | closes the extraction gap the §19 `missing`-role probe predicted: zero-shot precision 0.63 + confidence-cut selection bias |
| `casualty-multievent.yaml` | `casualty_multi` | [[EKF_MHT_DESIGN]] §20, [[COUNTING_LAYER]] | the §20 corpus had exactly one `casualty_report` in all 31,539 docs, so the count head only ever saw "1". This fixes that |
| `casualty-docee.yaml` | `casualty_docee` | [[EKF_MHT_DESIGN]] §20-21 | successor to `casualty-multievent`: synthetic trajectories paired with **real** DocEE contexts |

## Appendix — configs that are not links in any lineage

| config | working paper | what it is |
|---|---|---|
| `eval-preservation-ner.yaml` | [[PAPER_0_FOUNDATION]] §8 | **eval-only.** Scores a checkpoint on general-domain NER base-v1 is already good at. Exists because the synthetic configs cannot answer their own question — their val measures in-distribution gain, not preservation. Run twice, same command, different `--checkpoint` |
| `parity-deberta-rams.yaml` | [[JOINT_IE_SCALING]], [[BOUNDARY_ARCHITECTURE]] | port validation. One short run exercises from_encoder + bce_posweight + metric sweep + global decode + sliding window + events on a real encoder. Not a result |
| `probe-nan-a.yaml` | [[BOUNDARY_ARCHITECTURE]] §13 | NaN isolation, arm A: sdpa + bf16 (the failing baseline) |
| `probe-nan-b.yaml` | [[BOUNDARY_ARCHITECTURE]] §13 | arm B: sdpa + fp32 (isolates dtype) |
| `probe-nan-c.yaml` | [[BOUNDARY_ARCHITECTURE]] §13 | arm C: flash_attention_2 + bf16 (the fix) |
| `stopwords.yaml` | — | **not a training config.** A stopword supplement keyed by ISO 639-2 codes, merged with the `stopwordsiso` package by `build_stopwords()` |

The three `probe-nan-*` configs are generated by `tools/train/make_nan_probe_configs.py` — do
not hand-edit.

## Reading the lineage backwards

Given a checkpoint, walk up:

| pushed checkpoint | produced by | its parent |
|---|---|---|
| `whr778/gliner2-joint-boundary-mmbert-137k` | `joint-boundary-mmbert-137k.yaml` | mmBERT-base (raw) |
| `whr778/gliner2-joint-boundary-rams-137k` | `joint-boundary-rams-137k.yaml` | the 137k base |
| `whr778/gliner2-joint-boundary-warmstart-natural` | `warmstart-natural.yaml` | the 137k base |
| `whr778/gliner2-warmstart-natural-clean` | `warmstart-natural.yaml`, clean-data re-run *(inferred)* | the 137k base |
| `whr778/gliner2-warmstart-natural-evwide2-clean` | `warmstart-natural-evwide2.yaml`, clean-data re-run *(inferred)* | the 137k base |
| `whr778/mmbert-base-rams` | `mmbert-base-rams.yaml` | mmBERT-base (raw) |
| `whr778/mmbert-base-combined` | `mmbert-base-combined.yaml` | mmBERT-base (raw) |
| `whr778/mmbert-base-combined-wikievents` | `mmbert-base-wikievents.yaml`, **re-pointed** at the combined base | `whr778/mmbert-base-combined` |
| `whr778/scaling-mmbert-{10k,40k,100k}` | `scaling-mmbert-*.yaml` | mmBERT-base (raw) |
| `whr778/gliner2-base-v1-synthetic` | `gliner2-base-v1-synthetic.yaml` | `fastino/gliner2-base-v1` |
| `whr778/deberta-base-fromenc-synthetic` | `deberta-base-fromenc-synthetic.yaml` | deberta-v3-base (raw) |
| `whr778/gliner2-large-v1-docee` | `gliner2-large-v1-docee.yaml` | `fastino/gliner2-large-v1` |
| `whr778/gliner2-maven-tier2-*` | `maven-tier2-*.yaml` | `whr778/gliner2-joint-boundary-rams-137k` |

**A config is not a permanent record of the run that used it.** `pretrained:` gets re-pointed
between arms and the previous value leaves no trace in git if the edit was never committed —
`mmbert-base-wikievents.yaml` above is the proven case, where only [[PAPER_0_FOUNDATION]]
§10.7 records that a second arm existed and what its base was. Two rows marked *(inferred)*
rest on checkpoint naming alone: the `-clean` pair is documented in no working paper, only in
the `pretrained:` fields of `rams-clean-b`/`-c`. Treat both as claims to confirm against a run
log, not as established lineage.

## Two standing cautions

**Contamination.** Split uniqueness is a gate, not a nicety: any arm read across a
train/val/test overlap is invalid regardless of how clean the lineage looks. See [[TODO]] for
the known table.

**Matched thresholds.** Curve points and A/B arms must be compared at the same decision
threshold. [[JOINT_IE_SCALING]] §4b documents a case where an unmatched threshold produced a
result that survived review and was later retracted.

## Related

- [[RESEARCH_PROGRAM]] — which working paper feeds which of the three papers
- [[PIPELINES]] — the inference path these models are deployed into
- [[BOUNDARY_ARCHITECTURE]] — what the boundary head actually does with these configs
- [[PROJECT_JOURNAL]] — chronological record, including the decisions later overturned
