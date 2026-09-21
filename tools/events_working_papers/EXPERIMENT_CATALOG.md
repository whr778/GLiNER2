# Experiment catalog

Every experiment this programme has run, newest first. Compiled 2026-09-19 from the session
memory store, `PROJECT_HISTORY.md`, and the git history of all four branches.

**How to read it.** "Outcome" states what the run actually measured, including the nulls and
the negatives — those are the majority, and they are the point. A cost of "—" means no GPU was
rented: a local measurement, an analysis, or a build. Costs are the real billed figures where
they were recorded, not estimates.

**Branch attribution is mechanical, not editorial.** `mmbert_training` ran from 2025-07-07 to
**2026-08-05**, when it was merged whole into `merge/main-20260805` at `ca118ce`; it is an
ancestor of that branch, so nothing was lost. Everything dated on or before 2026-08-05 is
`mmbert_training` work; everything after is `merge/main-20260805`.
`fix/consistency-loss-half-precision-nan` exists only to carry upstream PR #155.

| branch | span | commits |
|---|---|---|
| `mmbert_training` | 2025-07-07 → 2026-08-05 | 373 |
| `merge/main-20260805` | 2026-08-05 → present | 1,185 total |
| `fix/consistency-loss-half-precision-nan` | → 2026-09-17 | 189 |
| `main` | → 2026-07-01 (upstream tracking) | 147 |

## Research lines

Every experiment belongs to one, and the column says which. The lines are not equally
productive and the counts say so.

| line | n | what it is | where it stands |
|---|---:|---|---|
| **Base** | 14 | the base model itself — scaling curves, label space, warm starts, head init | head-init identified as the argument bottleneck; the 137k curve is clean; label unification confounded by a changed test set |
| **Events** | 13 | event records, arguments, decode, and Option 4 | the live line — a third of `event_argument` loss is spans NEVER PROPOSED, and extraction supervision is the only untried lever |
| **EKF** | 9 | text→tracking: the disaster pipeline, MHT, casualty extraction | front-end rebuilt and winning (binding precision 67–100% vs 0–7.7%); gate 2 proven unreachable by any data-mix change |
| **Gates** | 8 | relevance gating ahead of the pipeline | the model we REPLACED still wins swept (AUC 0.9635); the lever was never training, it was the operating point (0.5 → 0.998) |
| **Infra** | 7 | tooling, DDP, throughput, upstream fixes | PR #155 filed upstream; most runs now fail locally in seconds instead of minutes on a billing box |
| **Data** | 7 | corpus generation and annotation purchase | synthetic alone costs −23% on general NER, and the real+synthetic MIX preserved worst of three arms |
| **Multiling** | 2 | non-English capability | the extractor cannot read Turkish (proven with an English control); the gate now can, after a language fix |

**Standing rule this catalog exists to serve:** a delta is not a result without a noise floor
beside it. Measured floors: `event_argument` seed sd **0.0009–0.0020**, `entity` **0.0139**,
`classification` **0.0013–0.0178**, relations **±0.041**, single runs **±0.02**.

---

## merge/main-20260805 — September 2026

| date | line | experiment | purpose | where | outcome | cost | refs |
|---|---|---|---|---|---|---|---|
| 2026-09-21 | **Events** | **typed-margin CEILING measured** (`w_S` brackets) | before spending $34 on the typed-ON arm: how much probability mass can the margin actually move? | local, held-out `val`, 4 corpora | **CONDITIONAL, and the condition is unmeasured.** The loss effect is `p_gold'/p_gold = 1/(1+(e^d-1)*w_S)`, so the mass `w_S` on disallowed candidates bounds everything at any delta. Three brackets at d=ln2: disallowed candidates placed RANDOMLY give 0.0-0.7%, placed at the BOTTOM give 0.000%, placed at the TOP give **20.0% (cmnee), 4.2% (scierc), 2.1% (duee), 0.0% (casie)**. So the margin has real travel ONLY if type-disallowed fillers are top competitors -- exactly the case it exists for, and exactly the one these brackets cannot see. Candidate distributions are peaked (H 0.46-3.07 nats, i.e. 1.6-21 effective candidates). Train vs val barely differs, so memorisation is NOT the explanation. **DO NOT LAUNCH until the real `typed_margin_mask` replaces the brackets** | — | `OPTION_2_TYPED_ROLE_CONSTRAINTS.md` |
| 2026-09-21 | **Events** | corpus familiarity vs logit scale | is the logit scale a (language, domain) FAMILIARITY signal rather than a nuisance to normalise? | local, mmBERT-base MLM, 12 corpora | **POSITIVE but n=4.** Base-encoder masked pseudo-perplexity rank-predicts the RERANK logit sd perfectly, Spearman **rho = -1.000** (cmnee 4.7ppl/6.58sd, casie 5.8/5.17, duee 6.5/4.56, scierc 6.7/2.99); a perfect ordering by chance is p=0.042. Proposal path rho=-0.200, i.e. nothing -- and it has only 1.39x of spread to explain against rerank's 2.20x. Language alone predicts NOTHING: zh finance is the most familiar cell (3.5) and en biomedical the least (10.9-18.8), and two zh news corpora differ 1.4x. The **cell**, not the margin, is the unit | — | `tools/train/measure_corpus_familiarity.py` |
| 2026-09-21 | **Infra** | two calibration instruments committed | the scale/familiarity work was ad-hoc scratch that could not be re-run on a new corpus | local | BUILT: `measure_logit_scale.py` (per-corpus live-logit sd, EMA-stabilised CV, implied k, mixed-stream dose, clamp sweep) and `measure_corpus_familiarity.py` (base-encoder masked pseudo-perplexity per (lang, domain) cell). Both run offline from a local checkpoint. Typed-margin clamp moved 1.5 -> **2.0** on evidence: 1.5 clipped 50% of proposal batches and moved the scale <2%, 2.0 clips 3% for an identical answer | — | `tools/train/measure_logit_scale.py` |
| 2026-09-21 | **Infra** | instrument double-counted the proposal path | `reranker_listwise_loss` DELEGATES to `proposal_listwise_loss`, so patching the name in `losses` counted every rerank call twice | local | **FOUND BY TRACING, and it had already corrupted a reported table.** Signature: n=16 proposal against n=8 rerank on 8 batches, exactly 2x. Proposal sd was a 50/50 blend of both paths -- cmnee read 4.26, true value **2.36**. Fix: patch only `model.py`'s namespace, leaving the internal delegation alone. Same family as the one-counter-two-places lesson in CLAUDE.md. A second bug surfaced with it: rerank passes FLOAT `labels` where proposal passes a BOOL `gold_mask`, so `&` raised and a swallowed exception zeroed every row | — | `tools/train/measure_logit_scale.py` |
| 2026-09-19 | **Events** | **absent-negatives A/B** (launch 3, commit `6e41923`) | do label negatives reach the RANKING loss, and does it matter? | 2× A100 us-east-1 `7dc4effa` / `db9b1d5f` | **KILLED at ~30% and RELAUNCHED** -- both arms had INERT negatives (sliding-window wiring gap), so the stated premise was false and it could not test its hypothesis; ~$13 spent. Gate readings before the kill: `absent_negatives_used` non-zero on 210 of 750 treatment steps (mean 0.464) and 0 of 227 control steps. Read the gate as a RATE -- ~75-80% of live treatment steps read zero, so a single line proves nothing. DOSE MEASURED: injection reaches 66.3% of records (0.73 labels/record, ~2.9 per batch) but only 0.366 absent queries per forward reach the denominator (9,159 over 25,000) -- ~87% are lost for reasons not yet explained, and label subsampling is NOT it. A null here reads as UNDER-DOSED, not as the mechanism refuted. Launches 1-2 died before training: `KeyError: 0` from iterating `ExtractorOutput` at the first logging step, then a relaunch onto a commit the failed push had never delivered. ~$0.50 | ~$65 | `OPTION_2_TYPED_ROLE_CONSTRAINTS.md` |
| 2026-09-19 | **Data** | cc_news_long annotation | a blind test set that actually reaches the sliding window | Batch API | **733 rows** of 743 (98.7%) after a retry at a raised output cap recovered 45 of 54; verbatim 93.7%, all five tasks; 1 document removed for overlapping cc_news_haiku45.train | ~$18 | — |
| 2026-09-19 | **Data** | cmnee + duee entity annotation | buy the entity gold that lifts option 2 past its 35.7% ceiling | Batch API | **DONE** — 26,343 records, verbatim 99.1%/99.4%. cmnee now yields **23 constrained roles covering 75.9% of its arguments**, having contributed ZERO before; combined coverage 68.8% | **$6.92** | `OPTION_2_TYPED_ROLE_CONSTRAINTS.md` |
| 2026-09-19 | **Infra** | label negatives never reached any sliding-window run | `negatives=` was passed at ONE of the two dataset branches from f89ebfd (2026-09-16) to 78dd190 | local | **FOUND + FIXED + GATED + CONFIRMED IN A LIVE RUN** (absent queries per forward 0.38 -> 4.11, 10.8x, both arms identical at 822/719); proven through `_prepare_data` on the real config: sliding_window=False gives the injector, True gives None. Hit absneg (both arms) and `eb16-eventrecords-neg`, whose verdict is now CONFOUNDED. A config declaring negatives whose dataset lacks the injector now refuses to start | — | `tests/training/test_negatives_wired.py` |
| 2026-09-19 | **Infra** | stop 4: box-side idle guard + three-valued probe | a box whose job never started had NO watchdog, and the recovery probe could not tell silence from an answer | local | BUILT; 4 guard cases pass, both probe branches run against live boxes, `pgrep` pattern path-anchored and confirmed to match a real runner | — | `tools/lambda/idle_guard.sh` |
| 2026-09-19 | **Infra** | absent-negatives in the listwise denominator | negatives reached NEITHER listwise loss — 0.6 of combined weight | local | BUILT, per-task, fail-closed; default path bit-identical; 9 tests, 2 of which fail without it | — | `OPTION_2_TYPED_ROLE_CONSTRAINTS.md` |
| 2026-09-20 | **Events** | **absent-negatives A/B** (`absneg`, asia-south-1) | do absent labels in the listwise denominator help, now that negatives actually reach the dataset? | 2× A100, ~17h | **CONFOUNDED — do not quote.** 19 metrics up, 0 down (event_argument +0.1657, event_trigger +0.2084) but the arms shipped **epoch 2 vs epoch 4** checkpoints: `metric_for_best: eval_loss` selects on a quantity the two objectives do not share. Uniform gains on `classification` (+0.0436) and `relation` (+0.0732), which the treatment cannot reach, are the tell. Needs re-running with selection on a task metric | ~$45 | `LABEL_NEGATIVES_PLAN.md` |
| 2026-09-21 | **Events** | **absent-negatives A/B, 3rd attempt** (`absneg2`, us-east-1) | do absent labels in the listwise denominator help, compared legally? | 2× A100, ~19h | **REAL VERDICT — the first legal comparison.** `event_argument` strict **+0.0376** (0.1840→0.2215), `classification` **−0.1977** (0.8538→0.6561); 3 up, 8 down outside the ±0.02 floor. Operating point verified IDENTICAL by the tool; both arms shipped their final checkpoint, same test set, treatment proven applied. Hits its target, breaks an untargeted head — not shippable as-is | ~$50 | `LABEL_NEGATIVES_PLAN.md` |
| 2026-09-21 | **EKF** | **option 4 roles A/B RE-RUN** (`roles2`) | does the derived corpus help once BOTH arms are regularised? | 2× A100 us-east-1, ~17h | **VOID — two confounds.** Arms shipped epoch 2 vs epoch 5 (`metric_for_best: eval_loss`), AND were scored on different test sets (20,602 vs 23,326) because the treatment's corpus adds its own test split. `compare_runs.py` REFUSED it — the provenance fix working. Also retracts the over-proposal DIAGNOSIS of the first run, which rested on the void entity row; that verdict's 7 other heads stand | ~$40 | `OPTION_4_ROLE_TO_ENTITY.md` |
| 2026-09-20 | **EKF** | **option 2 steps 1-4** (map, TypedRole, emission, decode A/B) | can a typed-role constraint be built, proven to fire, and does it help at DECODE time? | local, free | **BUILT + MEASURED**; map 174 roles/74 event types, reach **35.7% → 99.3%** via the document join; decode-only is a **null on F1** (0.0541→0.0540, 0.0713→0.0743) with precision up and recall down, 98% of drops were false positives. Emission from `_decode_joint` alone refused 918 edges and changed NOTHING — the record head overwrites them | — | `OPTION_2_TYPED_ROLE_CONSTRAINTS.md` |
| 2026-09-20 | **Infra** | negative pools derived from the config (`auto`) | a static pools file silently omits any newly added corpus | local | **BUILT + GATED**; `cmnee_roles_ner` was absent from the committed JSON so 300/300 of its records were `no_candidate`. Auto-derivation covers 18/18 and REFUSES to start on a gap. Ordering bug found in production: derive AFTER restoring corpora (~$0.90) | — | `tests/data/test_auto_negative_pools.py` |
| 2026-09-20 | **EKF** | **option 4 roles A/B** (`5afdc353` / `35adf5be`, commit `92c0605`) | does cmnee's arguments-as-entity-spans move the never-proposed third? | 2× A100 asia-south-1, ~16h | **NEGATIVE** — event_argument strict 0.3506→0.3206 (**−0.0300**), event_trigger strict 0.6057→0.5516 (**−0.0541**), 0 up / 10 down outside the ±0.02 floor, 9 inside. Both arms calibrated to threshold 0.3 on their own val sets at the same commit, so the precision-up/recall-down is the models, not the operating point. Option 4 joins 1 and 3 as refuted; only option 2 is untested | ~$65 | `OPTION_2_TYPED_ROLE_CONSTRAINTS.md` |
| 2026-09-18 | **Events** | **negatives blind-test rescore** | recover a blind test lost to a network outage, like-for-like | A100 | `event_argument` **+0.0322** under the FULL menu (invisible under gold) but `classification` **−0.1492**; both real vs measured floors. **NOT shippable** | $2.52 | `TODO.md` |
| 2026-09-18 | **Events** | cmnee roles→NER derivation | manufacture entity supervision where a corpus has none | local | BUILT as namespaced `Event<Role>`: claiming a ROLE not a TYPE keeps all 11 roles, 76,863 pairs, **0% lost** | — | `OPTION_4_ROLE_TO_ENTITY.md` |
| 2026-09-18 | **Events** | false-negative probe (3 mechanisms) | explain the classification collapse | local | **ALL THREE REFUTED** — denominator dilution, labels-as-input, false negatives (0.25–0.6%, not 35.8%). No defect needed | — | `TODO.md` |
| 2026-09-17 | **Infra** | **upstream PR #155 + issue #156** | bf16/fp16 clamp is a no-op → NaN grads with a finite forward; CJK splitter abort | local | Submitted to fastino-ai/GLiNER2 | — | — |
| 2026-09-15 | **Events** | **event-capable base** (eb16-eventrecords-tr) | first base whose record head trains on events | A100 | Trained; became the incumbent for every later event comparison | ~$28 | `EVENT_ARGUMENT_DIAGNOSIS.md` |
| 2026-09-15 | **Base** | cardinality A/B | does declaring scalar cardinality help structure? | 4 boxes | **NULL** — structure −0.0061, inside floor; precision up, recall down, they cancel | ~$8 | — |
| 2026-09-15 | **Events** | event threshold sweep | find the event operating point | A100 | Threshold buys **recall** (never-proposed 48.8%→29.4%) and **zero F1**; incumbent was never at 0.5 | — | `EVENT_ARGUMENT_DIAGNOSIS.md` |
| 2026-09-15 | **Infra** | `event_records` throughput | price the flag before committing | A100 | Costs **9%** (18.4→16.7 samples/s), not the 5× implied | — | — |
| 2026-09-08 | **Events** | **decode arms** (greedy vs joint) | close a −0.0454 structure deficit | 3 boxes | **87% of the deficit was a parameter nobody passed**; joint now matches greedy on all 7 heads | ~$6 | `JOINT_IE_DESIGN_RECORD.md` |
| 2026-09-07 | **Base** | fastino teacher re-scored | is the teacher better at its native window? | A100 | Verdict HOLDS — loses 6 of 8 heads, wins structure +0.081, relation +0.069 | — | — |
| 2026-09-06 | **Base** | record_metadata repair | price a data repair before doing it | local | Worth **+0.182 structure F1** and moves nothing else | — | — |
| 2026-09-06 | **Gates** | gate3 warm cells (3 seeds) | do no-replay warm cells forget? | GPU | **Catastrophic** — loses on all 8 heads, classification −0.403, for +0.024 on target. Later supplied the classification noise floor | — | `GATES.md` |
| 2026-09-03 | **EKF** | casualty_ml language-balanced | multilingual casualty extraction | A100 | Launch 1 died silently at step 0 (anchor defect); launch 2 trained | ~$14 | — |
| 2026-09-01 | **Base** | **137k v2 label unification** | train on the unified label space | A100 | Model on HF; **cross-version deltas CONFOUNDED** (the test set moved); relation −0.098 unexplained | ~$60 | `LABEL_SPACE_COLLAPSE.md` |
| 2026-09-01 | **Base** | 137k eb16 rebuild | make the label space the ONLY variable | A100 | Rebuilt at v1's effective batch 16 | ~$21 | `LABEL_SPACE_COLLAPSE.md` |

## merge/main-20260805 — August 2026

| date | line | experiment | purpose | where | outcome | cost | refs |
|---|---|---|---|---|---|---|---|
| 2026-08-29 | **Gates** | gate2 Turkish | can the gate read Turkish? | A10 | **BOTH BARS PASS** — Turkish AUC 0.4980→0.8105 on 9 outlets; English RMSE unchanged like-for-like | — | `GATES.md` |
| 2026-08-29 | **Multiling** | extractor vs Turkish (English control) | can the extractor read Turkish at all? | local | **PROVEN NO** — confident wrong figures (location is a digit 78.2% vs 5.8%, p=1.5e-39) | — | — |
| 2026-08-29 | **Data** | annotation purchase pricing | price annotation from the pool, not assumptions | local | A **free regex** composed with the gate beat the gate alone (78.8% vs 65.3%) and halved cost | — | — |
| 2026-08-28 | **Data** | Turkish adjudication pilot | buy a labelled Turkish region | Batch API | 5,638 docs, 2,387 positives (42.3%), on HF private | $4.12 | — |
| 2026-08-28 | **Infra** | DDP test on Linux | is the macOS DDP failure the code or the platform? | A10 | **Platform** — passes 2/2 on Linux; the skip guard cannot fire on Darwin | $0.47 | — |
| 2026-08-27 | **Gates** | gate2 v2 four-way | is v2 better than v1? | A10 | **INDISTINGUISHABLE** once thresholded on val (18/18, p=1.0). The lever was never training: ships at 0.5, needs 0.998 | $1.94 | `GATES.md` |
| 2026-08-27 | **Gates** | gate2 multilingual | train a multilingual gate | A10 | **NEGATIVE** — F1 1.0000 in-distribution, **0 of 71** real news admitted. Root cause: positives 99.9% synthetic | $1.10 | `GATES.md` |
| 2026-08-27 | **Gates** | gate swept vs fastino | which gate actually wins? | local | **The model we REPLACED wins** (AUC 0.9635); the switch decision had used no recall column | — | `GATES.md` |
| 2026-08-27 | **Gates** | two-task gate collapse | why did the gate admit nothing? | local | A 2nd classification task collapses a BOUNDARY gate to `other` 1.0 — a gate admitting nothing has a PERFECT FP rate, so it hid for two runs | — | `GATES.md` |
| 2026-08-26 | **Base** | mmBERT classification head | is the head dead? | local | **RECONCILED** — not dead: 100% on 5 labels, 97.5% on 9, collapsing above ~29 | — | — |
| 2026-08-26 | **Gates** | gate → casualty-docee | replace fastino as the relevance gate | local | FP on related=0 **34/410 → 1/410**; keeps are a strict subset | — | `GATES.md` |
| 2026-08-23 | **Data** | **real vs synthetic 2×2** | does mixing real and synthetic preserve better? | A100/A10 | The MIX preserved **WORST** (−38.6%), worse than either arm alone | $7.47 | — |
| 2026-08-23 | **Gates** | gates 3+4 eval | does the rebuild beat the incumbent? | A100 | **PASS** — beats it on all 8 strict heads; found the `_event_split` silent-truncation bug | $1.83 | `GATES.md` |
| 2026-08-23 | **Base** | record threshold sweep | find the structure operating point | A100 | 137k structure reference is **0.1119** swept to record threshold 0.1 | — | — |
| 2026-08-23 | **EKF** | mention-path instance keys | can a data mix fix EKF gate 2? | local | **NO** — two same-type events pool into ONE instance; unreachable by any data change | — | `EKF_MHT_DESIGN.md` |
| 2026-08-21 | **EKF** | **EKF front-end FULL** | rebuild the front end | A100 | **WORKED** — binding precision 67–100% vs incumbent 0–7.7%; beats it on all 8 heads. Two of our own instruments had to be fixed to see it | ~$35 | `EKF_MHT_BUILD_RECORD.md` |
| 2026-08-20 | **EKF** | casualty muting arm | does muting cross-event locations help? | A100/A10 | **SUPERSEDED** — a one-line per-event plausibility ceiling beats it | $2.35 | — |
| 2026-08-20 | **EKF** | EKF front-end smoke | price the full run before buying it | A100 | 18.5 samples/s; full run $35; `num_workers` refuted | $1.70 | — |
| 2026-08-19 | **EKF** | MHT re-priced | was MHT correctly rejected? | local | **NO** — the +0.055 that rejected it came from an oracle that cannot reject; real headroom +0.111 | — | `EKF_MHT_DESIGN.md` |
| 2026-08-19 | **Base** | **137k curve restart** | clean scaling curve on repaired data | A100 | 30% **exact replay IMPROVED** the base | ~$47 | `JOINT_IE_SCALING.md` |
| 2026-08-17 | **EKF** | Track A (clean re-run) | binding false positives | A100 | **POSITIVE** — binding FP 30.1%→9.6%; bf16 fixed the fp16 crash; loc-split saturates at epoch 1 | $3.00 | — |
| 2026-08-17 | **EKF** | Track B | is the formulation or the corpus the bottleneck? | A100 | **NEGATIVE** — in-domain 0.532 but **ZERO** on real news: the corpus is the bottleneck | $2.30 | — |
| 2026-08-17 | **Data** | cc_news 10K Haiku annotation | real-text half of the real/synth 2×2 | Batch API | 9,978 records, gate clean | $26.66 | — |
| 2026-08-15 | **Base** | clean re-baseline | does loss-weight reach matter? | A100 | **`scope=all` makes the event weight a REAL lever** (+0.013 event strict); cost is event_type −0.019. Produced new noise floors | ~$4 | — |
| 2026-08-15 | **Events** | event-loss sweep | is a flat event weight a lever? | A100 | **NULL** — it reaches 18.5% of the loss while events hold 1.6% of the gradient | — | — |
| 2026-08-15 | **Base** | GIST query-axis veto A/B | does a guide veto help? | A100 | **NEGATIVE on every metric**; entity −0.025 survives the floor, relation −0.033 does not | — | — |
| 2026-08-15 | **Events** | RAMS warm-start | does an intermediate mix_natural stage help? | A100 | **WASH**; the control shows the head-init curve does NOT turn at 100K | — | — |
| 2026-08-14 | **Data** | synthetic sanity fine-tune | what does synthetic-only cost? | A10 | Event 0.008→0.547 in-distribution but **−23% relative on general NER** | — | — |
| 2026-08-13 | **Data** | synthetic 5K Haiku batch | generate a synthetic corpus | Batch API | Corpus built; entity-coercion fix and clean rerun | ~$35 | — |
| 2026-08-13 | **Events** | MAVEN Tier 2 | do event records help MAVEN? | A10 | **Gained nothing** (trigger strict −0.008); the "+0.049 win" was a checkpoint-selection defect | — | — |
| 2026-08-12 | **Events** | CASIE Tier 2 | multi-instance events | A10 | Multi-instance **works** but scores 0.0036 vs control 0.2998 — head-init, record head never trained on events | — | — |
| 2026-08-12 | **Events** | RAMS base-word A/B/C | does lemmatising help arguments? | A10 | Lemma beats dup-control **+0.0119** strict argument — but PROVISIONAL (unswept threshold) | — | — |
| 2026-08-12 | **Base** | guide precompute (local) | precompute GIST guide scores | local | DONE 21.2h; use the DEDUP cache (194 conflicting sha1 keys) | — | — |
| 2026-08-11 | **Infra** | GIST precompute | does a GPU help this job? | A100 | **NO** — 3.9 vs 4.55 s/rec at 4% util: it is Python-bound. Buy the cheapest card | ~$10 | — |
| 2026-08-10 | **Base** | **joint_ie boundary scaling** | scaling curve on the boundary head | H100 | **NaN root cause SOLVED** (sdpa+bf16 on mmBERT); FA2 via `kernels` fixes it and is 11× faster | — | `JOINT_IE_SCALING.md` |
| 2026-08-10 | **EKF** | EKF/MHT pipeline | text→tracking, end to end | A100 | Validated held-out: EKF **0.291** vs ceiling 0.115 | ~$2 | `EKF_MHT_DESIGN.md` |
| 2026-08-06 | **Infra** | MPS vs CPU for events | is MPS worth using? | local | MPS is **3–4× SLOWER** for many-label event decode, ~28% faster for few-label classification. Workload-dependent | — | — |

## mmbert_training — July–5 August 2026

| date | line | experiment | purpose | where | outcome | cost | refs |
|---|---|---|---|---|---|---|---|
| 2026-08-05 | **Base** | **mmBERT head-init finding** | why is the RAMS argument gap so large? | A100/A10 | **Head-init is the bottleneck.** Data-scaling curve run (10k/40k/100k → arg 0.050/0.115/0.158), knee 10–40k, **no plateau**. Single-run variance ±0.02, so no point-to-point claim survives | — | `PAPER_0_FOUNDATION.md` |
| 2026-08-03 | **Base** | combined base + WikiEvents A/B | does combining help? | A100 | Complete; fed the head-init finding | ~$12 | `PAPER_0_FOUNDATION.md` |
| 2026-08-03 | **Data** | base-v1 synthetic sanity | fine-tune fastino base-v1 on synthetic only | A10 | Per-epoch reporting; pushed to HF | — | — |
| 2026-08-03 | **Infra** | main boundary-rewrite triage | is origin/main safe to merge? | local | **Naive merge unsafe** — a ~170-file boundary+joint_ie rewrite; deferred until after the running experiment | — | — |
| 2026-08-02 | **Multiling** | WikiANN multilingual | multilingual NER base | A10 | Model public at `whr778/gliner2-multi-v1-wikiann`; retained as the cu128 + push/terminate runbook | — | — |
| 2026-07-21 | **Infra** | merge main → DDP decision | whose DDP survives the merge? | 2× A10G | Kept **our** DDP, adapted main's tests; validated on 2× A10G | — | — |
| 2026-07-20 | **Events** | **AWS event-training run** | first multi-model event training | AWS EC2 | 7 of 10 models trained and pushed; retained as the AWS runbook | $1.86 | — |

---

## Standing lessons that came out of these runs

Not experiments, but the instrument and infrastructure findings each one was paid for. They
are why later runs cost less.

| lesson | what it cost to learn |
|---|---|
| **A gate must be able to fail** | a form gate scored best-over-range inverted a verdict for a day; a gate admitting nothing scored a PERFECT FP rate and hid for two runs |
| **Prove the treatment applied, from inside the run** | two A/B arms whose caps both exceeded the corpus ran identical data, and their identical failure read as reproduction |
| **A delta is not a result without a noise floor** | relations carry ±0.041, single runs ±0.02 — quoted floors now precede every verdict |
| **Pick on validation, score the blind test once** | a "+0.049 win" turned out to be a checkpoint-selection defect |
| **Never compare across a changed test set** | the 137k v2 label-unification deltas are permanently confounded |
| **Measure a property before classifying on it** | `candidate_pool` was assumed structural for months and blocked a real experiment; building it both ways showed 340 identical tensors |
| **Publish metrics before the model** | metrics are the finding and are kilobytes; a 16-hour run once lost its only copy to a network outage |
| **Identify a checkpoint by sha256, never name or size** | all three eb16 checkpoints are exactly 1,257,024,736 bytes; a probe ran on the wrong one and its finding was mis-attributed |
| **An unregistered corpus kills a run, not just a backup** | `_fetch_corpus` returns silently without an `hf_jsonl`; two A100s died in the data phase |
| **A dropped SSH is not a dead job** | a laptop network blip terminated two A100s 20 minutes into training |
