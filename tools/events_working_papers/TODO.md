# Open items

**Only outstanding work lives here.** Everything resolved, refuted or superseded is in
[[EXPERIMENT_CATALOG]] (what was run, newest first) and [[PROJECT_HISTORY]]
(narrative, including [the full pre-2026-09-21 TODO](PROJECT_HISTORY.md#todo-archive-2026-09-21)
preserved verbatim). Last rewritten 2026-09-21.

## Outstanding, at a glance

| # | item | why it matters | state | next action |
|---|---|---|---|---|
| 21 | **eb18-balanced: English event supervision + reach + capacity + global_decode** | eb17's English arguments score 0.0000 because they are under-learned (17% of argument gold, one 798-doc corpus) and 78.5% of gold is unreachable at 16 starts per query | **CONFIG BUILT 2026-09-30 (7149c2a), NOT LAUNCHABLE.** Pending: P1 cc_news full-menu purchase (pilot `msgbatch_01134CegRtA7QfdcbxPZkC34` running) + capped synthetic slice; P2 recall-at-precision>=X selection (X unset, code missing); P3 validation-window double counting (untraced); P4 label gate on the new corpora; P5 GPU smoke of the 768 budget; P6 Hub persistence + registry for the purchase | close P1-P6 in order, then launch; after training, read argument recall on CASIE's TRAINING docs to tell data from defect |
| 1 | ~~**absneg4: absent pool scoped to event roles**~~ **CLOSED -- NEGATIVE** | the last hope for the only lever that ever moved `event_argument` | **DONE 2026-09-22, ~$34.** Scoping LOSES the gain (`event_argument` -0.1027 vs absneg2-treatment, recall 0.1481 -> 0.0681, below control) while recovering only 15% of the classification collapse | none. The lever is dead: the gain depends on the unscoped pool and the -0.1977 classification cost is inseparable from it |
| 2 | **eb17-best: the warm-start base** | folds in every measured lesson; the base everything warm-starts from | **UNBLOCKED 2026-09-22 — the crash is FOUND and FIXED.** `validate_data: true` made `sanitize()` strip a structure field the `record_metadata` anchor named, leaving a declaration pointing at nothing. eb17 is the first base to set that flag, which is why only it crashed. `sanitize()` is now metadata-aware | relaunch. Note it now also gets: corpora in sync (60-label docee), chunked records carrying `record_metadata`, `negative_pools: auto` (was leaving 23.6% of records with none), `partial_annotation` actually reaching the trainer, and an LR that anneals to 0 |
| 3 | **classification collapses under interventions it never targets** | −0.1977 on absneg2, −0.403 on warm cells, −0.1492 before that; blocks shipping negatives | ROOT CAUSE NARROWED: 96% is `docee_event` alone | see whether (1) spares it; else isolate docee |
| 4 | ~~**`record_anchor_threshold` defaults to 0.5**~~ **CONFIRMED + REACHABLE; the gate is the TRIGGER score, and it IS trained (verified 2026-10-02)** | arguments and instances decode through the record gate | `record_anchor_threshold_wins` (3117569) makes the gate reachable; eb17 blind test 0.1752 -> 0.2331 at 0.1. **CORRECTION of 142de4d** ("the gate we tune is the object head: untrained"): in `natural` mode `forward_group` sets `object_logits = anchor_logits`, the anchor query's `candidates.pair_logits` -- the TRIGGER span score; `object_head` is used only in `anchorless`. `object_loss = zero` there because no separate head exists, not because the gate is unsupervised: boundary_preprocessing adds a MentionTarget for every field of every instance, the trigger included, and a real eb18 batch with event_records on shows the trigger query carrying its gold spans (3 for 'races'). Every event in cmnee/casie/cc_news_events/rams_merged/wikievents has a trigger (100.0%). So the 0.83 AUC measured a TRAINED score, and 'train the object head' (option A) is unnecessary | the record gate IS the trigger threshold: tune it with the argument gate (#22) as a 2-D sweep on VALIDATION on the final base; the lever beneath it is trigger detection itself |
| 5 | **Cross-event contamination** | 6 of 86 audited Helene `dead` observations belong to OTHER events | **keying root cause FIXED at source; both anchors turn out to ALREADY EXIST as `--scope-filter`/`--event-year`; the cached artefact is IRREPRODUCIBLE** | do NOT overwrite the cache; re-score `--scope-filter` on corrected labels |
| 6 | ~~`candidate_pool: shared` A/B~~ **CLOSED — REFUTED** | at 2 epochs entity −0.0745; at **4 epochs still −0.0435 against a 2-epoch control**. Needs 2x the compute and still loses the biggest head | **DONE 2026-09-21**, ~$6, all 3 arms, gates discriminated | — **NOTE 2026-09-30:** #180 (dense record loss diluted by padding) was live in the shared arms and is now fixed locally (426a38d), so those arms' EVENT deltas are confounded. The entity verdict stands (the entity head does not use the record loss). A rerun could only move the event heads |
| 7 | ~~Purchased NER gold idle~~ **DONE** | swapped into eb17-best; train entity mentions 707,007 → 905,691 (**+28.1%**) with the blind test byte-identical | **DONE 2026-09-21** | — |
| 8 | ~~`turkish_event` lemmatised gold~~ **DONE** | train+val 93.02% → **99.83%** aligned, **9,648 mentions recovered** | **DONE + PUSHED TO HF 2026-09-21** — this was a BLOCKER, not housekeeping: a fresh box fetches corpora from HF, so the repair would never have reached a GPU run | — |
| 9 | **Relation warm-start regression (−0.037, −22%)** | `task_lr: 5.0e-4` was tuned for COLD heads | hypothesis unverified | try a lower task_lr on the warm stage |
| 10 | **EKF: §10 crux reopened, §14 does not reproduce** | the EKF's claimed edge under unreliability | open | re-derive on real streams |
| 11 | **EKF: exposure counts are not casualties** | 12 of Helene's 106 `dead` are audited non-casualty; schema has nowhere to put them | **THE LABELS ARE MACHINE-READABLE and durable** (verified 2026-09-29): `tools/ekf_showcase/helene_audit_labels.json`, 86 per-occurrence labels — **helene 64, non-casualty 11, cross-event 6, unclear 5** — with `label/value/span/n_obs/why/value_error`. Keyed on `sha1(value\|context)[:16]` **so a label survives re-extraction**, and it replaced a `(span, value)` string match that was only **27% correct on its own positive class** | extend the schema. This is a PREREQUISITE for item 12's attribution scoring: 11 of the 86 are observations the schema cannot express, so they cannot be scored right or wrong |
| 12 | ~~**No benchmark that can score the filter**~~ **PREMISE STALE — REWRITTEN 2026-09-29. The non-oracle benchmark EXISTS and the filter has been scored on it.** What is open is the NEXT MECHANISM | Türkiye's baseline was an oracle *because truth and feed share a source* — `score_helene.py` says it outright: "the truth was read from the sentence the extractor reads, and the baseline scored **0.000**". Helene fixes that: truth is Wikipedia's casualty table, the feed is AP prose, so `est_last_value` no longer reproduces truth by construction | **SCORED, and the result is that ASSOCIATION IS AT ITS CEILING**: scope gate takes pooled 314.5 → **29.3 deaths**, a random-removal control of the same size gives 244.7 (so the gate SELECTS), and the ground-truth oracle also gives **29.3** — headroom for better association **−0.1 deaths**. The residual is elsewhere: `NC_RESIDUAL.md` prices North Carolina at 52.4 achieved vs a **23.0 oracle floor** — **half of it is unreachable from this feed**, 25 deaths never appear in it and the largest figure published is 98 against truth 123. No extractor recovers a number nobody printed | **The next mechanism is a LOWER SCOPE BOUND** — reject a figure too SMALL to be the place's own total, the mirror of the gate that already works. The surviving NC errors are sub-state figures filed at state level (`"one"` as the state total, `"three"`, `"dozens"`→32). The innovation gate is a crude proxy and COSTS Tennessee (14.4 → 24.7 deaths) to get it. Rank this work in DEATHS against each stream's ceiling, not in nRMSE: Tennessee has the worst nRMSE (0.817) and the smallest absolute error |
| 14 | **docee trained at a 59-label menu on every GPU run** | `docee_event` is **96% of the classification collapse** (item 3), and until today docee offered 59 labels (no `none`) while docee_zh and turkish_event offered 60 -- for the SAME task, inside one run | **FOUND + FIXED 2026-09-22.** HF and local are now identical; the menu gate resolves the trainer's real file list and fails on the shipped data (`sizes=[59,60,60]`) | re-read item 3 against this: the collapse may be a menu inconsistency, not an intervention side-effect |
| 15 | **1.34% of gold surfaces never align** | a mention that misses the tokenized path is dropped with no error; `surface in text` reports these corpora as clean | **MEASURED 2026-09-22** with `tools/data/measure_surface_alignment.py`. bio_ner_relations 4.77%, paraloq_json 3.31%, chfinann 3.03%, biored 2.56% | repair only the EXTENDABLE 8.2%; SUBTOKEN must NOT be extended (changes the referent); TOKENIZATION 37.7% is a splitter question, and it is the largest bucket |
| 17 | **Upstream main has 4 fixes we do not have — DEFERRED, not rejected** | two touch the record head and eval-loss, which are exactly what eb17 exercises | **ANALYSED 2026-09-22, deliberately not merged.** Merge base `3c913c7` (PR #141); 12 commits since, 7 substantive, one of them OURS (`9d54ecb`, PR #155, already here). Cherry-pick tested: `e1007d4` (classification decoding on node-budget exhaustion) and `d9b26b2` (choice/enum record fields collapsing to one document-wide answer) apply **CLEAN**; `f619651` (TypeError in `build_boundary_batch_metadata` for fallback records under eval-loss -- a fabricated `(0,0)` mention, and **we run eval-loss**) and `a7a69c3` (warn on fallback substitution) each conflict in **ONE trivial hunk** -- the `_create_fallback_record` docstring plus a ruff one-lining of `dummy_tokens`; resolution is take-theirs, and they must be applied in order since the second rewrites the first's docstring | Take all four as cherry-picks AFTER eb17 lands -- not a merge. **SKIP `7b400b7`**, a pure ruff pass touching runtime.py (625), processor.py (407), schema.py (93): ~1,100 lines of reformatting against files we rewrote, for zero behaviour. Also read `a7a69c3`'s OPEN ITEM: it says the fallback record omits `schema_special_positions`, so the layout declares a query the encoder builds no marker for, and the record is loss-neutral only BY ACCIDENT -- relevant because `error_policy: skip` is what creates fallback records in our configs |
| 16 | **EVENT ARGUMENTS: does the binding objective we ALREADY have work?** | strict argument F1 requires the argument to attach to the right trigger; we train that binding through the record head (`record_loss_weight` 1.0, instances seeded from the ANCHOR) | **STEPS 1-2 DONE, row was stale (re-verified in code 2026-10-01).** (1) Arg-C shipped as `eval_argc_external_*` (`56de584`; Trig-I/Trig-C/Arg-I in `cd8def7`), quarantined from the `event_*` heads. (2) ANSWERED on eb17, an `event_records: true` base (EVENT_ARGUMENT_DIAGNOSIS O2): strict argument F1 **0.1752** (0.2331 at anchor 0.1) vs the incumbent's 0.0991 -- **yes for Chinese** (CMNEE 0.2019, 88.4% of the pool), **no for English** (WikiEvents, CASIE, RAMS 0.0000), diagnosed as UNDER-LEARNING (S15), which eb18 (#21) is built to fix | **(3) waits on eb18:** read strict argument F1 PER CORPUS off eb18's blind test. ONLY IF English is still at the floor with balanced supervision, consider a pairwise role objective (the pair set is quadratic over candidates) |
| 22 | ~~**ARGUMENT threshold: make `record_field_threshold` reachable**~~ **BUILT 2026-10-02** | arguments were recall-starved at every measured operating point, and the one record knob moved instances and arguments together | `record_field_threshold_wins` (opt-in) routes all five field-gate sites (decode_group, the absent-able field filter, both choice-field calls) through `record_field_threshold`; anchor/object stay on the record gate. Eval-time key; `eval.py --record-field-threshold`. **Traced on eb18, 24 real val docs:** default and anchor-0.1 arms BIT-IDENTICAL to before; argument gate 0.1/0.05 keeps instances at 81 and trigger/type unchanged while argument F1 0.0678 -> 0.0916 / 0.1043 -- beats the old single knob (anchor 0.1: 0.0755, 231 instances, P 0.168). PROVISIONAL (eb18 under-trained, 24 docs) | sweep it per head on VALIDATION on the final base, then score the blind test once with a same-commit control |
| 18 | **Viewer: the model list was 50 behind -- DONE 2026-10-01** | it is how models are chosen, so a missing checkpoint is invisible rather than broken | `viewer/backend/sync_hub_models.py` adds every published GLiNER2 extractor (config.json `model_type: extractor`), keeps hand-written labels, reports but never deletes vanished entries. Run 2026-10-01: 125 on the Hub, 76 listed -> **49 added** (incl. eb17-best), 0 gone, 2 skipped -- `gliner2-eb16-composed` and `gliner2-warmstart-137k-realsynth-replay30-repaired` are EMPTY repos (only .gitattributes: created, never pushed) | re-run after each publish (eb18 when it lands); decide whether to delete the 2 empty repos |
| 19 | ~~**Viewer: one slider silently drives TWO gates**~~ **DONE 2026-10-02** | a user lowering `threshold` also loosened RECORD decode | The viewer has a **record (instance) gate** and an **argument gate**, each unchecked = follows the threshold, pre-set from the checkpoint's own `boundary_head` when it carries `*_wins`. Applied per request under a lock and restored after, so the cached model is never left changed. Traced over HTTP on eb18 x a CMNEE val doc: defaults (5 instances, 8 args) -> argument 0.05 (5, 10) -> defaults (5, 8) | the browser click path is type-checked, not clicked |
| 20 | **Viewer: structure fields came back empty -- RESOLVED 2026-10-01, NOT A DEFECT** | an empty structure reads as "the model found nothing" | **MEASURED through the viewer's own `extract()`** on casualty_docee val#9: `gliner2-casualty-multilingual` (trained on the structure) fills dead "2" / injured "26" at 0.5, 0.3 and 0.1, matching the text; `eb17-best` (never trained on casualty data) returns `{}` at 0.5/0.3 and junk at 0.1. An empty structure means the model does not know that schema. Two side observations, ONE document each, not yet claims: the doc's gold lists 2 of its 4 casualty reports (Paris 2/26 and Phachi 20 absent), and the training metadata anchors on `injured` where the viewer preset anchors on `dead` **GOLD GAP MEASURED 2026-10-01** (`tools/data/measure_casualty_gold_coverage.py`): of casualty_docee train's 34,635 paragraphs (one snippet each), 72.4% carry gold, **9.1% lose it to `_locate_in_slice` collisions** (a toll stated twice, e.g. "20 killed ... death toll of 20", has 2 hits and is dropped by design), and **14.0% (4,833) state a UNIQUE numeric toll with no gold** -- 1,570 of them in the FIRST paragraph, which the builder never mutes; val/test match (31-32% ungolded). Cost: training teaches those real tolls as negatives, and eval's DEFAULT gold menu (`_schema_from_gold`, fields = union of the doc's gold fields) still queries them, so a correct reading scores as FP. NOT eb18 (does not train casualty_docee); IS casualty-docee (the production EKF extractor), casualty-loc-split, loc-control. **ROOT CAUSE PROVEN 2026-10-01** (`tools/data/trace_casualty_gold_gap.py`, every paragraph traced to its `disaster_streams_docee250` source `gt`): casualty_docee was built at `9b2764e` (2026-08-09 19:37), **18 minutes BEFORE** `c5641a4` added `_standalone`; the pre-fix `_locate_in_slice` matched raw substrings, so small tolls collided with dates ("2"/"20" inside "2026") and were dropped. Replaying the pre-fix locator predicts record presence on 34,629/34,635 paragraphs (99.98%) and explains **all 3,756** unexplained ones; muting is ruled out (it arrived 2026-08-12, after the build). The builder is correct today; the CORPUS was never rebuilt. Smaller, separate: the realizer broke its exact-digits rule on 4.2% ("a couple" for 2, "four", "dozens") | **REBUILT 2026-10-01 (v2, Hub `6b09fd3`; v1 kept at `0ac8531`)** by `tools/data/rebuild_casualty_docee.py`, args recovered by reproducing v1 byte-for-byte (157 streams, 125/16/16, max_interference 3, seed 42). Unexplained paragraphs 3,756 -> **0**; coverage 72.7% -> 85.0%; 47 docs whose ONLY gold was read out of a larger number removed and 105 such fields dropped (all verified non-standalone); 0 real gold lost. Splits disjoint, 0 duplicates; pushed, hash-verified, restore-tested. **NEXT: re-score the incumbent casualty-docee on v2's test before any comparison** (v1 scores are void against v2); the realizer word-number gap (4.1%) remains |
| 13 | ~~**Chunking silently drops `record_metadata`**~~ **CLOSED -- FIXED 2026-09-22 (`aedcbb2`), row was stale** | every chunked long document lost record supervision silently | **RE-VERIFIED 2026-10-01 on real data** at eb18's window (4096 subwords, stride 3072): text2json 7,754 records -> 7,891 chunks, **0** without `record_metadata`; the same count on `aedcbb2^` gives **235** (the gate bites, and matches the fix's own measurement). casualty_docee does not split at this window. `aedcbb2` is an ancestor of eb18's `40b15e0`, so eb18 trains with the fix | none |

---

## 1. In flight

**absneg4 (`absneg4-roles.yaml`)** — `absent_negatives_scope: roles` excludes the event ANCHOR
query from the absent pool. absneg2 showed the two halves of the `events` bucket move in
opposite directions: `event_argument` recall +0.0329 against `event_type` recall −0.0818 and
`event_trigger` −0.0468. The Ortmann decomposition says the argument gain is **754 gold
arguments leaving "never found"**, which happened DESPITE fewer instances — so it does not
depend on the suppression.

Read it against **both** absneg2 arms (same operating point, same 20,602-record test set):
vs `absneg2-control`, is pooling better than none? vs `absneg2-treatment`, did scoping remove
the collateral? **A null is informative** — it would mean the argument gain did depend on
instance suppression, and the lever is dead.

## 2. Ready to launch

**eb17-best** (`tools/train/config/base/eb17-best.yaml`) — vanilla mmBERT → the warm-start
base. Four changes against `eb16-eventrecords-tr`, each forced by a measurement and all
documented in the file: task-metric selection instead of `eval_loss`, threshold 0.3, the split
gate on, and **label negatives added**. That last is the substantive one: no base this project
has trained has ever had a single absent query (0 in 574 measured), so `null_loss`,
`count_loss`, `negative_query_ratio` and `abstention_loss`'s entire positive class have been
supervised on an empty set in every model to date.

**LAUNCHED 2026-09-21 20:31 UTC without waiting for absneg4, deliberately.** The earlier plan
held it back because `absent_negatives_in_denominator` is a base-level decision — but that
setting is ABSENT from eb17 (defaults off), so absneg4's verdict cannot invalidate it either
way. If absneg4 is positive, eb17 becomes the clean one-variable CONTROL for a follow-up;
baking a setting that cost classification −0.1977 into a general base is the aggressive
choice, not the safe one. Waiting would have cost a full day.

**On 1x H100 PCIe (us-west-3), not an A100** — `gpu_1x_a100_sxm4` had no capacity, and H100
PCIe at $3.29/hr is roughly cost-neutral (~$29 against ~$28) for ~5h less wall clock.
`batch_size` 4 → 8 with grad-accum 4 → 2, so the EFFECTIVE batch stays 16 and the recipe is
unchanged — `batch_size` is PER-GPU, the one axis where a card swap silently becomes a
different experiment. `num_workers` 0 → 4 (the 0 was an MPS constraint, not a CUDA one) and
`pin_memory` follows.

**WATCH FOR THIS AND DO NOT MISREAD IT:** `__getitem__` runs in FORKED workers, so the
injector's counters live in the child and `composition_line()` reads the parent's. A
`[composition] ... 0/0 records` line under `num_workers > 0` is an artefact of forking, **NOT**
evidence that negatives failed. The guard that still works is `ExtractorTrainer._wired`, which
refuses to start when negatives are configured and the dataset carries no injector — it runs
in the parent at dataset construction.

Selection is on `eval_overall_strict_head_min_f1`, the first run to use it.

**Selection now has an aggregate.** `eval_overall_{strict,relaxed}_{micro_f1, head_macro_f1,
head_min_f1}` landed 2026-09-21. `head_min_f1` is the interesting one for a base: it cannot be
improved by trading one head away, which is the failure mode this programme keeps hitting.
Consider it for `metric_for_best` in place of entity F1.

## 3. P0 — blocks the next experiment

**`record_anchor_threshold` defaults to 0.5 and nothing calibrates it.** The record threshold
itself was swept and moved the 137k structure reference to 0.1119 at 0.1; the anchor cutoff
never got the same treatment, and it gates whether a record FORMS at all.

**NO BOX IS NEEDED: the sweep already exists** as
`tools/ekf_showcase/record_threshold_sweep.py`, and it was run 2026-08-20 on
`joint-boundary-mmbert-137k-clean` with a decisive result -- **1 of 40 windows yields a
matching record at the default 0.5, against 37 of 40 at 0.10.** So the entry's premise
("nothing calibrates record cutoffs") was stale, and two earlier drafts of it -- a $2 box,
then "once absneg4 frees the queue" -- were both wrong. Lambda boxes are independent and
self-terminating, so nothing frees a slot either.

**RUN 2026-09-21 on `eb16-eventrecords-tr`: FLAT.** 1 record at every threshold from 0.50
down to 0.01, 0 span-matched. That is NOT a threshold finding -- the probe's casualty/structure
schema does not engage this checkpoint's record head, which is supervised on EVENTS
(`event_records: true`). The honest reading is that the instrument does not apply here, not
that 0.5 is fine.

**A REAL INSTRUMENT BUG WAS FOUND AND FIXED ALONG THE WAY, and it was NOT the cause.**
`model.boundary_settings` and `model.boundary_head.settings` start as the SAME frozen object,
so `dataclasses.replace` rebinds only the model's and leaves the head holding the original --
and the decode path reads the head's. Proven by trace: after replacing, the model read 0.01
while the head still read 0.5. The sweep now sets both. **Fixing it did not change the flat
result**, so both statements stand: the bug was real, and it was not what made this sweep
flat. Third one-of-two-places bug found today.

**NEXT:** sweep on eb17-best once it trains -- the threshold should be calibrated on the model
that will ship it -- or build a probe whose schema actually engages an `event_records` head.
PICK ON VALIDATION, SCORE THE BLIND TEST ONCE.
PICK ON VALIDATION, SCORE THE BLIND TEST ONCE -- sweeping on test and quoting the best is
fitting the test set, and a "+0.049 win" on this programme already turned out to be exactly
that.

### Cross-event contamination — not blocked, and the readout is already fixed

Whole-article reading via `extract_long` binds casualty figures lifted from unrelated stories
sharing an article body.

**TWO STALE BLOCKERS, both cleared.** The entry read "Gated by item 0 (2026-08-17)" -- item 0
CLOSED 2026-08-19. The 2026-09-21 rewrite additionally mis-cited it as "item 4". And "fix the
unsound readout first" was done twice over, in `tools/ekf_showcase/event_binding_probe.py`:

- **2026-08-17, the scorer.** Signal C scored `bool(bound) and not OURS.search(bound)`, so it
  fired on any non-empty string lacking "helene". A casualty-trained record head has no
  `event` field in distribution and copies the anchor number in, so `'230'` counted as a
  caught cross-event and C read 9/11 on pure artifact. Now `validate_binding` keeps a binding
  only if it names an event the schema ALSO found, `_NUMERIC` rejects copied numbers, the
  `recs[0]` fallback is gone, and "bound nothing" is reported separately from "bound ours".
- **2026-08-20, the LABELS.** The ground truth itself was **27% correct on its own positive
  class** (3 of 11). All six `'230'`s were marked cross-event (Milton) when every one is
  Helene's OWN national total (truth 228, windows read "Helene death toll hits 230"); `'250'`
  likewise; one `'1,400'` is "1,400 LANDSLIDES", not people. It MISSED Maria's 3,000, the 1916
  hurricanes' 80 and a Taiwan typhoon's "dozens".

**CONSEQUENCE: every published score on this instrument is uninterpretable.** A detector
reading 3/11 might have found the three real cases or three of the eight false ones. That
includes the three signal results still quoted in `EKF_MHT_BUILD_RECORD.md` §27.2 (nearest
3/11 at 32.5% FP, only-competitor 3/11 at 31.3%, bound 2/11 at 26.5%) and the 2026-08-11
"4.7% cross-event of 106" figure. **Do not reuse either.**

**THE CORRECTED GROUND TRUTH** lives in `tools/ekf_showcase/helene_audit_labels.json`, keyed
by `sha1(value|normalised context)` so a label survives re-extraction. 86 occurrences:
64 helene, 11 non-casualty, **6 cross-event**, 5 unclear.

| span | belongs to |
|---|---|
| 3,000 | Hurricane Maria, Puerto Rico |
| 1,400 | Hurricane Katrina |
| 80 | the **1916** Appalachian hurricanes |
| dozens | Typhoon heading to Taiwan (injuries, not deaths) |
| at least 16 | Bosnia floods/landslides |
| at least two | Hurricane John, Mexico |

**EXTERNAL ANCHORS LOOK RIGHT, and the two are complementary.** Against the corrected six:
spatial (outside Helene's six-state footprint) catches 5/6, temporal (outside Sept 2024)
catches 3/6, **composed 6/6** -- the one spatial misses is the 1916 Appalachian hurricanes,
IN the footprint and 108 years early, which is exactly what temporal is for. This matches the
pattern that external-anchor checks work where text-proximity inference does not: the temporal
filter already took Izmit from 15 false bindings to 3 with zero genuine losses, while all
three text signals infer ownership from the words near the number and Helene articles
routinely name Milton and Katrina for comparison.

**CAVEAT, and it is the one that matters.** That 6/6 is a feasibility analysis over the places
and dates recorded in the audit's own `why` fields -- NOT an end-to-end run. A real filter must
EXTRACT place and date per observation, and that extraction can fail. **The false-positive rate
against the 64 genuine observations is unmeasured, and FP is precisely where all three prior
signals died** (3/11 recall at 26-32% FP). Recall on six cases proves nothing on its own.

### RE-RUN 2026-09-21, against the corrected labels -- the text signals are dominated

`whr778/gliner2-base-v1-casualty-docee`, 104 observations, audit classes
{cross-event 6, helene 82, non-casualty 12, unclear 4}:

| signal | catches cross-event | FP on genuine Helene |
|---|---|---|
| A nearest is a competitor | 3/6 | 19/82 = **23.2%** |
| B only a competitor named | 3/6 | 16/82 = **19.5%** |
| **C bound event is a competitor** | **0/6** | 1/82 = 1.2% |
| C-raw (unsound, diagnostic) | 3/6 | 32/82 = 39.0% |

**C IS DEAD.** Honestly scored it catches NOTHING -- for all six cross-event cases it either
bound nothing (3) or named something the schema never found (3). Its old 9/11 was pure
artifact, and the fix is confirmed working by the fact that C-raw still reads 3/6 at 39% FP
beside it.

**A AND B CAP AT 3/6, and the per-case detail says why:**

    '1,400'        nearest='Hurricane Katrina'    caught
    '3,000'        nearest='Hurricane Maria'      caught
    'at least two' nearest='John'                 caught
    'at least 16'  nearest='Hurricane Helene'     Bosnia is a PLACE, not a storm
    'dozens'       nearest='Hurricane Helene'     Taiwan typhoon, not named nearby
    '80'           nearest='None'  events=[]      1916 -- names no event at all

**THE DECIDING RESULT: A/B's three catches are a STRICT SUBSET of spatial's five.**
Katrina -> Louisiana, Maria -> Puerto Rico, John -> Mexico are all outside the six-state
footprint, so spatial gets those three PLUS Bosnia and Taiwan, and temporal gets the 1916
case. The text signals find less, at 19.5-23.2% false positives -- roughly one genuine
observation in five discarded. **They are dominated and should not be shipped.**

### SPATIAL ANCHOR BUILT 2026-09-21 -- `tools/ekf_showcase/spatial_anchor.py`

The pipeline ALREADY binds every observation to a place; that is what `event_key` is. So the
anchor needs no extraction and no model call: is that place inside the event's footprint? The
footprint and its 38 aliases come from the event's own `rollup.json`, not an invented
gazetteer -- `hierarchy.parts` is the six states, aliases map `asheville` to `north carolina`
and `carolinas` to `__aggregate__`.

| signal | catches cross-event | FP on genuine | model call |
|---|---|---|---|
| A nearest is a competitor | 3/6 | 23.2% | no |
| B only a competitor named | 3/6 | 19.5% | no |
| C bound event is a competitor | 0/6 | 1.2% | yes |
| **SPATIAL** | **4/6** | **1.2%** | **no** |

**Strictly better than A and B on BOTH axes**, and cheaper than C, which catches nothing.

    at least two   event_key=mexico                 FLAG
    3,000          event_key=puerto rico            FLAG
    at least 16    event_key=bosnia                 FLAG
    1,400          event_key=reading pennsylvania   FLAG   (Katrina)
    dozens         event_key=tennessee              pass   <- MIS-ASSOCIATED upstream
    80             event_key=north carolina         pass   <- 1916, TEMPORAL's job

**ABSTAINING IS LOAD-BEARING.** `event_key` is sometimes a TYPE (`Storm`, `Floods`) rather
than a place -- `collapse_type` territory. Those carry no spatial evidence and are passed,
never flagged; 5 genuine observations abstain this way. Flagging them would manufacture
exactly the false positives that make A and B unshippable.

**THE ONE FALSE POSITIVE IS ALSO AN UPSTREAM BUG**, not a filter error: a genuine Helene
figure of 180 is keyed `scotland`. Both remaining misses and the single FP are association
errors, so the ceiling here is set by the keying, not by the test over it.

### TEMPORAL ANCHOR ADDED -- one case, and the naive version does NOT work

| anchor | catches | FP on genuine |
|---|---|---|
| SPATIAL | 4/6 | 1/81 = 1.2% |
| TEMPORAL (year < 1950) | 1/6 | 0/81 = 0.0% |
| **COMBINED** | **5/6** | **1/81 = 1.2%** |

**The permissive version reproduces the A/B failure exactly.** 10 of 81 GENUINE Helene
observations carry a non-2024 year, essentially all in a comparative clause -- "Helene is
already the deadliest hurricane to hit the mainland U.S. since Katrina in 2005", "Helene
passed the 35 killed after Hurricane Hugo" (1989). **The year is attached to the COMPARISON,
not to the figure**, which is the same proximity-is-not-attachment problem that caps A and B:

    a year < 2010 in context    2/6    9/81 = 11.1% FP
    a year < 2000               1/6    2/81
    a year < 1950               1/6    0/81

Only an ANCIENT year survives. `temporal_flag` therefore returns True or ABSTAINS and never
False -- absence of an old year is not evidence a figure is current, so the signal can only
ADD to spatial, never overrule it.

**STATE THE EVIDENCE HONESTLY: this rests on ONE positive** (`'80'`, the 1916 Appalachian
hurricanes). The sharpness of the threshold -- 11.1% FP at 2010 against 0% at 1950 -- is
itself a fragility signal. It is a conservative complement to the spatial anchor, not a
validated signal in its own right, and it adds exactly one case.

**WHAT REMAINED WAS NOT A DETECTION PROBLEM -- and the cause is now found and fixed.**
The single miss and the single false positive were both upstream association errors, and they
share ONE mechanism: **AP embeds a rail of unrelated headlines INSIDE the story body**, and it
is flattened into the text with no separator.

    dozens   keyed `tennessee`  -- "RELATED COVERAGE Typhoon headed to Taiwan injures
                                    dozens ... 11 workers at a Tennessee factory ..."
    180      keyed `scotland`   -- "... paired at pro-am event in Scotland More than 180
                                    people have been killed from Hurricane Helene ..."

A genuine Helene figure took its place from a GOLF headline. The boundary is unrecoverable
downstream -- the headlines run together with no punctuation -- so the fix belongs at
HTML->text time, where the rail is a DOM node (`build_helene_feed.plain`).

**TWO TRAPS, both measured before committing:**

- **Strip by text and you lose more than you gain.** 74% of articles carry the marker, and
  11 of 106 `dead` observations sit within 400 chars after one -- 7 of them GENUINE. Dropping
  everything after the marker would discard 7 real observations to remove 2 cross-event ones.
- **`contains(@class,"Enhancement")` is too broad.** It also matches `LinkEnhancement`, an
  inline link inside the prose; stripping those deletes real article words ("Broadway",
  "assassination attempts"). Measured 26-48 nodes per article against 1-2 for the block
  selector. The committed xpath targets `@data-gtm-region="RELATED COVERAGE"` and
  `div[contains(@class,"PageListEnhancement")]` only.

**A SECOND LATENT DEFECT, found in passing: `lxml` was never declared.** Without it `plain()`
fell back to stripping every tag with NO structural selection, returning navigation, rails
and footer as article text -- a 26,598-character median document against 5,100, which is how
an article about a four-day workweek comes to "name" Florida and Georgia. It now RAISES
instead of silently producing a feed worth nothing, and lxml is a declared dependency.

**VERIFIED on the real article**: `RELATED COVERAGE`, `Scotland` and `Typhoon headed to
Taiwan` are all gone, and the 180 figure now sits in clean body prose.

### THE REBUILD WAS ATTEMPTED AND MUST NOT BE INSTALLED

The feed rebuilt cleanly and offline from cached HTML: 70 -> 65 rows, 375,609 -> 332,419
chars, and `RELATED COVERAGE` present in 52 rows -> **0**. **The 5 dropped articles are a
CORRECTION**, every one off-topic and qualifying only because the rail injected Helene states
and toll words into it -- "Georgia Supreme Court restores near-ban on abortions"
(has_toll=True, states=[NC, GA]), "Elon Musk makes first appearance at Trump rally"
(has_toll=True, states=[NC, SC]).

**But the pipeline re-run cannot be installed, and a CONTROL is what showed why:**

| run | dead obs | joined to audit labels | cross-event |
|---|---|---|---|
| committed artefact | 106 | 102 | **6** |
| control: OLD feed, current config | 47 | 27 | **0** |
| rebuilt: new feed, current config | 44 | 22 | **0** |

**The current configuration extracts NONE of the six cross-event observations -- from the old
feed either.** So the text change costs 3 observations (47 -> 44), not 62; the other 59 and
all six positives are lost to a configuration difference. Overwriting the cache would destroy
the basis for every cross-event measurement and orphan 80 of 86 hand-assigned audit labels.

**The committed `tracked_rollup.json` is IRREPRODUCIBLE.** It records only `feed`,
`associate` and counts -- no models, thresholds or flags. `run_pipeline` now writes an
`invocation` block (args, argv, git_commit) and the artefact I produced carries it, but the
committed one predates that. Same class as the `eval_provenance` defect, in a different
pipeline, and already fixed going forward.

### BOTH ANCHORS ALREADY EXISTED -- I duplicated them

    --scope-filter   "drop observations keyed outside the rollup's declared hierarchy.
                      Needs --rollup. 4/6 cross-event at 7.3% FP on Helene, no model."
    --event-year     "reject casualty figures whose nearest date predates this year ...
                      The 1999 Izmit toll was tracked as a 2023 figure in every Turkiye
                      configuration until this existed"

`--scope-filter` IS the spatial anchor; `--event-year` IS the temporal one, and better
engineered (nearest date to the span, with slack). I missed them by searching the probe files
and "association" rather than the pipeline's flag list.

**What survives as new work:** the `--scope-filter` figure of **7.3% FP predates the
2026-08-20 label correction**; re-scored against the corrected labels it is **4/6 at 1.2%**.
That is the same uninterpretability that voided the three text signals. `spatial_anchor.py`
is therefore worth keeping as a SCORER against the audit labels -- which no flag does -- but
its filter logic should defer to `--scope-filter` rather than reimplement it.

**NEXT:** re-score `--scope-filter` and `--event-year` in place, on an artefact that still
carries the six positives. Do NOT regenerate that artefact until the configuration which
produced 106 observations is identified.

31 tests.

## 4. Open experiments, cheap

- **`candidate_pool: shared` — CLOSED, REFUTED, 2026-09-21 (all 3 arms).**
  Operating point verified identical (threshold 0.3, test, 2,831 records, same box):

  | head | per_query | shared | delta |
  |---|---|---|---|
  | entity | 0.1788 | 0.1043 | **−0.0745** |
  | event_trigger | 0.7064 | 0.6769 | −0.0295 |
  | event_argument | 0.3537 | 0.3446 | −0.0091 |
  | event_type | 0.9748 | 0.9693 | −0.0054 |

  **0 up / 3 down.** The hypothesis was that `per_query` makes entity and event queries
  compete for candidate budget, so `shared` should RECOVER argument recall; measured, it makes
  `event_argument` slightly worse and `entity` much worse.

  **QUOTE THE ENTITY NUMBER ONLY.** Those floors were measured on the 18,786/20,602-record
  blind test and this probe scores 2,831 records -- roughly 2.7x the noise by support alone --
  so the trigger and argument deltas may sit inside a properly-scaled floor. Entity's −0.0745
  survives that rescaling; the others should not be claimed.

  **BOTH GATES DISCRIMINATED**, which is what attempt one could never do: treatment
  3.853e+00, control exactly 0.000e+00 ("control confirmed inert on the shared pool, as
  designed"). So this is a real reading of a live treatment, not an arm that failed to switch.

  **`shared-long` (4 epochs) answered the last question: more training does not rescue it.**

  | head | per_query (2ep) | shared (2ep) | shared-long (4ep) | 4ep vs 2ep control |
  |---|---|---|---|---|
  | entity | 0.1788 | 0.1043 | 0.1353 | **−0.0435** |
  | event_trigger | 0.7064 | 0.6769 | 0.6882 | −0.0182 |
  | event_argument | 0.3537 | 0.3446 | 0.3625 | +0.0087 |
  | event_type | 0.9748 | 0.9693 | 0.9765 | +0.0017 |

  Doubling the epochs recovers only part of the entity loss and **still trails a control with
  HALF the training** by −0.0435. The nominal gains on argument and type sit inside the floor
  once rescaled for a 2,831-record test set. Operating point verified identical on all three
  (threshold 0.3, test, 2,831 records, one box).

  **VERDICT: do not ship `shared`.** The hypothesis -- that `per_query` makes entity and event
  queries compete for candidate budget, so one document pool would recover the 30-35% of
  argument recall an entity menu costs -- is refuted. `per_query` stays the default, and the
  `shared_pool_builder` tensors that sit untrained in every checkpoint are confirmed to cost
  nothing worth recovering. **~$6 for a question open since a guard wrongly called
  `candidate_pool` structural.**
  **The gate has PASSED on arm 1: `shared-pool grad norm 3.853e+00`.** That matters more than
  it looks: `shared_pool_builder` exists in every checkpoint and receives NO gradient under
  `per_query`, so an arm that failed to switch is indistinguishable from one that switched and
  did nothing — attempt one died on exactly that confusion. A non-zero reading proves the
  treatment is live before any result is believed.

  Three arms (shared 2ep, perquery 2ep, shared-long 4ep), 1,260 steps per epoch at ~20 min,
  so ~2.7h and **~$6** — not the ~$4/1.5h first estimated. The control is re-run on the SAME
  box deliberately: attempt one's control was bf16 on an A10, so reusing it would confound
  card with treatment. Note the script's header comment claims it runs only the two `shared`
  arms; `ARMS` actually defaults to three. The code is right, the comment is stale.
- **Negative documents** (data, not decode) for cross-event contamination.
- **Base-word positive/negative samples** — a different granularity, aimed at noun-phrase
  boundaries rather than event interference.
- **A lower `task_lr` on warm stages** — 5.0e-4 was tuned for cold heads and the relation head
  regresses −0.037 in the warm start.

## 5. Data debt

- **The purchased NER gold: both reasons for leaving it idle turned out to be wrong**
  (checked 2026-09-21).

  **1. The flag already exists and is already in use.** `partial_annotation` is documented in
  `build_negative_pools.py` and live in roles2/roles3: a corpus declared partial for a
  dimension "contributes POSITIVES to the dimension while being refused as a source of
  NEGATIVES for it". Nothing needs building.

  **2. It is a corpus SWAP, not an addition, and the exhaustiveness bar is already met.**
  `cmnee_ner` IS `cmnee` plus entities -- the same 9,281 documents (99.9% overlap), the same
  19,422 events -- and `cmnee` currently contributes **ZERO** entity mentions. Same for
  `duee_ner`/`duee`. Swapping adds **167,315 entity mentions, +23.7%** of the base's entity
  supervision, already paid for at $6.92.

  Probed for exhaustiveness *within an offered label*, which is the only risk that matters
  because the menu is built from the record's own gold keys:

  | corpus | annotated/doc | missed/doc | missed as % of annotated |
  |---|---|---|---|
  | cmnee_ner | 13.0 | 4.23 | **32.4%** |
  | duee_ner | 4.1 | 0.29 | **7.1%** |
  | **docee (already used as supervision)** | 8.7 | 2.87 | **33.0%** |

  `cmnee_ner` is exactly as exhaustive as `docee`, which supplies 26.9% of the base's entity
  mentions today; `duee_ner` is 4.6x cleaner than both. So the bar excluding them is one the
  incumbent mix does not clear either. The probe is an UPPER BOUND -- a string match is not
  proof of the same entity in context, and this methodology once read 35.8% loose against
  0.25-0.6% tightened -- but the comparison between corpora is fair, being the same probe at
  the same settings.

  **Note the `entity_types` decision still stands and is different.** `merge_entity_types.py`
  writes a typing SIGNAL rather than an `entities` block because it attaches types to
  argument spans for the typed margin; that is not the same as using a corpus's own entity
  gold as supervision.
- **`turkish_event` REPAIRED 2026-09-21** (`tools/data/repair_turkish_surfaces.py`). The
  `max_len` diagnosis was wrong: the annotator lemmatised the LABELS but not the TEXT, so
  gold read `mesafe` where the document reads `mesafenin`. 234 of 235 failures were the gold
  surface being a prefix of a longer word, every failing position well inside the window,
  0 case-only, 0 genuinely absent.

  Train+val went **93.02% -> 99.83% aligned, 9,648 mentions recovered**; 88 derivational
  repairs were REFUSED and left for `skip` (`futbol` -> `futbolcunun` is football -> of the
  footballer). TEST DELIBERATELY UNTOUCHED and verified byte-identical -- repairing it would
  move the blind set, and would buy nothing, since those surfaces score zero either way.

  **`tools/data/augment_baseword.py` had already written down this exact failure**: "never
  lemmatize text and label strings in separate passes ... the passes diverge and the label
  stops matching", and "a mention it cannot find is silently dropped -- it quietly shrinks
  supervision and reads as 'augmentation did not help'." The RAMS base-word corpus avoids it
  by rebuilding text and labels from the SAME token list. That is also the dup-control
  experiment (lemma beat dup-control +0.0119 strict argument, PROVISIONAL).

  REMAINING: `data/turkish_event.{train,val}.jsonl` now differ from the HF mirror
  `whr778/turkish-event`, which still holds the unrepaired originals. Re-upload when convenient.

- **`biored` aligns at 97.4%** and only 8 of its 126 failures are the prefix pattern, so it has
  a DIFFERENT cause. Not yet diagnosed.

## 6. EKF / casualty line

- **§10's crux is reopened and §14 does not reproduce.** The harder-regime ablation concluded
  the EKF's edge *widens* under unreliability; that does not hold on real streams.
- **Exposure counts are not casualties** — 12 of Helene's 106 `dead` observations are audited
  non-casualty (six are exposure figures), and the schema has nowhere to put them.
- **No benchmark can score the filter.** Türkiye's baseline was an oracle by construction:
  truth read from the sentence the extractor read.
- **Regional disaster profiles as a plausibility prior** — an operator observation, recorded
  as an idea. The flag-not-veto mechanism it needs already exists twice and is switchable.
- **Do NOT buy scope labels under the current scheme** — settled by a $2.25 dual-label probe.

## 6b. Closed 2026-09-21 by a cheap local check

- **`biored`'s 97.4% alignment: DIAGNOSED, and NO ACTION.** 108 of 126 failures are
  "word-bounded but the tokenizer splits it differently" -- biomedical hyphenation, where the
  gold marks a SUB-TOKEN of a compound: `'mannose'` inside `'mannose-binding'`,
  `'Catecholamine'` inside `'Catecholamine-induced'`, `'-31T/C'` inside `'IL1B-31T/C'`. This is
  NOT the Turkish case and must not be repaired the same way -- extending `'mannose'` to
  `'mannose-binding'` changes a chemical into a binding property, exactly like the derivational
  repairs that were refused. Splitting on hyphens would change tokenisation globally. At 126 of
  4,927 mentions on a corpus that is 0.7% of the base, leave it.

- **Chunking units: the claim is CORRECT and the configs are now right.**
  `gliner2/training/chunking.py` line 26 states "Window / stride are measured in **subword
  tokens**", and the function is `chunk_text_by_subwords`. The "word window" comments the
  entry complained about have since been fixed -- every current config says subword.

- **Classification inheritance is REAL but INERT at the current window, so it does NOT explain
  the docee collapse.** `chunking.py:200` copies a document's `classifications` verbatim to
  every chunk with no check that the fragment supports the label, which is genuine injected
  label noise. But the +124.5% figure was measured at a 384/256 window on deberta. At eb17's
  4096-subword window, measured over 1,200 documents each:

  | corpus | median subwords | % chunked |
  |---|---|---|
  | **docee** | 646 | **0.2%** |
  | chfinann | 336 | 0.0% |
  | docfee | 1,198 | 6.0% |
  | docee_zh | 620 | 2.2% |

  docee is barely chunked at all, so this mechanism cannot be behind its −0.333. Recorded so
  nobody chases it. It WOULD matter again at a small window.

- **"Boundary beats span at 10K": the entry MIS-CITES it, and the claim is stronger than the
  entry suggests.** The live claim (`PROJECT_HISTORY` line 793) compares **0.177 against
  0.050** -- the span curve's 10K point -- not 0.158. That is +0.127, **6.4x the +/-0.02
  single-run variance**, and the same passage had already retired every marginal claim on that
  curve ("no point-to-point difference on it is interpretable"). It appears in PROJECT_HISTORY
  only, NOT in PAPER_0 or RESEARCH_PROGRAM, so it has not propagated.

  **What remains genuinely unverified is the same-test-set question**: 0.177 and 0.050 come
  from different experiments, and a support mismatch (3,527 against 20,845) invalidated a row
  of this very curve once. Re-derive on a shared test set before it goes near Paper 0 -- but
  it is not the marginal claim the entry described.

## 7. Research direction

- **Does a fine-tune need an explicit regularizer, or is early stopping enough?**
- **Chunking distorts the real-news arm** — `window_size`/`stride` are **subword tokens, not
  words**, and the configs' comments say "word window". Measured, and not where expected.
- **We cannot compare event arguments to the literature** because we do not compute the
  metric the literature reports.
- **RESUMED 2026-10-04 -- the trigger-anchored argument line** (briefly held for design C). Built, opt-in:
  `proposal_gold: identity` (p2fast null, right direction), `absent_reduction`, `record_negative_instances`
  (K=8: precision +0.12, recall -0.016), `record_role_hard_negatives` (p3arg: argument recall +0.016*,
  F1 +0.023* -- the eb19 base candidate). Next: eb19 recipe = eb18 + `record_role_hard_negatives: 8` (negatives
  retune TESTED 2026-10-04 p4neg: net NEGATIVE at K=2/4/8@0.25 -- dropped; false triggers 52-74% of wrong-trigger
  attachments remain open); decode-time pooling (boundary share
  14-26%) -- TESTED 2026-10-04, NEGATIVE, dropped; existence head ON HOLD (`EXISTENCE_HEAD_SPEC.md` section 5: argument evidence still AUC ~0.5).
- **DONE 2026-10-05 -- the junction layer (`JUNCTION_LAYER_SPEC.md`, built 21f357e, A/B p5link).** The trigger->argument
  score was role fit with a weak join (per-column junction AUC 0.58-0.63; pair term below chance in-row). The junction
  + column loss lifts it to 0.74 / 0.77 and gives argument F1 +0.035* (English arguments move for the first time).
  Going into eb19 at column weight 0.3.
- **PARKED -- design C (events as slots: anchorless records).** The structural fix for the shared root cause
  (event identity = a trigger mention). Found 2026-10-04: events are hard-wired natural
  (`processing/records.py` `_event_record_cfg`); anchorless = 32 learned slots, one attention layer,
  `object_head` + Hungarian matching on EVERY slot, trigger becomes an ordinary (list) field; eb18 never
  trained a single anchorless/latent group, so the slot heads are at random init. Anchorless was A/B'd ONCE
  (0ca9447, 2026-08-10, casualty STRUCTURES not events, 9-doc probe): natural 7/9 instances, anchorless
  1/9, cause never established. **Resume with a free diagnostic first:** switch to let events run
  anchorless, trace one real event batch (Hungarian assignment, object vs field loss, slot divergence),
  then a one-document two-event overfit on CPU. No GPU until that passes.
- **Matryoshka representations (added 2026-10-04, explore AFTER the argument fix lands).**
  [Matryoshka loss](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html)
  trains nested prefixes of an embedding (e.g. 768 -> 256 -> 64) so a truncated prefix still works.
  The owner's prior: nearly all of a full embedding's signal fits in 64 dimensions. Two possible
  payoffs here, to be measured separately:
  1. *Speed of queries.* Encode a document once, cache its span/candidate states at 64 dims, and
     score many label menus by cheap dot products (the "full taxonomy" deployment case). Note the
     encoder dominates compute, so truncating head dims alone buys little; the win is in reuse.
  2. *Signal concentration.* Forcing the label/query and candidate states into a shared 64-dim
     prefix may regularise the many-label heads.
  **Measure first, before training anything:** the effective rank (PCA spectrum) of eb18's query
  states and candidate states on real batches. If 64 components already hold ~all the variance,
  truncation may need little or no retraining; if not, Matryoshka training is the lever.
  Trigger: after event_argument is fixed, so an argument regression is not confounded with it.

## 8. Tracked, not dropped — the beam-aware / structured loss (Phase B)

**The argument that made this interesting no longer holds, and the entry used to claim the
opposite.** It read: *"Phase B is now MORE interesting, not less — every measurement of the
beam to date ran with its scalar constraints disconnected."* Two things have happened since:

1. **The disconnected constraints were fixed**, and the decode-arms run then measured that
   **joint matches greedy on all seven heads**. So the null is no longer "a null about a beam
   with its constraints off" — it is a null about a beam with them on.
2. **The strongest candidate constraint was refuted.** TypedRole has almost nothing to act on:
   the probability mass on type-disallowed candidates is median 0.00003 / mean 0.00284, so the
   model already ranks them near zero. See [[OPTION_2_TYPED_ROLE_CONSTRAINTS]].

**What survives is narrower and still real:** the beam decodes JOINTLY over candidate scores
trained GREEDILY, so nothing in training ever optimises for constraint satisfaction. That
train/test mismatch is a genuine structural argument for putting the beam in the loss. But it
is now the *only* argument, and the constraint it would enforce has been measured to be nearly
inert. Treat Phase B as **lower priority than it was**, not higher.
