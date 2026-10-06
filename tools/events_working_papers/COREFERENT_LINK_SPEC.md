# Learned trigger x trigger link: one event per coreferent cluster at decode

**Status:** BUILT 2026-10-06 (section 9), opt-in; eb20 trains it at the user's office (section 10). Gates 3-5 wait for the trained link. This is `COREFERENT_OWNERSHIP_SPEC.md` 3b-ii. It replaces
3b-i, the merge on shared high-scoring arguments, which traced NEGATIVE.

## 1. Why (measured)

`record_coreferent_ownership` (3a) trains every mention of an event to own its arguments. Without
a decode step that recognises mentions of ONE event, each mention emits its own copy, so 3a
cannot ship alone.

3b-i merged instances that scored a shared argument candidate. On 15 real multi-trigger val docs,
**0 of 6 merged events were a correct gold cluster.** The pairwise measure below explains why.

**Baseline separability**, on the real p5link-junc_w03 checkpoint and 40 real
cc_news_events_sonnet55_v2 train docs. Positives are mentions of the same gold event (355 pairs).
Hard negatives are mentions of different same-type events in the same doc (443 pairs).

| pairwise score | AUC |
|---|---|
| cosine of argument-score rows (what 3b-i used) | **0.468**, below chance |
| cosine of raw trigger-instance states | 0.685 |
| cosine of `inst_proj` states | **0.709** |

So arguments cannot tell mentions apart. The scores are dominated by the role term, which is
identical for every trigger. **Trigger states already carry coreference signal untrained.**

**Training signal available** (cc_news_events_sonnet55_v2 train, 4,597 docs):
- 9,528 positive mention pairs (3,760 multi-mention events);
- 15,622 hard-negative pairs (different same-type events in one doc, 2,838 doc x type groups).

These are the first coreference clusters this programme has had at volume. mendeley_ed's
multi-trigger lists are keyword sets, not coreference, and are excluded (section 3).

## 2. Architecture

New opt-in setting `record_coref_link: true`. The module `TriggerLink` is built ONLY when enabled.
This is the strict `load_state_dict` lesson from the junction (`enable_link()`): a module created
unconditionally breaks every existing checkpoint.

Within one record group (one event type), for instances i, j:

    s_ij = <W t_i, W t_j> / sqrt(d) + w . g(i, j) + b

- `t` is the instance state (the same tensor `_assign_logits` receives).
- `W` is hidden -> d (d = 128).
- `g` is the pair geometry: log1p of the token distance between the trigger spans, a same-sentence
  flag, and an exact-lemma flag (killed / killing).
- The score is symmetric by construction.
- It reads instance states only, so it cannot be fooled by the role term.

**No effect when off.** No other loss or score reads the link, so every other output stays
bit-identical whether it is on or off. Initialise W from `inst_proj`'s weights (truncated or
projected to d) so that the untrained link already sits near the 0.709 prior rather than at
chance. That is a measured property of the projection, not an assumption, and the build traces it.

## 3. Training: pairwise loss on gold clusters

Inside each natural event group with at least 2 gold records or at least 1 multi-mention record:

| pair | target |
|---|---|
| two seeded mentions of the SAME gold record | 1 |
| seeded mentions of DIFFERENT gold records (the hard negatives 3b-i fused) | 0 |
| a gold mention and each of the K hardest false trigger instances (`_negative_instances`, K = 4) | 0 |

The third row stops the link from pulling false triggers into real events. That was 5 of 3b-i's
6 errors ('Cancel', 'winner', 'earthquake' as one event).

- BCE, mean over pairs, its own weight `record_coref_link_weight`, and its own denominator. It
  never enters another term's mean (the `absent_reduction` rule).
- Records whose multi-trigger list is a keyword set (mendeley_ed) contribute no positives. Gate
  this on the corpus name via a `coref_link_corpora` allow-list, so the model never learns that
  "remote" and "anxieties" corefer.
- 3a (`record_coreferent_ownership`) is on in the same arm. The two are designed together.

## 4. Decode

New opt-in setting `record_merge_coreferent: link` (the existing boolean becomes a mode: `off`,
`args` for 3b-i, kept for the record, and `link`).

- Among selected same-type instances, cluster by single linkage on sigmoid(s_ij) >= tau, with
  union-find as in 3b-i.
- The representative is the strongest instance. Its row goes to exclusive allocation; members add
  their trigger spans. The 3b-i formatting fix already keeps every mention in `triggers`.
- **tau is picked on VALIDATION** (cc_news_events_sonnet55_v2 val, which carries voted gold
  clusters). The blind test is scored ONCE at the picked tau.

## 5. Metric: a cluster-aware event score

Today an event is scored by trigger key, so finding the event through its second mention is a
miss, and a correct merge cannot raise any number. Add keys; do not change existing ones:

- `event_cluster_*`: a predicted event matches a gold event when the type is equal and they share
  at least one trigger mention. Arguments are scored within matched events.
- Per corpus, as always. English and CMNEE separately.

The gate for the metric: on one constructed doc, a prediction found through a non-first mention
must score a HIT under `event_cluster_*` and a MISS under today's key. Show both before trusting
either.

## 6. Gates (each must be able to fail)

1. **Off is bit-identical:** with `record_coref_link` off, the losses and decode on real eb19
   batches are identical to today.
2. **The treatment applied:** an in-run log line counting positive and negative link pairs,
   emitted from inside the loss after logging exists, with backoff.
3. **The link learns coreference:** post-hoc pair AUC on VAL docs, the same measure as section 1,
   rises clearly above the 0.709 prior. Target >= 0.85.
4. **Merges are right:** on VAL, the precision of merged clusters against gold clusters is at
   least 0.8. 3b-i scored 0 of 6, so this is the gate that failed before.
5. **No distinct-event fusion:** the hard-negative pairs merged at tau stay under 5% on VAL.
6. **The metric can fail** (section 5).

## 7. Build order (trace, code, trace, test)

1. Trace on real batches the module's input states, the pair labels from gold clusters, and the
   decode clusters with the link at init.
2. Code TriggerLink (built only when enabled) and the pair loss with its allow-list. Then gates 1-2.
3. Code the decode `link` mode and the `event_cluster_*` metric. Then gates 4-6 on the init link,
   which must be at least as good as the 0.709 prior.
4. Trace again on the same inputs.
5. Tests, mutation-checked:
   - drop the hard negatives;
   - drop the false-trigger negatives;
   - sum instead of mean;
   - keyword-set positives leaking past the allow-list.
6. **Fast A/B** (p5link design: warm start, one shared draw, fixed threshold).
   - Base: eb19 when it lands.
   - Arms:
     - control (3a off, link off), run twice for the floor;
     - 3a + link + `link` decode.
   - Every arm includes cc_news_events_sonnet55_v2 train, so the data is identical and the
     treatment is the only difference.
   - Readouts:
     - `event_cluster_*` on sonnet55 val and test;
     - val pair AUC;
     - merge precision;
     - every other head (no regression outside the floor).

   Lambda, ~3 h, ~$20-25.

## 8. Open questions

- Single linkage can chain (A~B, B~C, but A!~C). If merge precision fails gate 4, use average
  linkage before anything else.
- Should the link also see the event type's embedding? Within one group the type is constant, so
  it adds nothing to the pair score. Revisit only for cross-type coreference, which is out of scope.
- Argument coreference ("Smith" ... "the CEO" ... "he") stays string-matched. The head-span rule
  gives the annotator's chosen surface; argument clusters are a separate, later problem.

## 9. Build results, 2026-10-06 (CPU; the A/B waits for eb19)

**Built (opt-in, default off):**
- `TriggerLink`: built only when `record_coref_link` is on, initialised from `inst_proj`, with
  `enable_coref_link()` for warm starts.
- `RecordGroupOutput.coref_logits`.
- The pair loss `_coref_link_loss` (own mean, `record_coref_link_weight`), with the in-run log line
  "coref link: N mention pairs trained ... pair AUC".
- The decode mode `record_merge_coreferent: link` (`record_coref_link_threshold`).
- The metric keys `event_cluster_*` and `event_cluster_argument_*`.

**Deviations from sections 2 and 3:**
- **Pair geometry is distance only.** The same-sentence and lemma flags need the document text,
  which the record head does not see.
- **The `coref_link_corpora` allow-list is not built.** Records carry no corpus name. For the A/B,
  mendeley_ed (whose multi-trigger lists are keyword sets, not coreference) is left OUT of the
  training data in EVERY arm, so the arms still differ only in the treatment.

**Gates, traced on real data:**

| gate | result |
|---|---|
| 1 | link off == the code before this change, every loss term; link on at weight 0 == off |
| 2 | at weight 0.3, 428 mention pairs trained over 21 groups on 8 real docs; logged in-run. Init pair AUC 0.811, but that includes the easier false-trigger negatives (the hard-negative-only prior is 0.709) |
| 6 | a 3-mention gold event found through its 2nd mention with every argument right scores `event_cluster` 1.0 / `event_cluster_argument` 1.0, while today's `event_argument` scores **0.0** (its key is the whole trigger set). A prediction sharing no mention scores 0.0 on both. |

**Gates 3-5 need a TRAINED link.** At init, with threshold 0.5 on 15 val docs, the link
over-merges: 61 -> 26 events, merged clusters right 2, fusing different events 3, absorbing
non-gold triggers 11 (precision 2/16 = 12.5%, against the 0.8 bar). At 0.99 and above it merges
nothing.

**Found and fixed:** `record_merge_coreferent: link` on a checkpoint WITHOUT the module, set
through eval overrides, merged nothing at every threshold with no error. Decode now raises.

**Finding for the existing metric.** Against coreferent gold, today's `event_argument` scores a
fully correct event found through a non-first mention as ALL misses. Once cc_news_events_sonnet55_v2
enters training and eval, read arguments on `event_cluster_argument_*`, not `event_argument_*`.

## 10. eb20 and inference, 2026-10-06

**eb20 trains 3a + the link, merge OFF** (`tools/train/config/base/eb20.yaml`, link weight 0.3, unmeasured),
on the user's office GPU instead of the Lambda fast A/B of section 7.6. That A/B is not run.

**Gate 2 in a full run.** `coref link: ... pair AUC` read **0.873** cumulative over ~50,000 ranked pairs
at 5,000 natural groups. At init it was 0.811 on the same mix, so the link is learning. This is NOT gate 3:
- it is training batches, cumulative from step 0, so it understates the current link;
- it includes the easier false-trigger negatives, so it overstates the hard-negative score.

Gate 3 is read on val, hard negatives only, against the 0.709 prior.

**Inference.** `tools/infer.py --merge-coreferent link --coref-threshold T` (ea6e5e5) applies the merge
through `apply_boundary_overrides`, the eval path. Traced on the init link:
- 28 events / 0 multi-trigger become 14 / 8;
- the merge survives global decode (45 events either way);
- the merge runs INSIDE each window, so mentions in different windows of a long document are not joined;
- `link` on a checkpoint without the module refuses.

Pick T on `cc_news_events_sonnet55_v2` val against gates 4-5, then score the blind test once (`tools/train/INFER.md`).

## 11. Calibration at the end of training, 2026-10-06

The threshold is picked on VAL once, after training, by `gliner2/training/coref_calibration.py`, through
either entry point:
- `train.py`, when `eval.coref_calibration` names corpora (eb20: `data/cc_news_events_sonnet55_v2`). It runs
  after the decision-threshold sweep and BEFORE the blind test, so the test scores the setting that ships;
- `tools/train/calibrate_coref_threshold.py --write`, for a checkpoint already trained.

**Grid:** 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, beside an `off` row.

**Eligible only if all four hold:**
- at least 20 merges;
- merge precision >= 0.8;
- fusion < 5%;
- `event_cluster` F1 >= off.

The best eligible threshold by `event_cluster` F1 is written into `best/config.json` (`record_merge_coreferent:
link` and `record_coref_link_threshold`). If none is eligible, the merge stays `off`. The whole table always
goes to `best/coref_threshold_sweep.json`. `infer.py` uses the stored setting unless its flags override it.

**Two defects found by the trace before anything was written** (p5link with the link at init, 40 sonnet55 val docs):
1. **The gate passed on ONE merge.** At 0.7, "merge precision 1.000" rested on a single merged event. That is
   the gate-that-admits-nothing failure, so a minimum of 20 merges is required.
2. **Merges of a repeated word were invisible.** Triggers are casefolded into a set, so `["meeting", "meeting"]`
   counts as one mention. At 0.7, 91 -> 79 events showed only 1 merge, and 5 once counted from the raw
   trigger list.

**Deviation from section 6, gate 5.** The spec's pair-based fusion (gold hard-negative pairs merged) had ONE
pair on 40 docs, so a single merge swung it between 0 and 1. The gate instead uses fusion = the share of
merges joining different gold events, which rests on the same >= 20 merges. The pair figure is reported as
`pair_fusion`, with its support.

**At init the calibration refuses, correctly:**
- 0.5: 24 merges, 8 right (33%);
- 0.6: 20 merges, 40% right;
- 0.7: 5 merges, 4 right, but below 20 merges;
- 0.8 and above: nothing merges.

So the merge stays `off`. A written `link 0.7` was shown to reload and to become `infer.py`'s no-flag default.

**Control set and second blind test (added 2026-10-06 at the user's request).** The calibration gold
(sonnet55) is the ONLY coreferent gold, but a shipped merge also decodes CMNEE, DuEE and MAVEN, whose gold is
one trigger per event. So:
- **A control.** `eval.coref_control` (eb20: `data/cmnee` + `data/maven` val; 0 multi-trigger events, 593 / 345
  docs repeat a type, 0 docs overlap eb20 train) is decoded at every threshold. Eligible only if control cluster
  and argument F1 >= off AND control fusion < 5%. Startup refuses a control that lists multi-trigger events.
- **Why fusion and not F1 alone (traced, init link, 20 CMNEE + 4 MAVEN val docs).** The cluster metric matches on
  ANY shared trigger, so a fused event collects the pooled arguments of several gold events. At 0.5, 30 of 62
  control merges were fusions, yet control argument F1 ROSE 0.2134 -> 0.2417 and event F1 0.5403 -> 0.5405. An
  F1-only control would have passed. The same effect lifted calibration F1 0.364 -> 0.402 at 33% merge
  precision: `F1 >= off` is a guard, not evidence. The merge gates carry the decision.
- **The link merges far more on the control than on the calibration set:** 62 merges against 24 at 0.5.
- **A second blind test.** When the merge ships, train.py scores the same test again with it OFF, stored under
  `test_metrics["coref_merge_off"]`, so eb20 stays comparable with eb19. Traced: with `link 0.5` stored, cluster
  F1 is 0.3846 as shipped vs 0.3125 off, and the OLD strict `event_argument` falls 0.0253 -> 0.0000 under the
  untrained merge.

