# Open items

**Only outstanding work lives here.** Everything resolved, refuted or superseded is in
[[EXPERIMENT_CATALOG]] (what was run, newest first) and [[PROJECT_HISTORY]]
(narrative, including [the full pre-2026-09-21 TODO](PROJECT_HISTORY.md#todo-archive-2026-09-21)
preserved verbatim). Last rewritten 2026-09-21.

## Outstanding, at a glance

| # | item | why it matters | state | next action |
|---|---|---|---|---|
| 1 | **absneg4: absent pool scoped to event roles** | the only lever that has ever moved `event_argument` outside the floor (+0.0376), minus the collateral | **RUNNING** since 16:38 UTC 2026-09-21, ~19h, A100 `ceeee75126414…` | read it against BOTH absneg2 arms |
| 2 | **eb17-best: the warm-start base** | folds in every measured lesson; adds label negatives, which four dead mechanisms have waited on | BUILT, not launched, ~14h ~$30 | launch once (1) lands |
| 3 | **classification collapses under interventions it never targets** | −0.1977 on absneg2, −0.403 on warm cells, −0.1492 before that; blocks shipping negatives | ROOT CAUSE NARROWED: 96% is `docee_event` alone | see whether (1) spares it; else isolate docee |
| 4 | **`record_anchor_threshold` defaults to 0.5** | nothing calibrates record cutoffs; no model can use 0.5 well | open, unstarted | sweep on validation like the record threshold was |
| 5 | **Cross-event contamination** | the top real decode defect on multi-event documents | probe gated behind item 4 | build the probe |
| 6 | **`candidate_pool: shared` never A/B'd** | its 64 tensors sit untrained in every checkpoint; proven NOT structural, so the experiment is available | open | one arm, cheap |
| 7 | **Purchased NER gold is unusable as supervision** | 20,604 docs of real entity gold sitting idle, because it is not exhaustive | needs a partial-annotation flag | build the flag, then add the corpora |
| 8 | **`turkish_event` gold-surface failures** | costs real supervision and aborts tolerant runs | known; `on_missing_surface=skip` masks it | fix `max_len`/windowing properly |
| 9 | **Relation warm-start regression (−0.037, −22%)** | `task_lr: 5.0e-4` was tuned for COLD heads | hypothesis unverified | try a lower task_lr on the warm stage |
| 10 | **EKF: §10 crux reopened, §14 does not reproduce** | the EKF's claimed edge under unreliability | open | re-derive on real streams |
| 11 | **EKF: exposure counts are not casualties** | 12 of Helene's 106 `dead` are audited non-casualty; schema has nowhere to put them | open | extend the schema |
| 12 | **No benchmark that can score the filter** | Türkiye's baseline was an oracle by construction | open | build a non-oracle benchmark |

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

Hold until absneg4 lands — `absent_negatives_in_denominator` is exactly a base-level decision.

**Selection now has an aggregate.** `eval_overall_{strict,relaxed}_{micro_f1, head_macro_f1,
head_min_f1}` landed 2026-09-21. `head_min_f1` is the interesting one for a base: it cannot be
improved by trading one head away, which is the failure mode this programme keeps hitting.
Consider it for `metric_for_best` in place of entity F1.

## 3. P0 — blocks the next experiment

**`record_anchor_threshold` defaults to 0.5 and nothing calibrates it.** The record threshold
itself was swept and moved the 137k structure reference to 0.1119 at 0.1; the anchor cutoff
never got the same treatment.

**Cross-event contamination** is the top real decode defect, and its probe is gated behind the
threshold work above. Do NOT reach for sharper type boundaries — the standard mitigations were
considered and the data-side remedy (negative documents) is the proposed route.

## 4. Open experiments, cheap

- **`candidate_pool: shared`** — never A/B'd. Measured NOT structural (340 tensors either way,
  none added, removed or reshaped), so the only honest way to ask is to TRAIN an arm with it on.
  Its tensors currently sit at initialisation in every `per_query` checkpoint.
- **Negative documents** (data, not decode) for cross-event contamination.
- **Base-word positive/negative samples** — a different granularity, aimed at noun-phrase
  boundaries rather than event interference.
- **A lower `task_lr` on warm stages** — 5.0e-4 was tuned for cold heads and the relation head
  regresses −0.037 in the warm start.

## 5. Data debt

- **The purchased NER gold cannot be used as supervision** until there is a partial-annotation
  flag. cmnee_ner (9,281) and duee_ner (11,323) are real entity gold on documents already in
  the mix, but it is not exhaustive: adding it as `entities` would teach the entity head that
  every unannotated entity is absent. This is exactly why `merge_entity_types.py` writes
  `entity_types` and never an `entities` block.
- **`turkish_event` surfaces fail to align.** Currently masked by `on_missing_surface=skip`,
  which silently drops supervision. Needs the `max_len`/windowing fix, not a tolerance flag.

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

## 7. Research direction

- **Does a fine-tune need an explicit regularizer, or is early stopping enough?**
- **Chunking distorts the real-news arm** — `window_size`/`stride` are **subword tokens, not
  words**, and the configs' comments say "word window". Measured, and not where expected.
- **We cannot compare event arguments to the literature** because we do not compute the
  metric the literature reports.

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
