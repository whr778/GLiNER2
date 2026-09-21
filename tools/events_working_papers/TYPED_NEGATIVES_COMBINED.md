# Combining absent negatives with typed role edges

**Status: PLAN, nothing built.** Written 2026-09-21, the day absent negatives got their first
legal measurement and option 2's decode-time arm came back a null on F1. Companion to
[[LABEL_NEGATIVES_PLAN]] and [[OPTION_2_TYPED_ROLE_CONSTRAINTS]].

---

## 1. Why combine them, and why now

Two measurements point at the same place:

| intervention | measured | where |
|---|---|---|
| absent negatives in the listwise denominator | `event_argument` strict **+0.0376**, `classification` **-0.1977** | `absneg2`, 2026-09-21, operating point verified identical |
| TypedRole at DECODE time | **null on F1** (-0.0001, +0.0030), precision up / recall down, 98% of dropped arguments were false positives | local, 40 casie docs, 2026-09-20 |

The decode-time null is the informative one. The filter removes almost exclusively wrong
arguments and F1 still does not move, because **it can only DELETE**: the record head is
greedy, so there is no alternative assignment to re-route to. A constraint that cannot
substitute a better filler cannot raise recall, and precision alone does not move F1 here.

**The conclusion both results support: the type signal has to enter the OBJECTIVE, not the
output.** And the objective it should enter is the one absent negatives already changed --
the listwise denominator. That makes these two interventions the same mechanism carrying two
different signals, rather than two features bolted together.

## 2. What "typed negatives in training" means

**Typed hard negatives in the listwise denominator.** For a role query with gold, add to its
denominator the candidate spans the map says are TYPE-INCOMPATIBLE for that
`(event_type, role)`. The model is then trained to rank the right filler above a span that is
plausible in position but wrong in type -- which is exactly the discrimination the decode
filter was making after the fact, moved to where it can change what the model proposes.

It shares `absent_negatives_in_denominator`'s machinery: same loss, same per-task pooling,
same `task_ids` guard. Turning both on adds two kinds of negative to one denominator.

**Two alternatives, rejected with reasons:**
- *Logit masking during training.* The model never sees the forbidden pairs, so it never
  learns to rank them, and at inference on a schema the map does not cover its behaviour is
  undefined. The constraint would be training a dependency on itself.
- *A violation penalty term.* Introduces a weight to tune and duplicates what the denominator
  already does. Prefer one mechanism with two signals over two mechanisms.

## 3. The design -- a 2x2, and HALF OF IT IS ALREADY TRAINED

|  | typed OFF | typed ON |
|---|---|---|
| **negatives OFF** | `absneg2-control` **(exists)** | **NEW** |
| **negatives ON** | `absneg2-treatment` **(exists)** | **NEW** |

Both existing cells come from the same config family, the same 16 corpora, the same selection
rule (`eval_entity_strict_micro_f1` at threshold 0.3), and both shipped their FINAL epoch.
So two arms buy the full factorial: **~$50, not ~$100.**

**Reach is not a limit here.** The map's 174 roles across 74 event types derive from casie,
cmnee and duee -- all three are in this mix, as are their event files.

**THREE THINGS THAT MUST BE CHECKED BEFORE REUSING THE EXISTING CELLS**, because reuse is
where A/Bs go wrong:
1. **Code drift.** The new arms run at a later commit than `absneg2` (854d3ba). Diff
   `gliner2/` between them and show every change is either the intervention itself or inert
   under the old settings -- the same check that cleared reusing the roles2 treatment.
2. **`seed: None`.** These configs fix no seed, so the four cells differ by nondeterminism as
   well as by treatment. The interaction term inherits that noise. Either fix a seed for the
   new arms and state that the old cells did not have one, or read only the two
   typed-on-vs-off contrasts and treat the interaction as indicative.
3. **Identical corpora, val and test** across all four, verified by reading the configs --
   and verified with a check that can fail. A first attempt at exactly this check passed
   because both halves errored to empty output.

## 3b. THE PROGRAMME HAS RUN A TREATMENT FACTORIAL BEFORE, AND THE COMBINATION LOST

Checked 2026-09-21 rather than assumed. Two earlier things are called "2x2" in these papers
and only one is a treatment factorial:

| prior | what it was | outcome |
|---|---|---|
| **real vs synthetic 2x2**, 2026-08-23, $7.47 | a genuine treatment factorial on the DATA axis | **the MIX preserved WORST, -38.6%, worse than either arm alone** |
| JOINT_IE_SCALING's "2x2" | 2 checkpoints x 2 TEST SETS, a cross-scoring matrix | read down the columns, never across the rows |

**No negatives x typed-roles factorial has been run.** This one is new.

**But the precedent matters and argues against optimism.** The one time this programme
combined two interventions factorially, the combination was WORSE than either alone. So
prediction 2 above -- "the combination beats either alone, sub-additively" -- is the
optimistic branch, and the measured precedent points the other way. If the combination
UNDERPERFORMS both single arms, that is not an anomaly to explain away; it is this
programme's second observation of the same shape, and the 2x2 is precisely the design that
can tell the difference.

A related caution already recorded in PROJECT_HISTORY: **a genuine 2x2 needs a second
FACTOR, not a second dose.** Absent negatives and typed negatives are two different signals
entering one denominator, so they qualify -- but the check is worth making explicitly before
spending, because "more of the same mechanism" would not.

## 4. Gates -- each arm must prove its own treatment from inside the run

- `absent_negatives_used` non-zero on negatives-ON arms, **exactly zero** on negatives-OFF.
  Read it as a RATE: ~75-88% of steps are non-zero when live, so any single line is
  uninformative.
- `typed_negatives_used`, a new counter on the same pattern: how many typed hard negatives
  entered the denominator, emitted AFTER the pooling. **Zero on a typed-ON arm means the
  result is void** -- this programme has shipped that failure four times.
- The injector's `[composition] negatives:` line, identical across arms that share it.
- **Publish `final/` as well as `best/`** (already wired). Two arms winning on different
  epochs is what voided `roles2`; with both checkpoints published it is diagnosable instead
  of fatal.

## 5. Measurement, pre-registered

**Primary:** `event_argument` strict micro-F1 at a fixed operating point, `eval_provenance`
matching -- `compare_runs.py` REFUSES across differing points now, which is how the roles2
test-set defect surfaced.

**Floors, all from SEED REPLICATES and none from the confounded pair:** gate3's three seeds
(s1/s4/s21, same config and test set) give `event_argument` **0.0009**, `structure` 0.0025,
`relation` 0.0110, `classification` **0.0178**, `event_type` 0.0395; the clean re-baseline's
seed42/43 pair corroborates classification at 0.0013. Against these, absneg2's
`event_argument` +0.0376 is **~42 sigma** and its `classification` -0.1977 is **~11 sigma**.
They are transferred from a different config and an 18,786-record test set, so a replicate of
the absneg pair would make them exact -- but the floors themselves are measured, not assumed.

**Predictions, recorded so they can fail:**
1. Typed-ON raises `event_argument` **precision** more than recall, at both negative levels.
2. The combination beats either alone on `event_argument` strict, but **sub-additively** --
   both signals push the same head through the same denominator.
3. **The classification collapse tracks the NEGATIVES axis, not the typed axis.** If
   typed-ON alone also collapses classification, the damage is not specific to absent
   negatives and the whole diagnosis needs revisiting.
4. Gains concentrate in type-named roles (`Location`, `Date`, `Time`) and are absent on
   `Subject`-like roles -- which the map already omits as FLAT, so this is a check that the
   map's omissions were the right ones.

## 6. Cost, and what would make this not worth running

**~$50** for two A100 arms at ~17h, plus ~$2 to re-score if an operating point needs
matching. The 2x2 costs half what it looks like because two cells exist.

**Do not run it if:** the classification collapse turns out to be unfixable and
disqualifying, in which case the negatives axis is dead and only the typed axis is worth one
arm.

**A SEED REPLICATE IS NOT A PREREQUISITE.** It was proposed as one on the mistaken belief
that the classification floor was unverified. It is not: three seeds measured it. At ~42
sigma for `event_argument` and ~11 for `classification`, absneg2's two headline rows are not
plausibly noise, and the 2x2 can proceed. A replicate of this exact pair remains worth having
LATER -- it would turn transferred sigmas into exact ones, and give `event_trigger` a floor
it currently lacks -- but it is a refinement, not a gate.
