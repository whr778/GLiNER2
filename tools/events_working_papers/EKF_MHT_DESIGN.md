# Tracking Events Through a News Stream: A Filter, and the Association Layer It Needed

*Working paper. Rewritten for length 2026-08-25 — the previous 1,033-line chronological
version is at git `1e27878`. Every measured result and every negative below is carried over;
what was cut is narration, not findings. The build record with contemporaneous detail is
`EKF_MHT_BUILD_RECORD.md`.*

## Abstract

A real-world event reported across a stream of news documents has a state that evolves: a
death toll rises, gets revised downward, and is reported by sources that disagree, censor
("at least 12"), and lag. We build the pipeline — synthetic stream generator, LLM realizer,
schema-driven extraction, normalization, and a per-role Extended Kalman Filter — and the
programme's question has two halves: **track** an evolving quantity, and **diarize**
observations into the right event stream.

The filter half was solved early and validated on held-out synthetic data, where fine-tuning
the extractor closes **75% of the gap to the structured-observation ceiling**. The diarize
half then took eleven weeks, produced more negative results than positive ones, and is the
subject of this paper.

**It is now solved, and not by the mechanism the design specified.** A global Viterbi decode
over three states — *own place, aggregate, reject* — improves every event we have at one
setting:

*Pooled RMSE, in deaths — lower is better.*

| event | ratio gate | global decode | change | three-way oracle |
|---|---|---|---|---|
| Hurricane Helene 2024 | 29.3 | **20.7** | −29.4% | 17.6 |
| Türkiye–Syria 2023 | 11,581.5 | **10,695.5** | −7.6% | 3,287.4 |
| Aegean Sea 2020 | 74.4 | **15.7** | −78.8% | 19.1 |

*(σ=0.3, reject_cost=4.0, stay=0.1, warmup=0. The oracle column consults ground truth and
cannot ship; it prices the ceiling. On Aegean the decode beats it because that oracle applies
a fixed drop rule rather than optimising the estimate — see §7.)*

### Metrics used in this paper

Two scales appear, and they are not comparable. **Anything quoted below states which.**

| metric | units | meaning |
|---|---|---|
| **pooled RMSE** | deaths | one RMSE over every (stream, time-grid) point. **The headline.** |
| **nRMSE** | dimensionless | per-stream RMSE normalised by that stream's range, then macro-averaged |
| **binding precision** | % | of the figures a model binds to a role, the share that are the *correct* figure |
| **strict F1** | — | exact span *and* role match, no partial credit |
| **catch / false-reject** | counts, % | cross-event figures rejected, against genuine observations wrongly rejected |

nRMSE was the headline until the metric error in §6: normalising per stream lets a stream
with a tiny range outvote the largest one, which is how a correct mechanism was recorded as
a failure. Pooled RMSE in deaths replaced it. **Numbers from before that correction are in
nRMSE and are marked as such** — they are kept because the arguments built on them are still
the arguments, and re-running every historical arm on the new metric was not affordable.

Two properties carry it, and the specified design — a hypothesis tree with Hungarian
assignment — is neither. The decision must be **global**, because a greedy rule commits per
observation and one large figure admitted early poisons a stream's running scale for
everything after it. And it must be able to **reject**, because assignment headroom is
measured at **zero**: a two-way oracle scores exactly what the shipped gate does, and the
entire residual is the null hypothesis. Hungarian assignment optimises the half worth nothing.

The paper's second contribution is the negative space around that result. Four alternatives
were built and measured against the same events and all lost; a magnitude gate that won 9×
on the event it was designed against split held-out; a corpus built to teach cross-event
suppression was beaten by a one-line threshold; and four separate instrument defects
produced confident wrong answers that survived because the instrument returned the expected
result. §6 is that list, and it is the more transferable half.

---

## 1. The problem

Two halves, co-equal, and the second is not downstream of the first:

| half | mechanism | status |
|---|---|---|
| **track** — recover an evolving quantity from noisy, censored, lagged reports | Extended Kalman Filter | built, validated on synthetic |
| **diarize** — decide *which* event stream each observation belongs to | global decode over {own, aggregate, reject} | **built 2026-08-25**; §4 |

"Diarize" is borrowed from speaker diarization. It is not "who spoke when" but *which event
is this figure about, over time*, and naming it that way makes obvious that the two halves
fail independently and that the second can silently destroy the first.

## 2. Method

Per role (`dead`, `injured`, `missing`), a 1-D random-walk Kalman filter with **relative**
measurement noise: `R = (σ · max(ref,1))²` where σ comes from the source
(official 0.06 / major outlet 0.12 / preliminary 0.25) scaled by the qualifier
(point 1.0 … feared 2.5). Process noise grows as `q_rel · max(μ,1) · dt`, so genuine jumps
stay admissible between reports. Noise scales by the current **estimate**, not the observed
value — scaling by the raw report over-trusts a low reading and drags a rising estimate down.

Observations come from a four-stage pipeline: a relevance gate, event extraction, casualty
record extraction, and normalization to `(value, qualifier, source)`. `GATES.md` documents
every filter, its type, and its default.

## 3. What was established before the association work

**Extraction is solvable.** On held-out synthetic streams, fine-tuning the extractor closes
75% of the gap between zero-shot extraction and a structured-observation ceiling. The
generator, realizer and blind-test protocol are in the build record.

**On real news the tracker lost to a trivial baseline.** On the pre-registered Türkiye–Syria
2023 run, `est_last_value` beat the EKF (nRMSE 0.208 vs 0.136), a 1999 death toll quoted in an
article's history section was tracked as a 2023 figure, and one of the two affected countries
was never recovered at all. **Attribution, not filtering and not extraction, was the
bottleneck** — and that diagnosis has held ever since.

**A magnitude scope gate won 9× and split held-out.** Every state stream in the Helene run is
contaminated, and always upward: North Carolina (true peak ~123) receives 200, 215, 227, 230
and 250. Walking a stream in time order, the tracker's own running estimate supplies a scale,
and a reading far above it is not a rival claim about the state but an observation about a
larger scope. Reclassify rather than discard — a rejected figure moves to `__aggregate__`,
where it is a correct observation instead of a corrupting one. Pooled error on Helene:
**314.5 → 29.3 deaths**.

It then split on held-out data: 3.7× better on the contaminated stream, 2.3× worse on the
clean one (nRMSE, per stream). The reason is diagnostic and set up everything after it — **without a declared
scope hierarchy, "a larger scope" and "the largest part" are indistinguishable from the
numbers alone.** We declared one (`rollup.json`), and the reference it enables turned out to
be a measured negative in its own right (§6).

## 4. The association layer

### 4.1 The oracle that priced it — and priced it wrong first

Before spending on a hypothesis tree, we priced it. Assign every observation to the scope it
actually fits, using ground truth: a ceiling, not a method.

    shipped scope gate      0.591   nRMSE
    two-way oracle          0.537   nRMSE
    headroom               +0.055   (9.3% relative)

**That number prices the wrong thing, and the tell was already in the table.** The oracle is
*two-way* — every observation goes to its own place or to the national total — so a figure
belonging to no scope in this event has no correct home, and it scores Katrina's 1,400
exactly as badly as the shipped gate does. On two states the gate already **beat** the
"perfect" assignment, because it has a third option the oracle lacks: *drop*. That was read
as a curiosity rather than as a defect in the instrument.

Give the oracle a reject option and the ceiling moves to **+0.111 (18.8%)** — double. Later,
re-derived on the pooled metric: shipped gate 29.3 deaths, three-way oracle 17.6, and it gets
there **purely by dropping more** (106 observations kept down to 76).

**So assignment headroom is zero and the whole residual is the null hypothesis.** Both
properties of the eventual fix are in that sentence, and it took two more failed builds to
read them.

### 4.2 What the reject option is actually rejecting

Restricting to the gated population — the two-way oracle keeps everything else by
construction, which was inflating a `gated` feature to AUC 0.763 for free — the oracle's
rejects have **median value 6.0** against the kept set's **99.0**. Split by direction against
the place's own truth:

| | share |
|---|---|
| **stale, BELOW truth** | **63%** |
| over, above truth | 20% |
| no truth coverage | 17% |

The gate rejects *upward only*. It is blind to two thirds of its own prize by construction —
an article at t=83.8h still saying "three" dead in North Carolina when the toll is 46.

### 4.3 Track birth, built and lost — and what the failure localized

The cheapest piece of MHT that delivers a null hypothesis is track birth by innovation
gating. It was built and **lost to the fixed magnitude ratio it was meant to replace**
(nRMSE 0.608 against 0.591), degrading the national stream 6.7× while doing so, because judging a stream
against its own track is circular in exactly the way the scope reference was.

The missing property was not a better birth rule. It is that **the decision has to be made
over the whole sequence at once.**

### 4.4 The global decode — what ships

`scope_gate.hmm_gate`. Three states, decided by Viterbi over the stream, with hard-EM around
the decode because the `own` level is itself unknown.

Emissions are **one-sided**: a rising toll may legitimately exceed the level established so
far, so only a reading far *below* it argues against `own`. The band top is
`natl/part_ratio` — the same boundary the ratio gate uses — so this is a soft form of §3's
rule rather than a rival to it. Per-observation evidence from outside the magnitude channel
(an out-of-window date, a place outside the declared hierarchy, a syndication marker) is
added to the reject state, so those filters **argue rather than veto**.

Results are in the abstract. On Helene it beats the gate at all twelve knob settings swept
and captures 73% of the measured reject headroom; on Aegean the ratio gate is *inert* —
74.4 at ratio off, 4.0, 3.0, 2.0 and 1.5 alike — so that column is not a tuning margin.

**Design rule, measured:** keep every feature weight **below `reject_cost`**, so no single
feature can force a reject alone — it can only tip a case magnitude has already made
marginal. The sweep shows a cliff exactly at that boundary, and it is what protects Türkiye.

**It nearly read as another split result.** The first sweep had Türkiye worse, which would
have been the third intervention in a row helping Helene and hurting Türkiye, and we proposed
the divergence was structural — Türkiye's `global-max` reference is self-referential for its
dominant stream. **The per-stream diagnostic refuted that**, in the opposite direction: the
decode helped the self-referenced dominant stream (nRMSE 0.403 → 0.377) and hurt the *minor*
one (0.923 → 1.858). Chasing the inversion found the cause — a `warmup` parameter copied from the
greedy gate, which pins the first readings to `own`. Syria's first two are contaminating
Türkiye figures (9,057 and 17,674 against a true peak of 5,800); pinning them poisoned the
level and the genuine 3,317s were then rejected as stale. **That is precisely the greedy
commitment the global decode exists to remove, smuggled back in as a knob.**

### 4.5 Four alternatives, all measured, all rejected

- **Student-t measurement model.** One-sided IRLS reweighting. Helene −1.7 deaths, Türkiye
  +651. Retires none of `REJECT_SIGMA`, `MAX_RATE` or the gate's drop branch, which was the
  reason to want it: with the gate off, Helene is 314.5 deaths under every ν tested. **Whatever the
  gate catches, it is not a fat tail.** The symmetric textbook form is much worse on both
  events, so the one-sidedness is measured rather than assumed.
- **Soft PDA association.** Loses to the hard decode on both events (pooled RMSE, deaths:
  Helene 28.7 vs 20.7; Türkiye 19,150 vs 10,696). Two structural reasons: soft weighting cannot **remove**, and
  the headroom is in removal; and PDA arbitrates *independent* targets while ours are
  **nested** — a place's toll is part of the national toll — so an ambiguous reading is
  soft-assigned to both part and whole at once (106 observations became 118 assignments).
- **A fourth state for downward reclassification.** Correct and **inert**. Helene's North
  Carolina truth falls four times, but the reports never follow it down: after each revision
  the later readings are 230, 230, 230, 1,400, 98, 250 while the official toll falls to
  84–123. No state, filter or change-point detector can track a revision that is never
  reported. This reframes `CENSOR_AT_LEAST` — the filter is not wrong to refuse to descend,
  **the data never descends** — making it a source-acquisition problem, not a modelling one.
- **Folding the date and scope gates into the emission.** Works on its own terms — on Helene,
  cross-event rejection goes from 5/6 caught at 19.8% false rejections to 4/6 at 9.9%, at no
  trajectory cost — and cannot be validated on the headline metric. See §7.

## 5. The front end, and why it moved onto the critical path

The residual under the scope gate is **4.7% cross-event contamination**, and no decode-side
signal worked against it. The corpora offered no purchase either: **0.0% of training
documents contain a figure the model is supposed to leave alone.**

**So we built that corpus** — withholding an interfering event's records while keeping its
text — and trained on it. The suppression is real: it removes 15 of 20 large false positives
and more than halves the ungated error. **Then a declared per-event plausibility ceiling —
one threshold, no model, no training — beat it** (378.809 → 18.287 pooled RMSE in deaths,
on the production model),
because the large false positives were never other storms' tolls. They were Asheville's
population, FEMA flood-insurance policies, power crews, and years read as death tolls. Both
genuine cross-event figures survive muting *and* the ceiling.

The remaining candidate was a span-embedding router, and it was **blocked on the front end**:
no model then available emitted trigger-and-argument spans on wire copy at all. That put the
extractor on the critical path, and it was rebuilt (mmBERT, English trigger→argument
798 → ~39,800 examples). The rebuild **works**: it binds the correct death toll 67–100% of
the time (binding precision) against the incumbent's 0–7.7%, and beats it on all eight strict
F1 heads of the shared blind test.

**Its pre-registered gates 1 and 2 failed anyway, and gate 1 was the instrument.** Gate 1
counts a window as satisfied when a trigger and *any* bound argument appear, never checking
whether the bound figure is right — the incumbent's "65%" was 39 firings carrying **three**
correct tolls. Scored best-over-a-range, a form gate rewards indiscriminate firing. Gate 2 —
that two same-type events in one passage stay separable — fails for both models and is a
property of the decode, not the corpus: the record head pools same-type instances into one.
That remains open.

## 6. Negatives, and what each one closed

- **Our strongest prior claim does not reproduce.** Recorded in the build record; it did not
  survive re-measurement on a clean run.
- **Aggregates constrain the sum, and our metric only scored the split.** §6.2's original
  verdict — an aggregate row "does not pay off" — was decided by a scorer that measures only
  the split. Re-scored in absolute deaths, **the aggregate improves the national total at
  every density** — 87.6 → 28.5 at 10% part density — most where the paper said it
  "loses worst". The mechanism was right and
  the verdict was wrong, from the same sentence. The conclusion had survived three deliberate
  robustness attacks in two days — proportional Q, correlated Q, and a seed/noise floor —
  none of which could find the problem, **because all three varied the model and none varied
  the scorer.**
- **Restructuring the text does not fix attachment.** Rewriting articles into self-contained
  bullets does not repair number-to-place binding.
- **Guide-filtered negative sampling: wired, tested, negative.** The guide veto lost on 7 of
  8 metrics.
- **The implied-max reference loses badly** (nRMSE 2.590 vs 0.591 on Helene) — the same circular
  self-reference as track birth.
- **Declared knowledge beats learned signal at cross-event rejection.** Scope membership,
  with zero model calls, catches 4 of 6 cross-event figures at **7.3% false positives**; the
  best learned signal tried, one call per observation, catches the same 4 at **31.7%**. Same
  recall, a quarter of the false positives, no model.

## 7. Limitations

- **The gate collapse cannot be validated on the headline metric, on any feed we have.**
  Folding the date gate, the scope gate and syndication boilerplate into the emission works
  on its own terms (Helene: 5/6 caught at 19.8% false → 4/6 at 9.9%, no trajectory cost) but
  moves pooled RMSE by **nothing** — on Helene, and on the Aegean event built specifically to
  test it (15.74 at every feature weight from 0 to 3, while demonstrably rejecting three of
  six contaminants). The reason is structural: **association routes contaminants out of the
  scored streams before the gate ever sees them.** Aegean's 17,000s are keyed `unknown` and
  `marmara`, never `Izmir`, and only 12 of 53 observations are bound to a gated place at all.
  A fourth event would not change this.
- **A pre-registered prediction failed, and the failure was informative.** Aegean was chosen
  for scale separation — 119 deaths against an İzmit 1999 reference of ~17,000, 143× where
  Helene's contaminants differ by 6× and Türkiye's by 3×-and-crossing — with a registered
  prediction that the collapse would show a large gain. It did not, for the reason above,
  which has nothing to do with scale separation.
- **Two same-type events in one passage remain inseparable.** A property of the record head,
  which pools them into one instance. No corpus fixes it; the instance dimension is the next
  constraint.
- **Four instrument defects produced confident wrong answers.** A threshold sweep that was
  inert for five measurements; a strict-vs-relaxed comparison that turned a doubling into an
  apparent halving; gate 1 counting firings as though they were hits; and a metric that
  scored only the split. **Each survived because the instrument returned the expected
  answer** — a harness that "correctly" fails the incumbent is not audited. Pre-registration
  constrains the threshold, not the measurement apparatus.
- **The exposure/casualty boundary is unmodelled.** 11% of Helene's `dead` observations are
  hand-audited non-casualty; half are exposure counts (300 rescued, 50 patients rescued, 32
  evacuated) and half unit confusion (a two-day period, six states, 1,400 landslides). A
  rescued person is a *counterfactual* casualty. The schema has four roles — location,
  injured, missing, dead — and none for exposure, so a number with no correct home lands in a
  wrong one. That is a data-modelling gap, not a gate.
- **The relevance gate does not filter non-English at all.** The shipped model admits 199 of
  200 clean Turkish articles, because it is English-only by vocabulary — worse than the 58.5%
  that forced an earlier rewrite, and silent. Translation does not fix it.
- **Conflict casualties are out of scope for validation.** The taxonomy extends (KIA/WIA are
  realized harm; belligerent side is the scope axis again), but conflict ground truth is
  contested by construction and could not be scored the way these three events are.

## 8. Reproducibility

**The Helene observation set cannot be regenerated.** It depends on archived captures whose
availability has since changed; the extracted observations and the ground-truth trajectory
are committed, the article text is not (it belongs to its publishers) and lives in a
gitignored cache. The Türkiye feed reads truth from the same sentence the extractor reads,
which is why `est_last_value` scores nRMSE 0.000 there by construction — a defect the Helene and
Aegean feeds were built to avoid. **Aegean is the only feed with genuinely independent
sources on both sides**: ground truth from the Wikipedia revision history (55 timestamped
points, İzmir 12 → 116, including a real downward reclassification), documents from Turkish
English-language wire copy.

Code, feeds and every sweep referenced above are in `tools/ekf_showcase/`, indexed in its
README with the verdict attached to each probe. Gate defaults and types are in `GATES.md`.
Contemporaneous build detail is in `EKF_MHT_BUILD_RECORD.md`; the running narrative,
including the errors as they were made and caught, is in `PROJECT_JOURNAL.md`.

## 9. References

Carion et al., *End-to-End Object Detection with Transformers* (2020) — the set-prediction
framing and the Hungarian matching argument that prompted the global-decode design, and
whose §4.2 "why greedy fails" is the same argument as §4.3 here.
