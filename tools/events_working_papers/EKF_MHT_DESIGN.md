# Tracking Events Through a News Stream: A Filter That Works, an Attribution Layer That Does Not

**William Roe**¹ (whr778@gmail.com) and **Claude**² (noreply@anthropic.com)

¹ Project author and maintainer  ·  ² AI assistant (Anthropic, Claude Opus 5) — design, implementation, and drafting

*Revision of 2026-08-19. Companion to `PAPER_0_FOUNDATION.md` (the extraction substrate)
and `JOINT_IE_SCALING.md` (the within-document instantiation of the same thesis). The
build record — codebase attachment points, CLI, training plan, decision log, generator
specification, test protocol, and the negative-sampling implementation notes — is kept
verbatim in `EKF_MHT_BUILD_RECORD.md`.*

---

## Abstract

A real-world event reported across a stream of news documents has a state that evolves:
a death toll rises, gets revised downward, and is reported by sources that disagree,
censor ("at least 12"), and lag. We build the full pipeline — synthetic stream generator,
LLM realizer, schema-driven extraction, normalization, and a per-role Extended Kalman
Filter — and validate it end to end on held-out synthetic data, where fine-tuning the
extractor closes **75% of the gap to the structured-observation ceiling**.

Then we run it on a real event, and it loses to repeating the last reading.

This paper reports that outcome and what it localizes. The programme's question has two
halves — **track** an evolving quantity, and **diarize** observations into the right
event stream — and only the first is built. What ships for the second is hard assignment
on a string key, not the specified multi-hypothesis association. Every real-event failure
we have measured is in that second half: a 1999 earthquake's death toll tracked as a 2023
figure, one of two affected countries never recovered at all, and a hurricane's toll
contaminated by three unrelated storms quoted in the same articles. A magnitude-based
scope gate recovers a **9× improvement** on the event it was designed against but
**splits held-out** — 3.7× better on the contaminated stream, 2.3× worse on the clean
one — and the reason is diagnostic: without a declared scope hierarchy, "a larger scope"
and "the largest part" are indistinguishable from the numbers alone. We then declared one,
and the reference it enables is itself a measured negative.

Before building the multi-hypothesis association the design specifies, we price it, and the
first price is wrong. A two-way oracle — assign each observation to its own place or to the
national total — improves the shipped gate by only 0.055, which reads as "association is not
what is missing". But that oracle cannot *reject*, so a figure belonging to no scope in this
event has no correct home; give it one and the ceiling moves to **0.111 (18.8%)**, double the
figure, with about half of it needing a null hypothesis and half not. We then build the
cheapest piece that delivers one — track birth by innovation gating — and **it loses to the
fixed magnitude ratio it was meant to replace**, degrading the national stream 6.7× while it
does so, because judging a stream against its own track is circular in exactly the way §5's
scope reference was.

The residual underneath is 4.7% cross-event contamination against which no decode-side signal
has worked, and the corpora offer no purchase on it: **0.0% of training documents contain a
figure the model is supposed to leave alone.** So we build that corpus, withholding an
interfering event's records while keeping its text, and train it. The suppression is real: it
removes 15 of 20 large false positives and more than halves the ungated error. **Then a
declared per-event plausibility ceiling — one threshold, no model, no training — beats it**,
because the large false positives were never other storms' tolls. They were Asheville's
population, FEMA flood-insurance policies, power crews, and years read as death tolls. Both
genuine cross-event figures survive muting *and* the ceiling.

The remaining candidate — embedding each event's own trigger-and-argument span and matching
against live filters — could not be run at all, because no extractor we have emits that input
on wire copy. Counting only corpora that bind an argument to a trigger, the boundary base has
**798 English rows against 20,884 Chinese**. **The critical path is therefore the extractor,
not the tracker** — a conclusion three built-and-failed association mechanisms were needed to
reach.

So we rebuilt the extractor against exactly that arithmetic: a cold start with **50× more
English trigger→argument supervision**, gates fixed on AP prose before spending. **The
rebuild works**, and establishing that took two corrections to our own instruments. It beats
the model it replaces on **every one of eight held-out heads**, measured by one command on
one machine at a pinned threshold. On the wire copy it was built for, its `dead` bindings
are correct **67–100%** of the time against the incumbent's **0–7.7%**, and it delivers
three times the correct death tolls at a matched threshold off one third the firings.

The pre-registered gate said the opposite, and we believed it for a day. Gate 1 counts
windows carrying a trigger and a bound argument — it never asks whether the bound figure is
the right one — so the incumbent "wins" it by firing on 39 of 60 windows at a permissive
threshold while binding `dead` to `"Mexico"`, `"Pacific coast"` and `"car Hurricane
Helene"`. Three of those 39 are correct. **A gate scored best-over-a-threshold-range
rewards indiscriminate firing**, which is the same comparison error this programme already
documents for A/B arms and had never applied to its own gates.

What does not move is the second gate: neither model separates two same-type events,
because the decode emits one instance per event type and pools every trigger and argument
into it — a passage naming two hurricanes returns a single event with both tolls bound to
it. That, not the corpus mix, is the next constraint.

We also report that the claim which justified that spend was our own instrument. The
harness scoring the gates set a per-event threshold the boundary decode never reads, so
every row of a five-point "sweep" ran at one value — the value at which the incumbent
genuinely scores zero. Corrected, the incumbent *passes* the gate the rebuild fails.

We also report that our own strongest prior result does not reproduce. An ablation
concluding the filter's advantage *widens* under noise and censoring was measured on
synthetic streams generated by the dynamics the filter models. On real revision data the
advantage is real, reliable, and **1–2%**. And the cached observation set behind every Helene
number here — including the 9× scope-gate result — turns out to be unreproducible from any
committed state of the repository, which we document rather than quietly re-baseline.

---

## 1. The question, stated as it was actually asked

**Can a filter track real-world events reported in text, and separate them into the right
streams?**

Two halves, co-equal, and the second is not downstream of the first:

| half | mechanism | status |
|---|---|---|
| **track** — recover an evolving quantity from noisy, censored, lagged reports | Extended Kalman Filter (§2) | built; validated on synthetic streams |
| **diarize** — decide *which* event stream each observation belongs to | multi-hypothesis association | **not built** |

"Diarize" is borrowed deliberately from speaker diarization. It is not "who spoke when"
but *which event is this figure about, over time*, and naming it that way makes obvious
that the two halves fail independently and that the second can silently destroy the
first.

**The association half is unbuilt, and this should be stated plainly rather than
discovered by a reader.** The design specifies association as gate → Hungarian assignment
→ top-K hypotheses → track birth and death. What ships is **hard assignment on an
observable string key**, feeding one single-stream filter per key: no hypothesis
enumeration, no deferred decision, no track birth or death. The only Hungarian solver in
the repository is inside the extraction model's record loss, matching predictions to gold
during training — unrelated to tracking.

That reframes every result below. The failures reported in §4 and §5 are failures of a
**placeholder**, not of multi-hypothesis tracking. The mechanism the design names for
exactly this problem has never been tested, so the programme's central question is not
yet answered — it is, so far, unasked.

## 2. Method

**Granularity.** Situation level, not event-instance level. A static instance has weak
dynamics and the filter degenerates to fusion.

**State.** Per track: a pooled embedding, the continuous quantitative arguments
(normalized counts, amounts, dates, magnitudes), and a salience term, with covariance.

**Dynamics.** Slow-drift embedding; monotone, random-walk or parametric process for the
quantities; decay on salience, which provides track death. As process noise goes to zero
the filter degenerates to optimal static fusion, so the single-document case is safe by
construction — a mild win over naive merging, never a loss. Dynamics are inert by default
and engaged only by evidence of change.

**Measurement.** Each document's extraction is a noisy and often censored observation —
"at least 12", "roughly $3M" — which is why the filter is *extended*: the measurement
function is linearized.

**Gating.** A one-sided innovation gate, plus a learned mixture-of-experts router blending
the local per-document read against the tracked estimate. The router is what withholds the
filter in exactly the regime where it would degrade a static read.

**The pipeline under test.** Synthetic stream generator → LLM realizer producing news-like
prose → schema-driven extraction → confidence cut → normalization → per-role filter. Each
stage is separately ablatable, which is what made the diagnosis in §3 possible.

## 3. The synthetic arc: extraction is solvable

Working backwards from a structured-observation ceiling of 0.115 normalized RMSE — the
score achievable if extraction were perfect — the zero-shot extractor reached 0.291. A
probe localized the gap to extractor precision and confidence calibration rather than to
normalization, and predicted that fine-tuning would close it. Fine-tuning on 29,198
casualty-structure examples:

| held-out test | zero-shot | fine-tuned |
|---|--:|--:|
| role precision | 0.627 | **0.914** |
| role recall | 0.906 | **0.965** |
| value exact (number binding) | 0.991 | **1.000** |

| end-to-end (normalized RMSE; ceiling 0.115) | zero-shot | fine-tuned |
|---|--:|--:|
| best (confidence cut 0.99) | 0.291 | **0.165** |
| no confidence cut | 1.16 | **0.193** |
| the weak `missing` role | 0.458 | **0.122** |

The fine-tune closed **~75% of the gap to the ceiling**, exactly as predicted, and removed
the dependence on a confidence cut — uncut performance went from unusable to 0.193.
Number binding is perfect.

**The caveat that turned out to matter most:** this was fine-tuned and tested on the same
synthetic distribution. Real news was left as the generalization test, and it is where the
arc breaks.

## 4. The first real event, and the first real defeat

Türkiye–Syria 2023, pre-registered before running. Ground truth is 16 daily points scraped
from a single news tracker page via an archive service; search summaries were checked and
found to **contradict each other on dates**, so they were rejected as a source.

**On real news the trivial baseline wins:** `est_last_value` 0.208 against the filter's
0.136 — reported here in the direction the metric runs, where the filter's 0.136 is the
better score, but the finding is that the ranking reverses relative to every synthetic
benchmark, and the margins are small enough that neither is a good result.

Three failures, none of them in the component the previous three sections optimized:

- **A 1999 death toll is tracked as a 2023 figure.** The 17,500 dead of the Izmit
  earthquake, quoted in an article's historical background section, enters the stream as a
  current observation — in every configuration.
- **Syria is never recovered.** The association key is computed per *document*, outside the
  envelope loop, so a two-country event collapses to one country.
- **Extraction is not the problem.** It reads the real trajectory nearly point for point.

**So the bottleneck is attribution, not filtering and not extraction.** That is the
finding, and it redirected the programme away from further extractor work.

**A method rule learned the hard way.** A pre-registered prediction was scored a *success*
by range-normalized RMSE while 12 of 20 readings came from a 1999 earthquake: the metric is
blind to contamination that lands mid-range. Always report the trivial baseline beside the
filter — it would have caught this immediately.

## 5. The scope gate: a 9× win that splits when held out

Reporting mixes scopes: a national total and its state components appear in the same
sentence. Filing the national figure under whichever state is nearest — what shipped — is
plainly wrong.

A magnitude gate gives each observation a scope test against a reference. On Hurricane
Helene, the event it was designed against:

| ratio | national stream | per-state mean |
|---|--:|--:|
| off | 0.402 | 5.247 |
| 2.0 | 0.316 | **0.591** |

Per-state error falls **5.247 → 0.591**, a 9× improvement, while the national stream
*improves* rather than paying for it, and the effect is flat across ratios 1.5–2.5 rather
than being a knife-edge setting. **Control:** removing the same number of observations at
random over 40 trials gives 4.427, so the gate is *selecting* rather than thinning the
stream into looking better.

Held out on Türkiye–Syria at the ratio fixed from Helene, it splits:

| | Türkiye | Syria | mean |
|---|--:|--:|--:|
| off | **0.228** | 3.401 | 1.815 |
| gate @2.0 | 0.522 | **0.923** | 0.723 |

Syria was 65% contaminated — 11 of 17 values are Türkiye's tolls — and improves 3.7×. But
**Türkiye, already clean, degrades 2.3×.** The reason is structural rather than a tuning
failure: Türkiye–Syria declares no scope hierarchy, so the reference falls back to the
running maximum across streams, which Türkiye's own values dominate. Türkiye is judged
against a reference it defines itself. The gate rerouted a value that was Türkiye's *true*
reading at that time.

**The diagnostic finding: the gate needs a declared scope hierarchy, not a magnitude.**
Without one, "a larger scope" and "the largest part" are genuinely indistinguishable from
the numbers alone — Türkiye's 41,000 filed under Syria and Türkiye's 41,000 filed under
Türkiye look identical to any ratio.

**And the obvious fix for it was built, measured, and does not work.** Both events now
declare containment explicitly — `rollup.json` carries a `hierarchy` block naming the
aggregate and its parts — which enables the reference the diagnosis calls for: judge a part
against its **implied maximum**, `aggregate - sum(other parts)`, rather than against a bare
magnitude. That is the right shape for the dominant-part problem on paper. Turkey against an
implied max of 46,800 − 5,800 = 41,000 sits exactly at its ceiling and is kept, while the
same 41,000 filed under Syria faces an implied max of 5,800 and is rerouted.

On Helene it is **much worse than the plain aggregate reference: 2.590 against 0.591.** The
parts are themselves contaminated, so their raw sum exceeds the whole and every implied
maximum clamps to zero; a two-pass version recovers most of that but Florida still sits at
9.437. And Türkiye–Syria — the event that motivated the refinement — **still cannot test it**,
because an implied maximum needs *independent* observations of the whole and that feed has
none. Declaring the hierarchy does not manufacture the data.

So the diagnosis stands and its first implementation is a measured negative. Plain aggregate
remains the recommended reference wherever an aggregate stream exists, and where none exists
no gate is possible at all. Detail in `EKF_MHT_BUILD_RECORD.md` §25.5.

**Honest scorecard.** A 9× win with a clean control on the event it was designed against;
held out, 3.7× better on one stream and 2.3× worse on the other. Both belong in any
writeup, and the Helene number alone is the misleading half. The ratio was also chosen
after seeing Helene's contaminated values; the plateau across 1.5–2.5 mitigates that
without removing it. And 0.591 is nine times better than catastrophic, not good in
absolute terms.

**What the gate does not address**, visible in the same audit: North Carolina's 1,400 is
*Hurricane Katrina's* toll quoted inside a Helene article, its 250 is a typhoon, and its
230 is Hurricane Milton. That is **cross-event** contamination, not scope. A magnitude gate
removes these for the wrong reason — because they are large, not because they belong to
another event — so it would keep any cross-event figure that happened to be small.

## 6. Results that did not reproduce, and results that were negative

Reported at the same weight as the positives, because collectively they are what localizes
the remaining work.

### 6.1 Our strongest prior claim does not reproduce

An earlier ablation concluded the filter's edge *widens* under unreliability and censoring,
and that conclusion was standing as the answer to "does the filter earn its keep". On the
only real revision data available — a hurricane's North Carolina toll falling 123 → 102 →
96 as deaths were reclassified — it does not hold:

| regime | filter | last_value | gain | filter wins |
|---|--:|--:|--:|--:|
| noise 0.05, report 80% | 0.094 | 0.095 | +1.3% | 191/240 |
| noise 0.10, report 60% | 0.189 | 0.192 | +1.5% | 205/240 |
| noise 0.20, report 40% | 0.322 | 0.327 | **+1.8%** | 217/240 |
| noise 0.30, report 30% | 0.499 | 0.504 | +1.0% | 222/240 |
| noise 0.40, report 20% | 0.628 | 0.634 | +0.8% | 217/240 |

A revision is the case where `last_value` looks best — it adopts the new figure at once
while a filter lags — so the filter can only win by smoothing noise the baseline chases.
**It wins reliably**, in 80–93% of trials, on the first benchmark in this project whose
baseline is not an oracle. **And it wins by 1–2%**, roughly flat, *shrinking* at the
hardest setting rather than widening.

**Why the earlier result was wrong: it measured synthetic streams generated by the same
rise/decay dynamics the filter models, so the filter was matched to the process by
construction.** Real trajectories — plateaus, reclassifications, a genuine downward step —
are not that process. *A dynamics model validated on data generated from it is not
validated.* The case for the filter must rest on something other than the
harder-regime argument, most plausibly multi-source disagreement, which no benchmark here
has tested because every feed so far has a single source.

### 6.2 Aggregates cannot constrain their parts

Making the state a vector so a national total becomes a linear observation over states is
the theoretically right move — a Kalman filter fuses "120 in North Carolina" and "227
across six states" natively, where any clustering method would work in value space and get
it backwards. Tested against a parts-only filter on real per-state trajectories, it loses
at every realistic reporting density and only reaches parity at 80%:

| per-state report rate | parts-only | vector | vector wins |
|---|--:|--:|--:|
| 10% | **0.4348** | 0.6085 | 4/40 |
| 35% | **0.2292** | 0.2860 | 10/40 |
| 80% | 0.1556 | **0.1520** | 30/40 |

Right in principle, and it does not pay off at the densities real feeds provide.

### 6.3 Restructuring the text does not fix attribution

The proposal was a summarizer emitting self-contained bullets so number-to-place binding
becomes local. Tested with hand-written bullets first, so a negative would cost nothing:

| arm | correct | false positives | fabrications |
|---|--:|--:|--:|
| raw text | **3/5** | 1 | 0 |
| free bullets | 2/5 | 3 | **2** |
| extractive bullets | **3/5** | 1 | 0 |

Restructuring is neutral at best and the free variant fabricates. **The motivating example
does not occur in the corpus at all** — the real failures are non-casualty numbers and
cross-event leakage, not aggregate splitting. There is also a structural tension: the
summarizer's highest-value act is normalizing prose into digits ("they died together" → "2
people died"), which is precisely what a verbatim-number guard must reject.

### 6.4 Guide-filtered negative sampling: wired, tested, negative

The residual confusion after description tuning is a genuine hard negative — "N people
killed" against "N people evacuated", both counts of people, separated only by the verb,
accounting for 11.2% of true death tolls. This is the case a guide-filtered loss exists
for: the competing type is genuinely correct sometimes and genuinely wrong here, so a loss
that penalizes it unconditionally would teach the wrong thing.

Implemented as a query-axis veto with a guide model and run as a two-arm A/B on 84,280
records, both arms swept to their own threshold:

| metric | control | guide veto | delta |
|---|--:|--:|--:|
| entity strict | 0.5858 | 0.5610 | **−0.0248** |
| relation strict | 0.1439 | 0.1108 | **−0.0331** |
| event_argument fair | 0.5786 | 0.5702 | −0.0084 |
| event strict | 0.3433 | 0.3443 | +0.0010 |

**Negative on seven of eight metrics**, including the argument axis the veto exists to
sharpen. The control validates the setup by reproducing the historical reference from
scratch. Against a measured run-to-run floor, the entity loss survives and the relation
loss does not, so the honest statement is that the veto is neutral-to-harmful and is not
the mechanism for this failure.

Wiring it did surface two ways it would have been **silently inert**, which is the
transferable part. A sample's query axis holds only the types its own record declares, and
just 0.23% of records name a competing count type natively — so the cross-record rivals the
method exists for are never present unless something injects them, and the veto would have
run, cost nothing and changed nothing. Separately, the veto resolved each candidate's
reference by position in the candidate list, which is only valid for a shared candidate
pool and is false for the per-query default that ships.

## 7. Where this leaves the programme

**Built and validated:** the filter, its measurement model, the learned gate, and the
text-to-observation pipeline, end to end on held-out synthetic data at roughly 1.4× the
structured-observation ceiling.

**Built and honestly weak:** the filter's advantage over repeating the last reading is 1–2%
on real revision data, and the argument that it widens under hard regimes does not
reproduce.

**Not built:** multi-hypothesis association — the half every real-event failure lives in.
The scope gate of §5 is the first concrete instance of that layer doing useful work, in its
simplest possible form and without a new model.

### 7.1 How much is left for better association — measured, not argued

The natural conclusion from §4 and §5 is "build the association layer properly," and MHT is
the specified answer. Before spending on a hypothesis tree, a cost matrix, Hungarian
assignment and track management, we measured what *any* better association could be worth.
The measurement needs no new model: assign every observation to the scope it actually fits,
using ground truth. It is a ceiling, not a method.

| | per-place mean nRMSE |
|---|--:|
| no gate | 5.247 |
| shipped scope gate | 0.591 |
| **oracle hard association** (uses ground truth) | **0.537** |
| headroom for any better association | **+0.055** (9.3%) |

**Perfect association buys 0.055** — and that number prices the wrong thing, which we found
only by trying to spend against it. The oracle is *two-way*: every observation goes to its own
place or to Total, so a figure belonging to no Helene scope has no correct home and it scores
Katrina's 1,400 exactly as badly as the shipped gate does. The tell was already in the table
below, read as a curiosity rather than as a defect in the instrument.

MHT's track birth/death **is** a null hypothesis, so give the oracle a reject option and sweep
the tolerance rather than fixing it:

| tol | kept | per-place mean |
|---|--:|--:|
| 2.00 | 100 | 0.533 |
| 1.00 | 100 | 0.533 |
| **0.50** | **85** | **0.480** |
| 0.25 | 76 | 0.499 |

**The corrected headroom is 0.591 → 0.480 = +0.111, 18.8% — double the figure MHT was
rejected on.** It splits almost evenly: 0.591 → 0.537 is reassignment (+0.055, mostly
Tennessee) and 0.537 → 0.480 is the reject option (+0.057), so about half the prize needs a
null hypothesis and half does not. Still a ground-truth ceiling, still one event, and the
tolerance is tuned and non-monotone, so there is no plateau to hide behind.

The per-place breakdown sharpens it. The gate already **beats** a perfect two-way assignment
on Florida (0.704 vs 0.734) and South Carolina (0.365 vs 0.558), because it has a third
option the oracle lacks — *drop*. Florida's 300 is not a casualty figure and North Carolina's
1,400 is Hurricane Katrina's toll: neither belongs to *any* Helene scope, so no assignment
scheme can place them correctly. Only Tennessee is a genuine association gap (0.817 vs
0.320), and its contaminants — 32, 32, 32, 36, 50 against a truth of 18 — are instructive:
too large for the state, too small to look national, so no magnitude test can catch them.
That is the real case for richer association.

### 7.2 The cheapest piece of MHT, built and lost

The reject option is where the prize is, and it is also the cheapest piece: track birth needs
no cost matrix and no hypothesis tree. So we built it. Tracks advance jointly in time order,
each observation is tested by normalized innovation against its candidate tracks, and one
that gates out of all of them is born into its own and leaves these streams. No ground truth.

Nothing swept beats the fixed magnitude ratio it was meant to replace:

| | Total | per-place |
|---|--:|--:|
| no gate | 0.402 | 5.247 |
| symmetric birth, own + aggregate (`q_rel` 0.20, the filter's own) | 0.555 | 1.059 |
| symmetric birth, tuned (`q_rel` 2.00) | 2.115 | 0.636 |
| one-sided birth, tuned | 2.115 | 0.608 |
| aggregate-only reference (`q_rel` 0.20) | 0.387 | 0.624 |
| **shipped magnitude scope gate** | **0.316** | **0.591** |
| three-way oracle (ground truth) | 0.308 | 0.480 |

**Both columns matter, and the second one alone would have flattered this.** The two tuned
arms buy their per-place improvement by dumping junk into the aggregate — Total 2.115 against
the gate's 0.316, a 6.7× degradation of the national stream, which is the one measurement
§4 calls honest. That is exactly the failure the shipped gate's three-outcome design was
invented to prevent: an earlier two-way version rerouted every reject to `__aggregate__` and
destroyed that stream. This associator reproduced a bug the project had already fixed.

Two causes, both measured. **Judging a stream against its own track is circular** — every
contaminant the track accepts moves the reference the next test uses — and removing the
self-reference is worth more than every other knob combined, 1.059 → 0.624. That is the same
failure the implied-maximum reference hit on Türkiye in §5, so it is now two independent
mechanisms defeated by one cause. And **the innovation is not informative about scope on a
rising toll**: at the filter's own dynamics the tracks are too tight to admit real growth,
Georgia keeping only 2 and 3 against a truth peak of 34, and the sweep must loosen them
tenfold before real rises survive — by which point only one to four observations are ever
born. Birth is never the lever; the rerouting is.

This does not kill deferred assignment; it is the first evidence *for* it. Hard assignment
commits early and poisons its own reference, which is precisely what keeping rival hypotheses
alive exists to prevent. Before this run that was a design preference.

**And it sets up the section that follows.** The observations the reject option would remove
are substantially the same observations the next section counts as cross-event contamination
— Katrina's 1,400 is both. There are therefore two routes to them: a decode-side one (M4,
now measurement-implicated but a large build against a 0.111 ceiling on a single-source
feed) and a data-side one. The data-side route is the subject of §7.3, and the reason it goes
first is that it is already built and instrumented while M4 is not.

### 7.3 What the residual actually is

A context audit of all 106 observations puts the remainder somewhere else entirely: 82.1%
genuine Helene casualties, **4.7% belonging to another event** (Katrina's 1,400, a typhoon's
250, Milton's 230, Bosnia's 16, Hurricane John's 2 in Mexico), 3.8% non-casualty numbers
(speeds, rainfall), and a 9.4% unclear tail.

Cross-event contamination carries the *large* values, so it does the most damage per
instance — and the scope gate removes the large ones for the wrong reason, because they are
big rather than because they belong to another storm. It therefore keeps any *small*
cross-event figure, which is exactly what happens with Bosnia's 16 and Mexico's 2. Bosnia is
structurally invisible to every signal tried so far: it is a *place*, not a named storm, so
nothing keyed on storm names can see it.

This audit describes the **archived** observation set, and that set can no longer be
regenerated (§9). The percentages below should be read as a description of one frozen
artifact rather than as a stable property of the pipeline.

**So the indicated next step was neither MHT nor more extractor fine-tuning, but negative
supervision on event identity.** The evidence for a training-data gap rather than a decode gap
is that binding collapses from 1.000 to 0.369 the moment documents become multi-event, and no
decoder change has moved it. Every training corpus is complicit — measured over 20,000 records
per corpus, **0.0% of training documents have zero records**, so the model has never once been
shown a casualty figure it is supposed to leave alone.

We built it. §7.4 reports what happened.

### 7.4 Negative supervision: built, and beaten by a one-line threshold

`casualty_loc_muted` keeps an interference snippet's *text* and withholds its *record*, so its
figures become negatives for the same queries — the first casualty corpus in which the model
is shown a figure it must leave alone. Two arms were trained identically for four epochs,
differing only in the corpus, with the control's own binding numbers measured beforehand
rather than inherited from a superseded checkpoint.

**The treatment took.** On the blind test, precision rose and recall fell (0.8119/0.8182 →
0.8273/0.7754) — suppression, pre-registered as the expected shape because the test split
carries gold on every interference snippet. On Helene it removed **15 of the control's 20**
large false positives and cut the ungated per-place error from **46.844 to 19.822**.

**Then the cheapest possible alternative beat it.** Auditing what those large values actually
were found that none of them is a Helene death toll:

| class | examples |
|---|---|
| resident population | 94,000 (Asheville) · 19,000 (Boone) · 3,100 (Saluda) |
| **people, but not casualties** | 1,500 active-duty troops · 8,000 power crews |
| not people at all | **129,933 FEMA flood-insurance policies** · 15,000 wellness checks · 1,100 churches |
| year read as a toll | 1,916 ("since 1916") · 2,004 ("**In 2004**… four people were killed") |
| cross-event | Katrina's 1,400 · Maria's 3,000 |

The 15,000 is the sharpest: its sentence exists to warn against this exact error — *"that was
mistakenly interpreted as meaning 15,000 people were missing"* — and the extractor made it
anyway. The 94,000 is emitted as `dead` *and* `injured` *and* `missing` at confidence 1.0.

A **declared per-event plausibility ceiling** — Helene killed on the order of 230, so a
five-figure figure in any of its streams is not a casualty count — needs no model, no training
and no GPU. Swept rather than chosen:

| ceiling | production model | control | muted |
|---:|--:|--:|--:|
| off | 378.809 | 46.844 | 19.822 |
| 20,000 | **18.287** | 26.961 | 19.822 |
| **2,000** | 18.190 | **5.853** | 6.194 |
| 1,000 | 18.190 | 5.057 | 6.194 |

Dropping the single 94,000 is worth **20×** on the production model. And at a ceiling of 2,000
— about nine times the true toll, so generous rather than tuned — **the control beats the
muted arm on both ungated (5.853 vs 6.194) and gated (3.336 vs 3.729)**, while carrying 81
*more* observations. The ceiling removes only junk; muting removed genuine signal alongside
it.

**The arm is therefore superseded, not vindicated.** Its pre-registered guard was the ungated
comparison, and it passes only against an *undefended* control — a configuration nobody would
ship. What stands is that the suppression is real and learned. What does not stand is that it
is worth having.

**And it fixed the class it was not built for.** Both cross-event tolls survive muting, and
survive the ceiling too: at 2,000 Katrina's 1,400 is a perfectly plausible magnitude, and a
ceiling set low enough to catch it is no longer a plausibility test but the magnitude gate
again, rejecting a figure for being large rather than for belonging to another storm. The same
holds for the 1,500 troops and 8,000 crews — living people in the affected area, wrong for a
reason no ceiling can see and no entity-type check can reach. **Cross-event contamination
remains unsolved.**

That is the paper's main practical contribution, and reaching it took a real event, a wrong
price, and two builds that lost: the obvious next build was first rejected on a ceiling
measuring the wrong thing, rebuilt in its cheapest form and lost on its own terms (§7.2), and
the data-side alternative was then beaten by a threshold.

### 7.5 The critical path moved to the extractor

*Added 2026-08-20, after §7.4.* With negative supervision superseded and MHT re-priced, the
remaining candidate is a **span-embedding router**: take `min(start)..max(end)` over an
event's own trigger and arguments, embed that block, and match a new observation against the
live filters. It is the first proposal that *builds* a representation rather than assuming
one, and the block is per-event and local — which is the discourse-attachment job that
proximity and type both failed at. On the Katrina passage the only named event the extractor
finds is `Hurricane Katrina`, so the block is Katrina-local, not Helene-local.

**We could not run it, because nothing available emits the input.**

The span architecture emits a *bag* of triggers and labels every one with the same role. No
threshold works: below 0.4 the Katrina block is the bare name, missing the 1,400 it should
bind; the Helene block runs the length of the passage and swallows Katrina; at 0.5+ Katrina
yields nothing. The boundary/mmBERT base yields nothing above threshold 0.3 on English
disaster copy and nonsense at 0.1 — trigger `"remote"`, with both `dead` and `location` bound
to `"Helene decimated"`.

**The cause is arithmetic, and finding it required correcting a claim we had just made.** A
first pass read the model card's row counts and said 68% of its event supervision was Chinese.
That was wrong: DocEE, ChFinAnn and DocFEE are not stored as events at all — they are
`entities` + `classifications`, so the doc-level type is a class label and the arguments are
NER spans, and counting them as event training flattered both sides of the ratio. Counting
only corpora that bind arguments to a trigger:

| | English | Chinese | English share |
|---|--:|--:|--:|
| the base as built | **798** (CASIE alone) | 20,884 | **3.7%** |
| every corpus available | 39,783 | 20,884 | 65.6% |

MAVEN and Mendeley are trigger-only. So that argument score is very nearly a Chinese-only
number, and English trigger→argument rests on 798 examples — which no threshold reaches.

> **Precision note added 2026-08-23.** This section originally quoted the base at
> "trigger 0.710 / argument 0.506". Those are **relaxed** figures on the model's *own* test
> set. Scoring a candidate's *strict* F1 against them — which we came close to doing — turns
> a 2× improvement into an apparent halving. Like-for-like on the shared blind test at a
> pinned threshold, the base is strict 0.7487 / 0.0913, fair 0.7523 / 0.4939.

A cold-start rebuild followed: 189,284 records, a 50× increase in English trigger→argument,
the Chinese corpora kept because they are why the argument head works at all. Gates on AP
prose rather than held-out DocEE, for the reason this section exists. The declared risk was
that 72% of the new English data is synthetic, on a programme whose recurring failure is
in-domain-good and real-news-zero.

**So the critical path is no longer the tracker or the association layer. It is the
extractor**, and that is a conclusion three failed association mechanisms had to be built
before anyone could reach.

### 7.6 The rebuild: better on every benchmark, and better at the job than the gate said

The rebuild trained cleanly — 6 epochs, 17.3 h, one A100, eval loss falling monotonically
to the last epoch. Then it failed the gates it was built to pass, and passed the ones it
was only meant not to break.

**On the wire copy, both pre-registered gates fail.** Scored over the registered 0.1–0.5
threshold range on 60 casualty-bearing Helene windows:

| gate | rebuild | incumbent |
|---|---|---|
| 1 — trigger + ≥1 bound argument on ≥50% of windows | **FAIL** 25.0% | **PASS** 65.0% |
| 2 — the Katrina block holds "1,400", not "Helene" | **FAIL** | **FAIL** |

**On held-out corpora it wins everywhere.** Both checkpoints scored by one command on one
machine, the same 11 files and 15,456 rows, threshold pinned to 0.5 — so no
different-test-set or different-operating-point confound survives:

| strict micro F1 | rebuild | incumbent | Δ |
|---|--:|--:|--:|
| entity | 0.6358 | 0.6200 | +0.0158 |
| relation | 0.0943 | 0.0350 | +0.0593 |
| classification | 0.6488 | 0.6328 | +0.0160 |
| structure | 0.0851 | 0.0755 | +0.0096 |
| event_type | 0.9542 | 0.9387 | +0.0155 |
| event_trigger | 0.7632 | 0.7487 | +0.0145 |
| event_argument | 0.1046 | 0.0913 | +0.0133 |
| event | 0.3752 | 0.3708 | +0.0043 |

Fair (Ortmann) entity / trigger / argument: 0.6732 / 0.7671 / 0.5508 against 0.6454 /
0.7523 / 0.4939. Structure swept to the record head's own thresholds — its maximum object
probability is 0.178, so a default 0.5 measures a cutoff it cannot reach — is 0.1184
against 0.1132.

**And then gate 1 turned out to be measuring the wrong thing.** It counts windows carrying
a trigger and at least one bound argument. It never asks whether the bound figure is
*correct*. Locating each window's gold death toll by character offset and calling a `dead`
argument a hit when its span overlaps:

| threshold | rebuild fired → hit (prec) | incumbent fired → hit (prec) |
|---|---|---|
| 0.50 | 3 → 3 (**100%**) | 0 → 0 (—) |
| 0.40 | 4 → 4 (**100%**) | 0 → 0 (—) |
| 0.30 | 5 → 4 (**80%**) | 4 → 0 (**0%**) |
| 0.20 | 6 → 4 (**67%**) | 10 → 0 (**0%**) |
| 0.10 | 12 → 9 (**75%**) | 39 → 3 (**7.7%**) |
| 0.05 | 20 → 16 (80%) | 55 → 16 (29%) |

**The incumbent's 65% gate-1 pass is 39 firings carrying three correct death tolls.** What
it binds to `dead` instead is `"car Hurricane Helene"`, `"Mexico"`, `"Pacific coast"`,
`"Carolinas"` — locations and fragments, not numbers. The rebuild's yield equals or beats it
at every matched threshold and triples it at 0.1, off one third the firings.

So the rebuild is the better extractor on wire copy as well as on corpora, and **the mix
change did what it was meant to do**. Two of our own instruments had to be corrected before
that was visible: the inert threshold sweep, and gate 1 itself.

**Gate 1's defect is structural, not a tuning error.** Scored as best-over-a-range against a
form-only criterion, it rewards a model for firing indiscriminately at a permissive
threshold. This programme already documents that error for A/B arms — see MODEL_LINEAGE's
matched-threshold caution — and had never applied it to the gates those arms are judged by.
A form gate needs a correctness companion; on its own it is a liveness check, not a score.

**What has not moved.** Yield is 15% at 0.1 and 26.7% at 0.05 — better, and still not a
front end the router can lean on. The remaining misses are mostly word-form golds (`six`,
`three`, `dozens`) where the model binds a nearby numeral (`160`, `70s`, `11`), which is a
normalisation gap rather than a binding failure. And gate 2 still fails for both models.

**The instrument, not just the model.** The claim that motivated the spend — "the incumbent
is ~0 at every threshold" — was an artifact of our own harness, which set
`Schema().events(trigger_threshold=…)`, a value the boundary greedy decode never reads.
Every row of the sweep therefore ran at the default 0.5, where 0.0% is the incumbent's real
score. Driving `extract(threshold=)` instead gives 0.0 / 0.0 / 8.3 / 20.0 / 65.0% across
0.5→0.1. The section's original premise survives — nothing usable at 0.3 and above,
nonsense at 0.1 — but the "~0 everywhere" strengthening of it did not, and it was the
strengthened form that justified a rebuild.

**Why more data cannot be the answer here.** `_decode_events` returns a single-element list
per event type and pools every trigger and argument into it; its own docstring records that
the mention path "carries no instance dimension". Measured on the incumbent, a passage
naming two hurricanes returns `n_event_instances=1`, with Helene's 246 and Katrina's 1,400
both bound as `dead` on the same event, at 0.1 and at 0.01. Gate 2 takes
min(start)..max(end) over that one pooled instance, so it rewards **sparsity rather than
binding** — which is why it only lands at the threshold where the rest of the output is
nonsense. That does not make gate 2 impossible; the incumbent does produce a local Katrina
block at 0.01, and that passage carries one Hurricane-typed event rather than two extracted
ones. It makes passing fragile and threshold-lucky, and no corpus teaches a model to emit
fewer spans on demand.

The pre-registered remedy for a gate-1 failure was to downsample the dominant synthetic
corpus first. We are not taking it — and the reason is now the opposite of the one we first
gave. The mix change **worked**: binding precision went from near-zero to 67–100%. There is
no real-news regression to repair. What remains is that neither model can separate two
events of one type, which is a property of the decode and not of the corpus;
`casualty_events` could not teach it in any case, carrying 8 event types and no named
identities. **The next constraint is the instance dimension — the record head — not another
corpus.**

That path has been tried once. The CASIE Tier 2 arm produced genuine multi-instance output
— 2–9 instances of one event type on 17 of 39 probed documents, structurally impossible
before it — and scored 0.0036 against a 0.2998 control, diagnosed as head-init: the record
head had never been supervised on events and got 375 steps to learn instance formation
cold. The fair test it asks for is a warm start on MAVEN followed by CASIE, with far more
steps.

## 8. Limitations

- **A pre-registered gate was scored by an instrument we had not verified.** The
  threshold sweep behind §7.6's gate 1 was inert for five measurements before anyone
  checked, and it produced the specific claim that justified a rebuild. The gate values
  were fixed in advance and the harness was tested against the incumbent — it "correctly"
  failed it — which is precisely why the defect survived: a harness that returns the
  expected answer is not audited. Pre-registration constrains the threshold, not the
  measurement apparatus.
- **Gate 1 measures form as though it were performance.** It counts a window as satisfied
  when a trigger and any bound argument appear, never checking whether the bound figure is
  the right one, and it is scored best-over-a-threshold-range. A model that fires
  indiscriminately at a permissive threshold therefore wins it: the incumbent's 65% is 39
  firings carrying 3 correct death tolls. We acted on that number for a day. Every form
  gate in this programme needs a correctness companion before it is used to compare models.
- **Gate 2 measures sparsity as though it were locality.** Taking min(start)..max(end)
  over a decode that pools all spans of one type into a single instance means a model is
  rewarded for emitting less. Both models "pass" it only where their output is otherwise
  unusable. The gate needs rewriting against a decode that carries an instance dimension.
- **One real event, one source per feed.** Both real evaluations draw ground truth from a
  single tracker page. The multi-source disagreement case — the most plausible remaining
  argument for a filter over a last-value baseline — is untested.
- **The scope-gate ratio was chosen after seeing contaminated values** on the event it was
  tuned against. The plateau across 1.5–2.5 mitigates this; it does not remove it.
- **Synthetic validation was circular in a specific way** (§6.1) and any future synthetic
  benchmark must generate trajectories from a process the filter does *not* model.
- **Cross-event contamination is unaddressed.** It is a distinct failure from scope, needs
  instance-level rather than type-level discrimination, and the one mechanism tried for it
  (§6.4) was negative.
- **A genuinely blind real test remains outstanding.** Both events evaluated here predate
  the model's knowledge cutoff.
- **The Helene observation set behind every number here is a cached artifact that cannot be
  regenerated** (§9.1). Comparisons among the published figures remain valid; placing a new
  model on the same scale does not.
- **Every association result here is bounded by an extractor that cannot express the
  alternative.** §7.5: no available model emits per-event trigger-and-argument spans on wire
  copy, so the span-embedding router is untested rather than refuted. A rebuild is in flight.
- **The plausibility ceiling of §7.4 is prior knowledge, declared per event.** It is not
  learned, it does not transfer to an event whose scale is unknown in advance, and on a feed
  with no large false positives it does almost nothing (5.247 → 4.283 on the archived set).

## 9. Reproducibility

### 9.1 The Helene observation set cannot be regenerated

**Every Helene number in this paper — 5.247 → 0.591, the oracle ceilings 0.537 and 0.480, and
the M5 sweep of §7.2 — is computed from one cached artifact,
`datasets/helene2024/_cache/tracked_rollup.json`, written 2026-08-10. That artifact cannot be
reproduced from any committed state of the repository.**

Established by elimination, 2026-08-20. The extraction model is unchanged on the Hub since
before the run; the CLI defaults are identical; a worktree at the contemporaneous commit
differs from current code by three observations out of ninety. The archived grid identifies
one undocumented flag (`--grid-step 12`). The blocker is elsewhere: **the `--rollup` flag did
not exist in any commit before the artifact was written**, and the rollup file itself was not
in the tree either — both were uncommitted working-tree state, committed later in a form that
does not reproduce it. There are no commits in the intervening window.

The signature is unambiguous. The archived run assigns every observation to a scope; every
reproduction leaves half of them keyed `unknown`, because the record head returns no location.
Reproductions also put a **94,000** into Tennessee, where the archived run's largest value
anywhere is 3,000.

What this does and does not cost:

- **Comparisons among the published figures stand.** They all read the same frozen artifact,
  so they are mutually consistent.
- **No newly trained model can be placed on that scale.** This blocked the pre-registered
  guard of §7.4 and forced a fresh baseline under one recorded invocation.
- The remedy is a new reference, not archaeology. The figures above should be read as
  belonging to a superseded observation set.

`run_pipeline.py` now writes its complete argument vector and git commit — with a `-dirty`
marker — into every output. That marker is the load-bearing part: the archived run came from a
tree matching no commit, and nothing in the artifact said so. This is the second time
provenance has blocked this programme; the Türkiye–Syria re-extraction is stalled the same way
(`EKF_MHT_BUILD_RECORD.md` §25.5).

### 9.2 Code and data

Data and code live under `datasets/disaster_streams{,_hard,_sonnet5,_model}/`: `generate.py`
(parametric streams, with a `--regime` switch), `realize.py` (LLM text via a batch API, with
`--batch-id` recovery), `model_arm.py` (extraction, caching raw output for `--from-raw`),
`extract.py` (surface parser and normalizer), and `evaluate.py` (tracker, baselines and
gate). Evaluation:

```
uv run python datasets/disaster_streams/evaluate.py --data <dir> --split <val|test> \
    [--min-conf 0.99] [--reject-sigma 4] [--learn-gate]
```

Always pass the trivial baseline alongside; `evaluate.py` reports `est_last_value` beside
`est_ekf` for the reason given in §4.

Known-negative settings, recorded so they are not retried: a symmetric 3σ innovation gate
is catastrophic on rising tolls (use the one-sided, dynamics-aware gate); confidence-as-soft
measurement noise does **not** beat a hard confidence cut, because zero-shot extraction
errors are gross false positives rather than graded noise; and record-mode precision
requires the boundary architecture, since the span decoder ignores structure mode.

The build record — attachment points, CLI, training plan, decision log, generator
specification, the blind-test protocol, and the full negative-sampling implementation notes
— is in `EKF_MHT_BUILD_RECORD.md`.

## 10. References

- Kalman, R.E. (1960). *A New Approach to Linear Filtering and Prediction Problems.*
  Journal of Basic Engineering 82(1), 35–45.
- Reid, D.B. (1979). *An Algorithm for Tracking Multiple Targets.* IEEE Transactions on
  Automatic Control 24(6), 843–854.
- Bar-Shalom, Y., Willett, P., Tian, X. (2011). *Tracking and Data Fusion: A Handbook of
  Algorithms.* YBS Publishing.
- Blackman, S.S. (2004). *Multiple Hypothesis Tracking for Multiple Target Tracking.* IEEE
  Aerospace and Electronic Systems Magazine 19(1), 5–18.
- Kuhn, H.W. (1955). *The Hungarian Method for the Assignment Problem.* Naval Research
  Logistics Quarterly 2(1–2), 83–97.
- Zaratiana, U., Pasternak, G., Boyd, O., Hurn-Maloney, G., Lewis, A. (2025). *GLiNER2:
  Schema-Driven Multi-Task Learning for Structured Information Extraction.* EMNLP 2025
  System Demonstrations. https://aclanthology.org/2025.emnlp-demos.10/
- Solatorio, A.V. (2024). *GISTEmbed: Guided In-sample Selection of Training Negatives for
  Text Embedding Fine-tuning.* arXiv:2402.16829. https://arxiv.org/abs/2402.16829
