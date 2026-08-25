# Every gate in the EKF pipeline: what it does, what type it is, and whether it is on

Written 2026-08-25. Three different things in this project are called a "gate" and they
live at different layers. Conflating them has already inverted one verdict for a day
(see the gate-1 lesson at the bottom), so they are separated here by name and by type.

## Type legend

| Type | Meaning |
|---|---|
| **NEURAL** | a forward pass decides it (GLiNER2 head output) |
| **LEARNED** | parameters fitted by gradient descent on a training objective, but not the extractor |
| **THRESHOLD** | a cut applied to a neural score. The score is learned; the cut is hand-set |
| **DECLARED** | reads knowledge a human wrote down (a hierarchy, an ontology). No model, no statistics |
| **HEURISTIC** | hand-written rule over text (keywords, character distance, nearest-match) |
| **ARITHMETIC** | a closed-form inequality on numbers already in the stream |
| **STATISTICAL** | derived from the event's own distribution at run time |
| **DECODE** | one decision over the WHOLE sequence, not per observation. Can revisit |
| **ORACLE** | consults ground truth. Prices a ceiling; can never ship |

---

## The flow

Article text in, trajectory out. Each stage can only remove or re-route what the stage
above it passed.

```
  raw article feed
        |
  [0] relevance gate ............ NEURAL      ON      drop non-mass-casualty articles
        |
  [1] stage-1 event extraction .. NEURAL      ON      events, envelopes, places
        |
  [2] extract(threshold=) ....... THRESHOLD   ON      candidate span score cut
      abstention head ........... LEARNED     ON      "does this label appear at all?"
      record_*_threshold ........ THRESHOLD   ON      anchor / field selection
        |
  [3] stage-2 casualty records .. NEURAL      ON      value + role + location per record
        |
  [4] normalize() ............... HEURISTIC   ON      qualifier + source from keywords
        |
  [5] out_of_window() ........... HEURISTIC   ON      reject figures dated before the event
      association_key() ......... HEURISTIC   ON      bind a figure to a place
        |
  [6] rollup .................... DECLARED    ON      map places into the event hierarchy
      scope_filter() ............ DECLARED    opt-in  drop places outside the event
        |
  [7] apply_extracted_scope() ... NEURAL      opt-in  route on the model's `scope` field
      hmm_gate() ................ DECODE      SHIP    own / aggregate / reject, globally
      viterbi_gate() ............ DECODE      -       hmm_gate without the extra features
      gate() upward ............. ARITHMETIC  legacy  keep / reroute / drop by magnitude
      gate() downward ........... ARITHMETIC  OFF     drop stale readings (down_ratio)
      hmm_gate4() ............... DECODE      OFF     + a downward-revision state (inert)
      plausibility_filter() ..... HEURISTIC   opt-in  hand-set per-event ceiling
      tail_filter() ............. STATISTICAL opt-in  median + k*MAD on log10, upper tail
        |
  [8] EKF update ................ STATISTICAL ON      relative-noise Kalman filter
      STUDENT_T_NU .............. STATISTICAL OFF     robust one-sided measurement model
      REJECT_SIGMA .............. STATISTICAL OFF     one-sided innovation gate
      MAX_RATE .................. ARITHMETIC  OFF     impossible-accrual-rate filter
      CONF_R .................... STATISTICAL OFF     fold extractor confidence into R
      CENSOR_AT_LEAST ........... ARITHMETIC  ON      `at_least` below estimate is a no-op
      --learn-gate router ....... LEARNED     opt-in  logistic blend of EKF vs last_value
        |
  trajectory per (place, role)
```

---

## Stage by stage

### [0] Relevance gate — NEURAL, on
`run_pipeline.build_gate_schema` -> `.classification("relevance", GATE_LABELS_V2)`.
Zero-shot classification, two labels: `mass_casualty` vs `other`. v1 described the
negative class as topically distant filler and admitted **58.5%** of definitively
non-disaster text at high confidence when benchmarked against 21k real annotated
messages. v2 describes the negatives it actually meets -- personal requests, policy
news, aid-logistics inventories carrying huge numbers, single-casualty medical items --
and names two traps verbatim: a lone death is not a mass-casualty event, and disaster
words used metaphorically ("explosion in crowdfunding").

**It does not filter non-English at all, and the failure is silent.** Measured on 200 clean
Turkish news articles, English label descriptions against Turkish text:

    fastino/gliner2-base-v1   199/200 = 99.5% false admits   <- THE SHIPPED DEFAULT
    fastino/gliner2-multi-v1   56/200 = 28.0%

The default is DeBERTa-v3, vocab 128,011, English-only. It cannot read the text, so it
answers `mass_casualty` to nearly everything — worse than the 58.5% that forced the v1→v2
rewrite, and nothing in the pipeline reports that the gate has stopped discriminating.
Translating the articles into English does NOT fix it (17/60 still admitted by the English
model on English text), so the remaining fault is the label descriptions, not the language.
Check the encoder before pointing this gate at any non-English feed.

### [2] Extraction thresholds — THRESHOLD, on
`extract(threshold=)` is the single global cut the boundary greedy path gates on.
**Trap:** `Schema().events(trigger_threshold=, argument_threshold=)` is read only by the
*span* engine. On the boundary path those values are silently ignored -- an entire
threshold sweep once ran at the default 0.5 without moving. `record_anchor_threshold`
and `record_field_threshold` default to 0.5 and are separate cuts on the record head.

### [2] Abstention head — LEARNED, on
`enable_abstention=True`, `abstention_threshold=0.5`, trained by `abstention_loss` at
weight 0.2. Per **query**, not per mention: the target is 1 when a label has no mentions
at all. It answers "does `dead` appear in this text?", **not** "should this particular
figure be rejected" -- so it is not a per-observation reject and cannot be used as one.

### [4] normalize() — HEURISTIC, on
`_detect_qualifier` / `_detect_source` keyword rules produce `qualifier`
(point/about/interval/feared/at_least) and `source` (official/major_outlet/preliminary).
This is the pipeline's weakest normalized field: 0.724 zero-shot, 0.691 after
fine-tuning. `--normalizer classify` swaps in a NEURAL alternative and `--normalizer
both` scores them against each other on the same feed.

### [5] out_of_window() — HEURISTIC, on
Takes the year of the nearest date by character distance and rejects the figure when
that year predates the event. Rejects only on **positive** evidence -- an absent or
unparsable date returns None and the figure is kept, because treating absence as
evidence would discard most of the feed. Nearest-by-character-distance is a weak proxy
for attachment and it *failed* for location (both countries within 26 chars of both
numbers on the Turkiye standfirst). It is defensible for dates only because dates are
sparse and clustered, so competing hypotheses sit far apart. Measured: 13/15 Izmit
envelopes resolve to "August 1999", and no genuine observation resolves to an old year.

**Two later fixes, both from feeds that broke an unstated assumption.**

*A feed with no dates at all.* The gate reads date spans from the events block and returns
None on its first guard when there are none — so on the Türkiye 2023 feed, whose events
block carries only event_type/confidence/casualties/location/cause, it was structurally
unable to fire while the 1999 İzmit toll of 17,500 sat in the 2023 observation stream. Now
falls back to scanning the raw text for bare years.

*Competing dates that are NOT far apart.* On the Aegean feed the nearest year to İzmit's
17,000 is **2020 at +117 chars**, beating **1999 at −152**, so nearest-by-distance returns
the current year and misses a 143× contaminant. 117 against 152 is not "far apart", and
this docstring's own justification does not hold there. `mode="any"` takes any
out-of-window year within a radius instead — lower precision, and the right shape for
additive evidence in the decode's emission rather than for a veto.

### [6] scope_filter() — DECLARED, opt-in
Rejects observations keyed to a place outside the event's declared hierarchy. Run after
the rollup. **The cheapest cross-event filter available, and it beats every learned
signal tried:**

| | cross-event caught | false positives |
|---|---|---|
| scope membership (no model) | 4/6 | **7.3%** |
| best learned signal, one call/obs | 4/6 | 31.7% |

It works because the contaminating events happened *somewhere else* -- Mexico, Puerto
Rico, Bosnia, Reading PA -- which is declared knowledge, not a statistical property.

### [7] hmm_gate() — DECODE, the recommended replacement for the ratio gate

`scope_gate.hmm_gate` decides the whole stream at once over three states — `own`,
`aggregate`, `reject` — instead of committing per observation. It is `viterbi_gate` plus
per-observation REJECT evidence from outside the magnitude channel (an out-of-window date,
a place outside the declared hierarchy, a syndication marker), so those gates ARGUE rather
than veto. Recommended σ=0.3, reject_cost=4.0, stay=0.1, **warmup=0**.

Measured at that one setting on every event we have:

| event | ratio gate | decode | |
|---|---|---|---|
| Helene | 29.3 | **20.7** | −29.4% |
| Türkiye–Syria | 11,581.5 | **10,695.5** | −7.6% |
| Aegean 2020 | 74.4 | **15.7** | −78.8% |

Two properties are load-bearing and both were predicted by the three-way oracle:
**global** (a greedy rule commits per observation, and one large figure admitted early
poisons a stream's running scale for everything after) and **able to reject** (assignment
headroom is measured at zero; the entire residual is the null hypothesis).

**Design rule, measured:** keep every feature weight BELOW `reject_cost`, so no single
feature can force a reject on its own — it can only tip a case magnitude has already made
marginal. The sweep shows a cliff exactly at that boundary.

**Two traps that cost real time, both from importing assumptions:** `warmup`, copied from
the greedy gate, pins the first readings to `own` and reintroduces exactly the commitment
the decode exists to remove — it alone flipped Türkiye from a loss to a win. And a MISSING
reference must not be read as evidence a value is too large: on a feed with no aggregate
stream `natl=0` made every value score above the whole event and dropped 52 of 53
observations at zero feature weight.

### [7] gate() — ARITHMETIC, legacy; superseded by hmm_gate
Three outcomes against the running larger-scope reference, per stream in time order:

```
keep     v < natl / ratio        plausibly this place's own toll
reroute  natl/ratio <= v <= natl*ratio    it IS the national figure -> __aggregate__
drop     v > natl * ratio        no scope in this event can exceed the whole
```

Reclassify, do not discard: rerouting a rejected figure to `__aggregate__` keeps the
national signal, which is the project's one honest measurement. `reference_for` chooses
what "natl" means -- `aggregate` (works while a part is small relative to the whole),
`global-max` (needed where no aggregate stream exists, but circular for the dominant
stream), or `implied`.

Helene at ratio 2.0: pooled 314.5 -> **29.3 deaths**, a 10.7x reduction.

`down_ratio` adds the second side (drop a reading far below the stream's own running
max, since a toll does not fall). **Measured 1-for-2 and left OFF:** Helene 29.3 ->
21.7, Turkey 14765 -> 15349 (worse). See `scope_field_results/two_sided_gate.txt`. The
global decode reaches the same place on Helene (20.7) and wins on Turkiye too, which is
why this knob is not needed.

### [7] plausibility_filter vs tail_filter — HEURISTIC vs STATISTICAL, both opt-in
`plausibility_filter` is a hand-set ceiling and has to be *told* the event's scale --
which it gets from the answer, so it is not a method. `tail_cut` derives the cut from
the event's own observations (median + k*MAD on log10, upper tail only), which is
scale-free and uses no ground truth.

### [8] The EKF and its gates — STATISTICAL, mostly off
Measurement noise is **relative**: `R = (sig * max(ref,1))^2`, with `sig` from
`SRC_REL_SIGMA` (official 0.06 / major_outlet 0.12 / preliminary 0.25) scaled by
`QUAL_FACTOR` (point 1.0 ... feared 2.5). Process noise grows as
`q_rel * max(mu,1) * dt`, so real jumps stay admissible between reports.

- **STUDENT_T_NU** (OFF): a Student-t measurement model, applied as one-step IRLS
  reweighting (`R` inflated by `w = (nu+1)/(nu+d^2)`) and ONE-SIDED, because the physics is
  — a rising toll may legitimately surge above the estimate, only a reading far below it is
  implausible. The symmetric textbook form is much worse on both events. Measured 1-for-2
  (Helene −1.7 deaths, Türkiye +651) and it retires none of the thresholds below, which was
  the reason to want it: with the scope gate off, Helene is 314.5 under every nu tested.
  Whatever the gate catches, it is not a fat tail.
- **REJECT_SIGMA** (OFF): one-sided innovation gate. A rising toll is non-decreasing, so
  it rejects only readings implausibly *below* the estimate; the decay roles invert it.
- **MAX_RATE** (OFF): drops an observation whose upward accrual rate exceeds the limit --
  the impossible jump the one-sided gate *admits* by construction, since that gate only
  rejects lows.
- **CONF_R** (OFF): folds extractor confidence into R, so low confidence widens the noise
  instead of hard-dropping the reading.
- **CENSOR_AT_LEAST** (ON): treats an `at_least` reading below the estimate as
  uninformative. Logically correct for a strictly rising toll -- "at least 96" is
  consistent with 123 -- and **measured wrong on Helene**, where ground truth falls four
  times in North Carolina and three times nationally as deaths are reclassified. The
  measured negative is recorded at the flag definition.
- **--learn-gate** (opt-in): the only genuinely LEARNED component outside the extractor.
  A logistic router `alpha = sigmoid(w.x)` over 8 features (staleness, source
  unreliability, qualifier coarseness, censored-bound flag, decay-vs-rise role,
  EKF-vs-last_value disagreement, reports seen so far, bias), fitted by gradient descent
  on peak-normalized blend MSE. It replaces the hand-set `SRC_TRUST` / `QUAL_TRUST` /
  `GATE_TAU` tables with fitted weights.

---

## The oracles — ORACLE, never shippable

All three consult ground truth. They price ceilings so we know whether a component is
worth building before building it.

| Oracle | What it does | What it prices |
|---|---|---|
| `oracle_gate` | sends every observation to its own place or to Total, whichever is closer to truth | perfect **assignment**, no reject option |
| `oracle_gate_three_way(tol)` | adds a third outcome: reject when relative error against **both** scopes exceeds `tol` | perfect assignment **plus** a reject option |
| `stream_ceiling` | at each grid point takes the observation closest to truth | method quality vs **coverage** |
| `random_control(n, trials)` | removes n observations at random | the null: does dropping *any* n help? |

**The measurement that matters.** On Helene the two-way oracle scores 29.3 -- identical
to the shipped gate. So **assignment headroom is zero**; there is nothing to win by
associating better. The three-way oracle reaches 17.6 at tol 0.25, and it gets there
purely by dropping more (106 kept -> 76). All ~11.7 remaining deaths are in the reject
option, and 63% of those rejects are stale readings *below* truth, which the upward-only
gate cannot see. See `scope_field_results/reject_headroom.txt`.

---

## What these gates are actually filtering

Eleven percent of Helene's `dead` observations are hand-audited **non-casualty**, and they
split into two kinds that need different fixes:

| kind | examples | fix |
|---|---|---|
| **exposure counts** | 300 rescued, 50 patients rescued, 32 evacuated, 11 swept away | a schema role |
| **unit confusion** | a two-day period, six states, dozens of vehicles, 1,400 landslides | a schema role |

A rescued person is a **counterfactual casualty** — averted harm, not realized harm. And
exposure counts run systematically LARGER than casualty counts in disaster copy (505
displaced against 6 dead; 300 rescued against Florida's true peak of 26), so a mis-bind is
the same magnitude as cross-event contamination and points the same way, upward. The gates
therefore catch some of them — for the wrong reason, magnitude rather than category, which
is the gate-1 lesson again.

**The root cause is upstream of every gate on this page.** `casualty_events` has exactly
four roles — `location`, `injured`, `missing`, `dead` — and no role for exposure, while the
prose is full of it. A number with no correct home lands in a wrong one. Adding `displaced`
and `rescued` removes the ambiguity at source instead of filtering it downstream. Full
taxonomy in `ekf_showcase/gate_results/EXPOSURE_VS_CASUALTY.md`.

## Not pipeline stages: the pre-registered gates 1-4

`frontend_gates.py` uses "gate" for **acceptance tests on a model**. They gate a spending
decision, not an observation. Sweep fixed before spending: thresholds 0.5-0.1 registered,
0.05/0.01 diagnostic only.

| | Test | Bar |
|---|---|---|
| 1 | usable events-form on the Helene feed: a trigger AND >=1 bound argument | >= 50% of windows |
| 2 | the span block is LOCAL: on the Katrina passage it must hold 1,400, not Helene's figure | pass/fail |
| 3 | event_trigger / event_argument F1 on the shared blind test | >= the incumbent's |
| 4 | entity / relation / structure F1 | not below the 137k-clean reference |

**The standing lesson.** Gate 1 counts *firings*, not correct ones. Scored best-over-a-
range it rewards indiscriminate firing, and it inverted a verdict for a day: the
incumbent's "65%" was 39 firings carrying **three** correct tolls, while the rebuild it
rejected binds the right figure 67-100% of the time. Pair every form gate with a
correctness companion -- that is what `binding_accuracy.py` is for.

---

## Should [5] and [6] be merged into one LEARNED gate?

Measured 2026-08-25 (`gate56_composition.py`), against the 86 hand-audited Helene
occurrence labels. **No -- but the current composition is the worst term in the system
and should be fixed.**

They are genuinely complementary. All six cross-event figures, and what catches each:

| value | keyed to | in declared scope | nearest-date year | caught by |
|---|---|---|---|---|
| 2 | mexico | no | - | [6] |
| 3000 | puerto rico | no | 2020 | [6] + [5] |
| 32 | tennessee | **yes** | - | **neither** |
| 80 | north carolina | **yes** | **1916** | **[5] only** |
| 16 | bosnia | no | - | [6] |
| 1400 | reading pennsylvania | no | 2005 | [6] + [5] |

So [5] catches one figure [6] structurally *cannot* -- the 1916 hurricanes' "80", keyed
to North Carolina, which is legitimately in scope. Series composition is already OR, and
it reaches 5/6.

But the trade is bad:

| rule | caught | false-rejects of genuine |
|---|---|---|
| [6] scope only | 4/6 | 6/81 = **7.4%** |
| [5] date only | 3/6 | 10/81 = 12.3% |
| UNION (as shipped) | 5/6 | 16/81 = **19.8%** |
| INTERSECTION | 2/6 | 0/81 = **0.0%** |

**[5]'s marginal contribution over [6] alone is +1 catch for +10 false rejections.** It
is a hard reject (`continue`), not a flag, so that cost is realised.

Three reasons not to make the merge *learned*:

1. **It was already tried and lost.** Best learned signal, one call per observation:
   4/6 at 31.7% FP, against declared scope's 4/6 at 7.4%. Same recall, 4.3x the false
   positives. Replacing knowledge that is written down and correct with a statistical
   estimate of it is strictly worse.
2. **The one figure neither gate catches is not a filter failure.** "Typhoon headed to
   Taiwan injures dozens" was bound to Tennessee because "11 workers at a Tennessee f..."
   follows it. That is `association_key`'s nearest-place heuristic failing at [5] -- the
   same failure mode `out_of_window`'s own docstring cites for the Turkiye standfirst. No
   filter downstream of a bad binding can repair it.
3. **There are six positive examples.** A learned gate cannot be fitted on that, and the
   audit set is 86 rows total.

**What to do instead:** the operating point between AND (0% FP, 2/6) and OR (19.8% FP,
5/6) is unexplored, and [5] is the miscalibrated half. Either require the old date to be
nearer the span than any current-year date before rejecting, or demote [5] from hard
reject to a feature and let [6] keep the veto. Both are arithmetic.
