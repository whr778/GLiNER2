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
      gate() upward ............. ARITHMETIC  ON      keep / reroute / drop by magnitude
      gate() downward ........... ARITHMETIC  OFF     drop stale readings (down_ratio)
      plausibility_filter() ..... HEURISTIC   opt-in  hand-set per-event ceiling
      tail_filter() ............. STATISTICAL opt-in  median + k*MAD on log10, upper tail
        |
  [8] EKF update ................ STATISTICAL ON      relative-noise Kalman filter
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

### [7] gate() — ARITHMETIC, upward on / downward off
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
21.7, Turkey 14765 -> 15349 (worse). See `scope_field_results/two_sided_gate.txt`.

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
