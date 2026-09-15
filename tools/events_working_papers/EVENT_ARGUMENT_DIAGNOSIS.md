# Why `event_argument` sits at 0.118, and what would move it

Status: **diagnosis, measured 2026-09-15** on `whr778/gliner2-eb16-rebuild-tr`, 18,786-record
blind test, greedy decode, threshold 0.5. Every number here is from that run's
`test_metrics.json` or from counting the test splits directly. Companion to
[[JOINT_IE_SCALING]] (Tier 2) and [[PAPER_0_FOUNDATION]] §10.

---

## 1. The model finds the arguments. It cannot attach them to the right event.

Same checkpoint, same predictions, three scoring keys:

| key | what it requires | F1 | P | R |
|---|---|--:|--:|--:|
| **strict** | (event_type, **trigger**, role, entity) | **0.1178** | 0.149 | 0.097 |
| **relaxed** | (event_type, role, entity) — trigger link dropped | **0.5783** | 0.701 | 0.492 |
| **fair** | relaxed + partial credit for boundary errors | 0.5737 | 0.707 | 0.483 |

**Dropping one element of the tuple multiplies the score by roughly five.** And
`fair ≈ relaxed` (0.574 vs 0.578) says partial boundary credit adds almost nothing on top
of relaxed — so boundaries are not where the loss is.

The error taxonomy over 18,557 gold arguments agrees:

```
COR          8,262   44.5%   correct
FN           9,059   48.8%   never proposed at all
FP           3,212
BES+BEL+BEO    988    5.3%   boundary errors
LE+LBE         248    1.3%   label errors
```

**Boundary and label errors together are 6.6%.** This is not a span problem and not a
typing problem. It is a *binding* problem, plus a large block of arguments that are never
emitted.

The per-event keys say the same thing from the other direction: **exact argument-set rate
0.0090** and **mean Jaccard 0.061** over 7,908 events. Almost no event is recovered
*completely*, while pooled relaxed micro sits at 0.578 — the model is getting many
individual arguments right and very few whole events right.

## 2. The mechanical cause: the mention path pools same-type events

On the mention path an events group is compiled as `[V]` trigger + role queries, and a
document gets **one instance per event type**. Two `Attack` events in one document collapse
to a single `Attack` trigger, so the arguments of the second either vanish or bind to the
first — and strict scoring keys on the trigger.

Measured on this blind test's own event corpora:

```
documents carrying events : 3,186
gold event instances      : 17,135
sharing a type in-document: 10,997   (64.2%)
```

by corpus: **cmnee 1,606 documents**, maven 346, casie 102.

**Two-thirds of gold event instances are structurally inexpressible by the decoder that
was measured.** The surviving third puts a ceiling of roughly `0.578 × 0.36 ≈ 0.21` on
strict argument F1; the observed 0.118 is the same order. That is consistency, not proof —
but it is the only hypothesis on the table that predicts the 5× strict/relaxed gap, the
negligible boundary error rate, and the near-zero exact-set rate simultaneously.

**AND THE CONFIG COMMENT ACTIVELY MISLEADS.** `eb16-rebuild-tr.yaml` sets
`enable_records: true` with the note *"load-bearing: events decode as trigger + role
edges"*. That describes JOINT DECODE-TIME behaviour and is wrong about TRAINING: without
`event_records`, an events group never reaches the record head at all. A reader checking
whether events were trained on the record path would read that line and conclude they were.

`compile_record_specs` already carries the upstream version of this number: the cap costs
**78.8% of gold instances on CASIE, 62.5% on WikiEvents, 38.3% on MAVEN, and 0.0% on RAMS**
(RAMS is 100% single-event documents — which is exactly why the RAMS-based argument curves
never surfaced this).

## 3. Was `event_records: true` ever configured? NO — and that is the answer to the obvious objection

Events *were* trained with mmBERT. The objection "so the head has seen events" is right
about the **mention** path and wrong about the **record** path, and the distinction is the
whole diagnosis.

`event_records` is a `boundary_head` setting, default **False**. Verified 2026-09-15:

| where | occurrences of `event_records` |
|---|--:|
| `config/base/eb16-rebuild-tr.yaml` | **0** |
| every config under `config/base/` | **0** |
| configs setting it **true** anywhere | **2** (both archived Tier 2 arms) |

So in every base run, events were supervised as mention-path queries and the **record head
was supervised on `json_structures` only. It has never been trained on an event.** That is
not an inference; it is what the configs say.

## 4. Why Tier 2 already failed, and what that does and does not prove

`event_records: true` routes events through the record head, which is multi-instance by
construction and lifts the cap. It was tried twice and both arms are archived:

- **CASIE Tier 2** — multi-instance events *worked structurally* and scored **0.0036
  against a 0.2998 control**.
- **MAVEN Tier 2** — trigger strict **−0.008**; nothing gained.

The recorded cause is head initialisation: the record head had no event competence to
decode with. **That is a confound, not a refutation.** Head-init is the largest single
effect this programme has measured — fresh → IE-pretrained heads moved RAMS arguments
**0.042 → 0.462**, eleven-fold ([[PAPER_0_FOUNDATION]] §10.5).

Switching the path and the head's competence at the same time measures their sum. The
arms did that, and the sum was negative.

## 4b. The OTHER half: argument recall is TRIGGER recall, and nothing has ever been calibrated

Sections 1–4 are about binding. This section is about the ceiling binding converges onto,
and it changes what "fixing event arguments" means.

**The recall cascade, same run, strict unless noted:**

| head | precision | recall |
|---|--:|--:|
| `event_type` | **1.000** | 0.606 |
| `event_trigger` | 0.783 | 0.493 |
| `event_argument` (relaxed) | 0.701 | **0.492** |

**Argument recall (0.492) is trigger recall (0.493).** The argument head is not
independently failing to find things — it inherits the cascade. An argument whose trigger
was never detected cannot be attached to it. So *"improve arguments"* mostly means
**improve triggers**, and triggers inherit from type detection above them.

**`event_type` precision is EXACTLY 1.000 at recall 0.606.** A head that never emits a
wrong type while missing 39% of them is not performing well; it is sitting far above its
optimal operating point. That is a calibration signature, not a capability one.

**AND NOTHING HAS EVER BEEN SWEPT.** Every number in this document — 0.1178, 0.5783,
0.6051, 0.7545 — is a single-point reading at **threshold 0.5**. `evaluate_config` calls
`_run_blind_test` directly and does not re-sweep, and the decode-arms run passed
`--threshold 0.5` explicitly. `sweep_record_thresholds.py` exists but sweeps the RECORD
head for structures, and its own docstring says `threshold_sweep` *"sweeps the general
decision threshold; it does not touch `record_anchor_threshold` / `record_field_threshold`,
so nothing has ever calibrated this head."*

**This project's most expensive recorded lesson is exactly this shape.** The stage-0
relevance gate ran its whole life at 0.5, needed 0.998, and moving it bought overall
accuracy 0.719 → 0.847 and the worst class 0.444 → 0.903 — *more than any training
intervention*, after two GPU fine-tuning runs had been spent on that class
([[RESEARCH_PROGRAM]] §2).

### Two independent levers, and the cheap one is untried

| lever | attacks | cost | status |
|---|---|--:|---|
| `event_records: true` | **binding** — the strict/relaxed gap | ~$28 | training 2026-09-15 |
| **threshold sweep** | **recall** — the ceiling itself | **~$2** | **never run** |

They are complementary rather than competing: binding converges strict *onto* relaxed, and
the sweep moves relaxed. **So the "~0.58 ceiling" stated in §5 is the ceiling AT THRESHOLD
0.5 and may not be the model's ceiling at all.**

### Method constraint, because this programme has retracted a finding for getting it wrong

**Pick the threshold on VALIDATION; score the blind test ONCE.** Sweeping on test and
quoting the best is fitting the test set. [[RESEARCH_PROGRAM]] §5: *"a comparison between
two models at a shared arbitrary threshold measures the gap between two operating points,
not between two models."* The same applies to a model against itself.

### What either outcome means

- **Relaxed recall moves materially** → part of the "48.8% never proposed" is calibration,
  not capability, and **the incumbent's honest baseline is higher than 0.1178** — which
  makes the event-records base's win *harder* to claim. That is the reason to run it before
  the base lands rather than after.
- **It does not move** → the ceiling is real, argument recall is a genuine training problem,
  and §4b becomes the next target after binding.

## 4c. OneIE solved this upstream — and its metric is not ours

Read from the paper (Lin, Ji, Huang & Wu, ACL 2020, `2020.acl-main.713`) on 2026-09-15,
prompted by the question "didn't OneIE solve this?". It did, and the answer has two halves,
of which the second matters more to us.

### The architectural half: OneIE never creates the pooling problem

OneIE identifies triggers by **token-level BIO tagging with a CRF**:

> *"We use a feed-forward network FFN to compute a score vector for each word... After that,
> we use a conditional random fields (CRFs) layer to capture the dependencies between
> predicted tags... We use the BIO tag scheme."*

So **one node per trigger SPAN**, not one per event type. Two `Attack` events in a sentence
are two nodes carrying two sets of argument edges, natively. There is no cap to lift because
the representation never pools. The famous global features — cross-subtask
(`DIE-VICTIM-GPE`) and cross-instance (*"a VICTIM of a DIE event is likely to be a VICTIM
of an ATTACK event in the same sentence"*) — are a refinement **on top of an
already-unpooled graph**, not the fix for pooling.

**That is a second, independent answer to the same problem.** `event_records: true` gives
the record head multiple instance queries; OneIE tags trigger spans and lets the count fall
out. Worth holding as an alternative if the record-head route disappoints.

### The metric half: our STRICT is stricter than the literature's

OneIE's argument criterion, verbatim:

> *"An argument is correctly identified (Arg-I) if its **offsets and event type** match a
> reference argument mention. It is correctly classified (Arg-C) if its **role label** also
> matches."*

**Offsets + event type + role. There is no requirement that the specific TRIGGER match.**

Our two metrics bracket that criterion; neither equals it:

| metric | span requirement | trigger required? | score |
|---|---|---|--:|
| our **strict** | exact, case-sensitive | **YES** | 0.1178 |
| **OneIE Arg-C** | **exact offsets** | no | — |
| our **relaxed** | **overlap** (substring or shared content token) | no | 0.5783 |

- **strict adds a requirement OneIE does not have** (trigger identity), so 0.1178 is a
  LOWER bound on an OneIE-comparable number.
- **relaxed drops a requirement OneIE does have** (exact spans; ours accepts
  `New York City` ↔ `New York`), so 0.5783 is an UPPER bound.

**The OneIE-comparable figure lies between them and we do not currently compute it.**

### What follows from that

1. **`0.1178` must never be quoted as "our event-argument F1" against published work.** It
   is a deliberately stricter criterion. Every external comparison needs the bracketed pair
   or, better, the missing metric.
2. **Measuring trigger-level binding is still right FOR US.** The EKF needs a figure bound
   to the right event instance, not merely to the right event type — so strict is the
   metric this programme actually cares about, even though it is not the field's.
3. **ACTION: add an Arg-C metric** — exact surface, type + role, no trigger requirement. It
   is a scorer change, costs no GPU, and is the only way this line can be compared to the
   event-extraction literature at all.
4. **The "catastrophic 0.118" framing overstated the gap against the field**, and that is my
   error to correct: relaxed at 0.578 is roughly where the field's own criterion already
   places this model.

### 4c-ii. What OneIE would cost us — the objections, before anyone adopts it

§4c calls span-tagged triggers "the other road". It is a road with tolls, and they are
recorded here so the fallback is evaluated rather than reached for.

**1. IT IS SENTENCE-LEVEL. Our problem is not.** Verbatim: *"our ONEIE framework extracts
the information network **from a given sentence**"*. The corpora that dominate our argument
mass — CMNEE, DocEE, ChFinAnn, DocFEE — are **document-level**, and the boundary
architecture exists partly to reach beyond the sentence ([[COUNTING_LAYER]]). OneIE's own
error analysis lists *"Cross-sentence reasoning"* as a **residual error category it does not
solve**. Adopting its representation means either restricting to sentences or extending a
design that was never document-level.

**2. BIO TAGGING NEEDS A CLOSED TAG SET, AND THAT IS ARCHITECTURALLY INCOMPATIBLE WITH
GLiNER2.** The trigger head computes *"a score vector ŷi = FFN(xi) for each word, where
each value in ŷi represents the score for a tag in a **target tag set**"*. The tag set is
fixed at training time.

**GLiNER2's premise is the opposite: labels are an INPUT at inference** — it is why
`CLAUDE.md` insists one concept has one spelling across every corpus, why `labels_file`
exists, and why the query protocol emits a marker per schema field. A BIO tagger cannot
accept an event type it was not trained to tag. **Adopting OneIE's trigger head would
forfeit schema-driven extraction**, which is the thing that makes this model worth having.
Any borrowing has to take the *graph* idea without the *closed-vocabulary tagger*.

**3. IT HAS ITS OWN MULTI-INSTANCE RESIDUAL, IN THE OTHER DIRECTION.** *"Multiple events
per trigger"* is a named category in OneIE's remaining-error distribution. One node per
trigger span solves *many triggers of one type* — our problem — and leaves *one trigger
belonging to several events* unsolved. The span representation is not a general answer to
event multiplicity; it trades one failure for another.

**4. IT REQUIRES ENTITY ANNOTATION WE DO NOT ALWAYS HAVE.** Argument edges connect trigger
nodes to **entity mention nodes**, so the model needs entity gold in the same sentences.
Several of our event corpora supply argument spans with roles and no entity typing; using
OneIE's formulation would need that annotation invented or inferred.

**5. THE GLOBAL FEATURES ARE HAND-DESIGNED TEMPLATES**, not learned — `DIE-VICTIM-GPE` and
similar. They encode an ontology's regularities, so they must be rewritten per ontology.
Across the 125-type vocabulary this project trains on, that is a substantial hand-authored
surface, and it is the opposite direction from schema-driven.

**6. THE COMPARISON IS 2020.** BERT-base/large, English/Chinese/Spanish, sentence inputs —
against mmBERT with an 8192 window and broad language support. Its *numbers* are not a
target for us; its *representation* is the transferable part, and only partly.

**Net:** OneIE is the right thing to have read and the wrong thing to copy wholesale. What
transfers is the insight that **trigger instances should be individuated by span rather
than by type** — which `event_records: true` achieves within the record head, without a
closed tag set and without giving up the document. What does not transfer is the tagger,
the sentence scope, or the hand-written feature templates.

## 5. What follows, in order

1. **Warm the record head on events before switching the path.** The Tier 2 arms changed
   the decode path while the head was naive; warming first separates the two.
   **BUILT 2026-09-15:** `config/base/eb16-eventrecords-tr.yaml` does
   this at COLD START rather than as a fine-tune, so the head is warmed on events from
   step 0 and the Tier 2 confound cannot arise.

   It is **eb16-rebuild-tr's data** -- trilingual, repaired, label-unified -- patched with
   `event_records: true`, because eb16 is the mix that is actually wanted and its events
   were still going through the mention path. The diff against eb16-rebuild-tr, comments
   excluded, is SIX lines: the flag, a new output_dir and experiment_name, and three added
   corpora (professorbob_re, scierc, paraloq_json). **It is therefore NOT a one-variable
   A/B** -- a difference against eb16-rebuild-tr is attributable to the flag or to the
   corpora, and nothing in the result separates them. Dropping the three additions restores
   the single-key diff if that comparison is what is wanted.

   Read against `whr778/gliner2-eb16-rebuild-tr`, whose own `event_argument` figure is the
   0.1178 above. 167,752 train documents x 5 epochs = 838,760 samples, 52,423 optimizer
   steps at effective batch 16 on one GPU. Aggregate leakage gate CLEAN (167,752 / 21,138 /
   19,874 unique documents, all three intersections zero).

   The flag was verified to change the compile before the config was written, through
   `collate_fn_train` on a real CMNEE document carrying Experiment x2 / Accident x1 /
   Injure x2: **0 record specs with it off, 3 with it on** (one per type, natural mode,
   5 fields each). Pinned by a test.
   **LAUNCHED 2026-09-15** on an A100, ~14h, ~$28, `tools/lambda/event_base_run.sh`.

   **THE COST QUESTION IS SETTLED AND THE ANSWER IS CHEAP.** Measured before booking, 120
   steps per arm, same config with the flag flipped:

       event_records=false   18.4 samples/s
       event_records=true    16.7 samples/s     -> the flag costs 9%

   NOT the 5x the record-head defect implied. That defect is real and unresolved -- 4.6
   samples/s against the curve's 22, "~5x slower to train", 2026-08-10 -- but it is NOT
   triggered by `event_records` on this mixture. Across 838,760 samples that is ~$28 on an
   A100 rather than ~$167. **This does not clear the 4.6 figure**; it was measured on a
   different mixture and nothing here reproduces or explains it.

   The original warning is kept below because it is what justified measuring first:

   **AND IT MAY BE EXPENSIVE, for a reason already on file.** `event_records: true` moves
   ~100k event records ONTO the record head -- the head this project measured at
   **4.6 samples/s against the curve's 22** on the same H100, "~5x slower to train than the
   rest", unresolved since 2026-08-10 with the idle-GPU / pegged-core signature of
   Python-side work. Across 838,760 samples that is the difference between ~$35 and ~$167
   for one base. A throughput smoke (`tools/lambda/throughput_smoke.sh`) measures the delta
   on identical data with the flag flipped, and should be read BEFORE the base is booked.
   The fix proposed here is not free, and the thing that makes it costly is the same head
   the fix depends on.

2. **Re-run Tier 2 on CMNEE, not CASIE.** CMNEE dominates the affected mass here (1,606 of
   2,054 affected documents). CASIE was the previous venue and it is both smaller and
   harder.
3. **Report relaxed beside strict, always.** A metric that moves 0.118 → 0.578 on one key
   change has been measuring binding while being read as extraction. Any future claim about
   "argument quality" must say which it means.
4. **Re-read the argument curves with this in mind.** The head-init data-scaling curve
   (10k/40k/100k → 0.050/0.115/0.158) was run on **RAMS**, which is 0.0% affected. It
   therefore says nothing about the pooling ceiling, in either direction.

## 5b. A pre-registered per-role prediction, and independent support for the hypothesis

Written 2026-09-15 **while `eb16-eventrecords-tr` was training and before any result was
seen**, so the run can falsify this rather than be read to fit it.

**THE ARGUMENT METRIC IS ONE CORPUS.** Every high-support argument role in the blind test
is 100% CMNEE:

| role | gold instances (test) | dominant corpus |
|---|--:|---|
| Subject | 6,879 | cmnee 100% |
| Equipment | 4,517 | cmnee 100% |
| Date | 3,522 | cmnee 100% |
| Location | 2,354 | cmnee 100% |
| Militaryforce | 1,862 | cmnee 100% |
| Object | 713 | cmnee 100% |
| Materials | 647 | cmnee 100% |

25 of 42 scored roles have strict F1 of exactly 0.0000, but they carry only **1.8%** of the
argument mass. **The long tail is not the problem — the high-support roles are.**

**WITHIN CMNEE, 67.3% of argument instances belong to events that share their type with
another event in the same document** (14,004 of 20,813), and the share varies by role. That
variation is a natural experiment, because the roles come from one corpus, one annotator
and one language — so corpus, domain and language are controlled:

| role | affected share | current strict F1 |
|---|--:|--:|
| Equipment | 81% | 0.041 |
| Date | 71% | 0.124 |
| Location | 68% | 0.128 |
| Subject | 67% | 0.153 |
| Militaryforce | 51% | 0.126 |
| Materials | 41% | 0.177 |
| Object | 40% | **0.328** |

**Pearson r = −0.789 (n = 7, t = −2.87, df = 5, two-tailed p = 0.035)** — the more a role
is hit by same-type pooling, the worse it scores.

**AND THAT NUMBER DOES NOT SURVIVE THE OBVIOUS CONTROL. Stated here rather than left for a
reader to find:**

| | r | p |
|---|--:|--:|
| affected share vs F1 | −0.789 | 0.035 |
| **support** vs F1 | −0.506 | — |
| **affected share vs support** | **+0.750** | — |
| affected share vs F1, **controlling for support** | **−0.718** | **0.108** |

The two predictors are entangled at +0.750: the roles most hit by pooling are also the
highest-support roles. Control for support and the effect shrinks and **loses
significance**. With seven points this design simply cannot separate *"pooling hurts"* from
*"high-support roles score worse"*, and the raw −0.789 should not be quoted on its own.

A jackknife shows the raw correlation is at least stable — dropping any single role leaves
r between −0.68 and −0.86 — but a robust confounded correlation is still confounded.

**So this is a motivating observation, not evidence.** The prediction below is worth making
because it is cheap and falsifiable, not because the correlation establishes anything.

### The prediction

If the binding hypothesis is right, `eb16-eventrecords-tr` should improve these roles **in
rough proportion to their affected share**: Equipment and Date most, Object and Materials
least. Object is the control that makes this falsifiable — at 40% affected and already
0.328, it has the least to gain.

**What would falsify it:** uniform improvement across roles regardless of affected share
(says something else changed — more data, longer training, the three added corpora), or
Object improving as much as Equipment (says the gain is not about pooling).

### The honest limits

- **THE HEADLINE CORRELATION IS CONFOUNDED WITH SUPPORT** (r = +0.750 between the two
  predictors; partial r = −0.718 at p = 0.108). This is the limit that matters and it is
  stated in the section above rather than only here.
- **n = 7 roles from ONE corpus.** With seven points a single role moves things materially,
  though the jackknife range (−0.68 to −0.86) says the raw correlation is at least stable.
- **Correlation is not the mechanism.** An intrinsically harder role could simply appear
  more often in multi-event documents. The within-corpus design controls for corpus,
  annotator and language but not for role difficulty.
- **This run is not a clean test of it** — `eb16-eventrecords-tr` differs from
  `eb16-rebuild-tr` by the flag AND three added corpora, so a per-role pattern is evidence
  rather than proof.

## 6. Caveats, stated because the number is quotable

- **64.2% is this mixture's number**, dominated by one Chinese corpus. It is not a
  universal property of event extraction, and a differently-composed test would give a
  different ceiling.
- **The ceiling arithmetic is consistency, not a proof.** Confirming it means running the
  decoder against gold with the cap lifted and seeing strict argument F1 move toward
  relaxed — that experiment has not been run.
- **This diagnosis does not explain the 9,059 never-proposed arguments** (48.8%). Pooling
  explains why correct arguments bind wrongly; it does not by itself explain absence.
  Recall at 0.492 *relaxed* means half the arguments are missing before binding is even
  considered. That is a second, independent problem and this document does not solve it.
- **The record head is entangled with work in flight.** The cardinality A/B trains the same
  head. If warming it on events is the real lever, the two are not independent and should
  be sequenced deliberately.
