# Line 3 — full event support, and what could sink it

**The runs on this line are rows in [[EXPERIMENT_CATALOG]]** (research line: Events).

Status: **stated 2026-09-15, first base training.** This is the research line, its named
incumbent, and its risks written down BEFORE the numbers arrive so they can be checked
against rather than reconstructed afterwards. Companion to
[[EVENT_ARGUMENT_DIAGNOSIS]] (the measurement that motivates it) and
[[RESEARCH_PROGRAM]] (the programme this sits inside).

---

## 1. Three lines, and which one this is

| line | what it was | status |
|---|---|---|
| 1 | **span architecture** — `max_width`, count-first decode, the 19-instance cap | closed; substrate for PAPER_0 |
| 2 | **joint boundary, STRUCTURE space** — events expressed through the record/structure machinery *at decode time*, but supervised through the mention path | where `eb16-rebuild-tr` sits |
| 3 | **joint boundary, EVENT space** — events supervised on the RECORD head itself | **this line** |

The distinction between 2 and 3 is one config key that nobody set. `event_records` appears
**zero times** in every file under `config/base/`, so no base this project has trained has a
record head that has seen an event. It was believed set at the August port — and reasonably,
because `enable_records: true` carries the comment *"events decode as trigger + role
edges"*, which is true of **decode time** and false of **training**.

## 2. The incumbent, named deliberately

**`whr778/gliner2-eb16-rebuild-tr`** is the comparison, and nothing else is.

- **Not the span line.** Different research line, different substrate; a comparison across
  it measures the architecture change, not the event change.
- **Not the fastino models.** `-v1` are span. The current boundary ones support events only
  *through the structure space* — which is line 2, not line 3 — and their training data is
  not stated precisely enough to attribute a difference to anything.
- **eb16-rebuild-tr is the honest control**: same architecture, same data lineage, same
  label space, differing in whether events reach the record head. Line 2 against line 3.

## 3. The starting line

Measured on eb16-rebuild-tr's own 18,786-record blind test, greedy, threshold 0.5:

> **SUPERSEDED 2026-09-15.** These figures are of uncertain provenance and are NOT at threshold 0.5 — the incumbent's model card says `Decision threshold: 0.3`. Measured at the validation-selected 0.2: event_argument strict 0.0991 / relaxed 0.5884, event_trigger 0.5984, event_type 0.8650. See EVENT_ARGUMENT_DIAGNOSIS.md §4c.
    event_argument   strict 0.1178    relaxed 0.5783    fair 0.5737
    event_trigger    strict 0.6051
    event_type       strict 0.7545
    event            strict 0.4081

**The model finds arguments and cannot bind them.** Dropping the trigger link from the key
multiplies the score by five; boundary and label errors together are 6.6% of gold.

## 4. Risk register

### 4a. The EKF may not follow, and the line assumes it will

[[RESEARCH_PROGRAM]] §3 records, pre-registered and measured: **on real news the tracker
loses to a trivial baseline** (`est_last_value` 0.208 against the EKF's 0.136 on
Türkiye–Syria), and **"attribution, not filtering and not extraction, is the bottleneck."**

The line's bet is that binding-an-argument-to-its-event and attributing-a-figure-to-its-
stream are the same operation at two scales, so fixing the first moves the second. **That is
a hypothesis, not a finding.** If cross-document attribution fails for reasons independent
of within-document binding — coreference across articles, temporal ordering, source
disagreement — then a perfect event model still leaves the EKF where it is.

**Cheapest test of the bet:** once the base lands, re-run the EKF front-end on the three
real events and see whether binding precision moves *at all*. If it does not, the
cross-document half needs its own diagnosis before more event training is bought.

### 4b. Fixing binding cannot take `event_argument` past ~0.58

> **SUPERSEDED 2026-09-15.** These figures are of uncertain provenance and are NOT at threshold 0.5 — the incumbent's model card says `Decision threshold: 0.3`. Measured at the validation-selected 0.2: event_argument strict 0.0991 / relaxed 0.5884, event_trigger 0.5984, event_type 0.8650. See EVENT_ARGUMENT_DIAGNOSIS.md §4c.
Relaxed is 0.5783 and relaxed IS the no-binding-required ceiling. Perfect binding converges
strict onto relaxed and no further. **The other half of the loss is recall: 9,059 of 18,557
gold arguments (48.8%) are never proposed at all.**

So this line has two targets, not one, and only the first is being worked. Do not let a
strict number approaching 0.5 read as "solved" — it means binding is fixed and the recall
problem is now the whole problem.

### 4c. `event_argument` is effectively ONE CORPUS

Every high-support argument role is 100% CMNEE — Subject 6,879, Equipment 4,517, Date
3,522, Location 2,354, Militaryforce 1,862. 25 of 42 scored roles have strict F1 of exactly
0.0000 and carry 1.8% of the mass between them.

**"Improving event arguments", measured this way, largely means improving one Chinese
military corpus.** Generalisation to RAMS, WikiEvents or real news is unproven and cannot be
read off this metric. A second event corpus in the blind test would be the fix.

### 4d. The current run changes two things

`eb16-eventrecords-tr` differs from the incumbent by `event_records: true` **and** by three
added corpora (professorbob_re, scierc, paraloq_json). A headline difference is attributable
to either.

**Mitigating fact, verified:** the three added corpora contribute **zero events and zero
arguments**, so the `event_argument` denominator is unchanged and that head's comparison is
sound. **Entity, relation and structure are NOT comparable** — the blind test grows from
18,786 to 19,874 documents, and this programme has already retracted findings for
cross-test-set comparison.

### 4e. Tier 2 failed twice, and a third failure needs to be distinguishable

CASIE Tier 2 scored 0.0036 against a 0.2998 control; MAVEN gave trigger −0.008. Both flipped
the path on a naive record head, so they measured a path change plus a head-init failure.
This run warms the head from step 0, which is why it is not the same experiment. **But if it
also fails, the two explanations — "cold start does not help either" and "something else is
wrong" — must be told apart**, and the per-role pattern (§4f) is the instrument for that.

### 4f. The pre-registered per-role prediction is weaker than it first looked

Within CMNEE the same-type-pooling share varies by role, and it correlates with F1 at
r = −0.789. **But affected share and support are entangled at r = +0.750**, and controlling
for support drops the partial correlation to −0.718 at p = 0.108 — not significant. With
n = 7 this cannot separate *"pooling hurts"* from *"high-support roles score worse"*.

Prediction retained because it is cheap and falsifiable: Equipment (81% affected, F1 0.041)
and Date (71%, 0.124) should move most, **Object** (40%, 0.328) least. Uniform movement
regardless of share says something other than pooling changed.

### 4g. Checked and NOT a risk

- **A new instance cap replacing the old one.** The record head allocates
  `record_instance_queries` per group; the default is **32** and CMNEE's worst document
  holds **16** same-type events. 100% of documents fit. The cap of 1 is lifted without
  introducing a cap of 8.
- **The 4096 training window.** Only **0.08%** of event documents exceed it (median 276
  tokens, p95 1,068), so windowing is not truncating event structure. **But the corollary
  is that the model's 8192 capacity is UNTESTED by this data** — long-context event
  extraction is a data gap, not a config one, and remains unproven.
- **Throughput.** `event_records` costs **9%** (18.4 → 16.7 samples/s), not the 5× the
  unresolved record-head defect implied.

### 4h. Our headline metric is not the field's, and the gap was overstated

> **AMENDED 2026-09-22 — read this first.** §4h was written from the PAPER. The SOURCE was
> read on 2026-09-22 ([[EVENT_ARGUMENT_DIAGNOSIS]] §4c-i) and it changes the conclusion, not
> just the detail. **OneIE's scoring is looser than ours; its TRAINING is tighter.** It
> represents an argument as an edge to a specific trigger node
> (`role = (trigger_idx, entity_idx, label_idx)`) **and optimises that edge** with its own
> cross-entropy over (trigger, entity) candidate pairs. Then its scorer discards the
> binding. We are the mirror image: our strict metric REQUIRES the trigger, and **nothing
> in our loss produces it** — `candidate_pair_loss` pairs a span's START with its END, not
> a trigger with an argument. **CORRECTED the same day:** that last clause was wrong.
> `record_loss_weight` defaults to 1.0 and `compute_group_loss` seeds an instance FROM THE
> ANCHOR, supervising fields into it -- so we DO train the binding, by anchor-seeded
> assignment rather than pairwise edge classification. What differs is the FORM (and that
> OneIE's pair set is negative-saturated by construction). Crucially, the 0.1178/0.5783
> spread was measured WITHOUT `event_records`, on a record head that had never seen an
> event -- so it is not evidence about the architecture eb17 trains. See
> [[EVENT_ARGUMENT_DIAGNOSIS]] §4c-i CORRECTION.

Read from the OneIE paper 2026-09-15 ([[EVENT_ARGUMENT_DIAGNOSIS]] §4c). OneIE's Arg-C
requires **offsets + event type + role** and does **not** require the specific trigger to
match. Our `strict` adds trigger identity; our `relaxed` drops exact spans for overlap. So:

> **SUPERSEDED 2026-09-15.** These figures are of uncertain provenance and are NOT at threshold 0.5 — the incumbent's model card says `Decision threshold: 0.3`. Measured at the validation-selected 0.2: event_argument strict 0.0991 / relaxed 0.5884, event_trigger 0.5984, event_type 0.8650. See EVENT_ARGUMENT_DIAGNOSIS.md §4c.
    our strict   0.1178   LOWER bound on an OneIE-comparable number (adds trigger)
    OneIE Arg-C     ?     not currently computed
    our relaxed  0.5783   UPPER bound (accepts overlapping spans)

**Consequences for this line:**

> **SUPERSEDED 2026-09-15.** These figures are of uncertain provenance and are NOT at threshold 0.5 — the incumbent's model card says `Decision threshold: 0.3`. Measured at the validation-selected 0.2: event_argument strict 0.0991 / relaxed 0.5884, event_trigger 0.5984, event_type 0.8650. See EVENT_ARGUMENT_DIAGNOSIS.md §4c.
- **0.1178 must not be quoted against published work.** The "catastrophic" framing
  overstated the gap against the field; the field's own criterion puts this model nearer
  0.58 than 0.12.
- **Strict remains the right metric FOR US.** The EKF needs a figure bound to the right
  event *instance*, not merely the right event type. So this line optimises something the
  literature does not measure — which is defensible, and must be stated rather than
  presented as beating a benchmark.
- **ACTION, no GPU:** add an Arg-C metric (exact surface, type + role, no trigger) so this
  line is comparable to the event-extraction literature at all.

**And OneIE is an ALTERNATIVE ANSWER to §1's problem, not just a benchmark** — but a
costly one, and the costs are enumerated in [[EVENT_ARGUMENT_DIAGNOSIS]] §4c-ii. It
identifies triggers with token-level BIO + CRF, so one node per trigger SPAN and
multi-instance falls out with no cap to lift.

**The separable part of it is the LOSS, not the tagger.** `event_records: true` already
gives us multi-instance, so the pooling half is answered. What remains unanswered is the
edge: OneIE trains argument→trigger and we do not. That half can be adopted WITHOUT the BIO
tagger and without forfeiting schema-driven labels, because a pairwise role objective over
(anchor, argument-span) candidates is indifferent to where the candidates came from. See
TODO item 16.

**It is not, however, a drop-in fallback.** BIO tagging scores *"a tag in a target tag
set"* fixed at training, and **GLiNER2's premise is that labels are an INPUT at inference** —
adopting that tagger would forfeit schema-driven extraction, which is the thing that makes
this model worth having. It is also sentence-level where our argument mass is
document-level, it requires entity gold our corpora do not all carry, its global features
are hand-authored per ontology, and *"multiple events per trigger"* remains a named residual
error in its own analysis.

**What transfers is the insight, not the machinery:** trigger instances should be
individuated **by span, not by type**. `event_records: true` achieves exactly that inside
the record head — without a closed tag set, and without giving up the document.

## 4h. What the negatives work changed about the risk register

The register above was written when `event_argument` was the line's single point of failure.
Measured since (LABEL_NEGATIVES_PLAN §5c-§5e):

- **A NEW RISK, now retired: the model could not say no.** Given a schema of only ABSENT event
  types the incumbent fired on **63 of 100 documents**, and its real full-menu `event_type`
  precision was **0.5521** against the 1.0000 the blind test reports. Nothing in the standard
  eval could express this, because training and eval shared the same gold-derived menu. Label
  negatives cut invented types **363 → 274** and lifted rejection precision **0.5310 → 0.5977**.
- **The argument ceiling is unchanged.** Negatives improve BINDING (precision) and not
  EXTRACTION: relaxed argument recall 0.42 means a third of arguments are never proposed, and
  no typing or rejection change recovers them. The §4i comparison stands.
- **`event_type` must be quoted as RECALL.** Its precision is an identity of the gold menu, and
  `F1 = 2R/(1+R)` in 12 of 12 readings on file.

## 4i. Three approaches, side by side

| | **OneIE** (2020) | **JB / structures** (line 2) | **JB / events** (line 3) |
|---|---|---|---|
| **an event instance is individuated by** | trigger **span** (BIO+CRF per token) | event **TYPE** per document | trigger **span** — the record anchor is `role_index 0`, always the trigger |
| **two `Attack` events in one document** | two nodes, natively | **pooled into ONE** | two record instances |
| **instance capacity** | unbounded (one node per tagged span) | **1 per type** | `record_instance_queries` = **32**; worst observed doc = 16 |
| **label vocabulary** | **CLOSED** — scores "a tag in a target tag set" fixed at training | **schema input** at inference | **schema input** at inference |
| **scope** | **one sentence** | document (8192 capable, 4096 window) | document (8192 capable, 4096 window) |
| **argument → trigger binding** | graph edge, trigger node → entity node | attaches to the pooled trigger | role edge to the anchoring trigger |
| **needs entity gold** | **yes** — arguments are edges to entity mention nodes | no | no |
| **structural constraints** | hand-authored global feature templates per ontology | none at training | cardinality / exclusivity in the record head |
| **what its argument metric requires** | offsets + type + role, **no trigger** (Arg-C) | — | — |
| **evidence** | **ACE05-E** Trig-C 74.7, Arg-I 59.2, **Arg-C 56.8**; **ACE05-E+** Trig-C 72.8, **Arg-C 54.8**; **ERE-EN** Trig-C 57.0, **Arg-C 46.5** (single model, F1 %) | **measured** on the 18,786-doc blind test: argument **11.78 strict / 57.83 relaxed**, trigger **60.51**, type **75.45** | **none yet** — training 2026-09-15 |
| **principal issue** | closed tag set is incompatible with schema-driven extraction; sentence-only; *"multiple events per trigger"* is a named residual | **64.2% of gold event instances are structurally inexpressible** | unproven; the record head has an unresolved throughput defect and two failed Tier 2 precedents |

### DO NOT READ THE EVIDENCE ROW AS A COMPARISON

The numbers sit in the same range and that is a coincidence of scale, not a result:

| | OneIE Arg-C | ours |
|---|--:|--:|
| ACE05-E | 56.8 | — |
| ACE05-E+ | 54.8 | — |
| ERE-EN | 46.5 | — |
| our blind test, `relaxed` | — | **57.83** |
| our blind test, `strict` | — | 11.78 |

**Three reasons this is not a like-for-like comparison, any one of which is disqualifying:**

1. **Different data.** ACE05-E and ERE-EN are English sentence-level newswire with a curated
   ontology. Ours is a 13-corpus multilingual document-level mixture whose argument mass is
   ~100% CMNEE, a Chinese military corpus.
2. **Different span rule.** OneIE requires **exact offsets**; our `relaxed` accepts
   **overlap** (`New York City` ↔ `New York`). Ours is the more permissive of the two, so
   57.83 is an upper bound on anything OneIE-comparable.
3. **We do not compute their metric.** Arg-C is offsets + type + role with no trigger; our
   `strict` adds the trigger and our `relaxed` loosens the span. The comparable figure lies
   between 11.78 and 57.83 and is **not measured** (`TODO.md`).

OneIE's authors make a related point about their own table — *"we hold the opinion that the
single-model scores in Table 3 better reflect the actual performance of ONEIE and should be
used for future comparison"* — so the single-model column is the one quoted here, not the
four-model ensemble (which reaches Arg-C 58.6).

**What the row is for** is calibration of expectation, not scoring: an Arg-C in the
**mid-40s to high-50s is what a strong 2020 system achieves on curated English data**. It
says a perfect system is not at 90, and that the headroom above our 57.83 relaxed is smaller
than the headroom implied by treating 11.78 as the starting point.

### What the table is actually saying

**Lines 2 and 3 differ on exactly one row that matters**, and it cascades: *individuated by
type* against *individuated by span*. Everything downstream — pooling, the 64.2%, the
strict/relaxed gap — follows from that single choice.

**Line 3 and OneIE agree on that row and disagree on almost every other.** Both individuate
by trigger span; OneIE pays for it with a closed tag set and a sentence boundary, line 3
pays for it with an unproven head and a capacity cap. **The insight transfers; the
machinery does not.**

**The capacity cap is a real difference from OneIE and is currently not binding.** 32
instance queries against a worst observed document of 16 same-type events. It would bind on
a denser corpus, and that is a property to re-check per mixture rather than assume.

**Line 2 is not a bad design; it is a design for a different task.** One instance per type
is correct when documents describe one event of each kind — RAMS is **0.0%** affected,
being 100% single-event documents — which is exactly why the RAMS-based argument curves
never surfaced this. It fails on multi-event corpora, and our argument mass is multi-event.

## 5. What "proves out" means

The line continues if, on an identical `event_argument` denominator:

1. **strict rises materially toward relaxed while relaxed HOLDS** — the binding signature;
> **SUPERSEDED 2026-09-15.** These figures are of uncertain provenance and are NOT at threshold 0.5 — the incumbent's model card says `Decision threshold: 0.3`. Measured at the validation-selected 0.2: event_argument strict 0.0991 / relaxed 0.5884, event_trigger 0.5984, event_type 0.8650. See EVENT_ARGUMENT_DIAGNOSIS.md §4c.
2. **the guard heads do not collapse** — `event_type` 0.7545 and `event_trigger` 0.6051 are
   the things a record-head switch could plausibly damage;
3. **the per-role pattern is not uniform** — §4f.

It does not continue on a strict rise alone if relaxed rose with it: that is more data or
longer training, not better binding.

**And success is now bounded on both sides.** Binding can converge strict onto relaxed and
no further (§4b), while relaxed itself is only ~0.58 at the untuned threshold 0.5 — which
the running sweep may move. A strict score near 0.5 would mean binding is essentially
solved and the recall ceiling is the entire remaining problem; it would NOT mean events are
solved.

## 6. Deliberately deferred

Backfill is accepted as a cost of moving forward, and is listed so it is a decision rather
than an omission: the single-variable `event_records` run (drop the three corpora, ~$28), a
second event corpus in the blind test (§4c), and the argument-recall problem (§4b). None
blocks the next step; all block a defensible claim.

## The trigger is the anchor, and the argument is keyed by the trigger (2026-09-29)

Measured on the eb17 blind test, control vs `record_anchor_threshold` 0.1, 26,724 records.

**The extra triggers are not badly-bounded duplicates -- they are 91.5% pure false
positives**, on spans carrying no gold trigger at all. The Ortmann error accounting
reconciles exactly:

| category | control | anchor 0.1 | delta |
|---|---|---|---|
| COR | 7,624 | 9,110 | +1,486 |
| FN | 10,708 | 8,415 | -2,293 |
| **FP** | 3,265 | 11,923 | **+8,658** |
| BES+BEO+BEL+LBE+LE | 552 | 1,359 | +807 |

2,293 recovered false negatives = 1,486 now exactly correct + 807 boundary/label errors,
to the unit. Separately, 8,658 brand-new FPs appear. **3.78 spurious triggers per gold
trigger recovered** -- that is the bar any separator has to clear.

**A seed/claim split at decode CANNOT work.** `_pred_event_argument_set`
(`eval_metrics.py:944`) keys every argument as `(etype, role, entity, trigger_key)` and
does `if not trigger_key: continue` -- an event with no emitted trigger contributes NO
arguments. So suppressing the trigger claim while keeping the instance seeded would delete
exactly the arguments the low threshold was bought for. The argument tuple also CONTAINS
the trigger text, so an argument matches gold only when its trigger does.

That is the whole shape of the +0.0579 result: argument recall rides on the 1,486
newly-correct triggers, and argument precision falls 0.4246 -> 0.2418 because each of the
8,658 spurious triggers carries spurious arguments with it.

**So the lever is INSTANCE REJECTION, not a second threshold.** Dropping a whole spurious
instance (trigger and its arguments) would lift trigger precision AND argument precision at
fixed recall. `object_logits` is exactly that per-instance existence score -- and in natural
mode it receives NO gradient (`compute_group_loss` returns `object_loss = zero`,
records.py:1453; the Hungarian branch that trains it is anchorless-only). Whether that head
carries usable signal once trained is unmeasured, and is the head-init question again.

### The gate we tune is the object head, and it is untrained but NOT uninformative

`record_anchor_threshold` does NOT threshold the anchor span score. `decode_group`
(records.py:731-736) computes `obj_prob = sigmoid(object_logits / temperature)` and selects
instances with `obj_prob >= anchor_threshold` in natural mode. So the existence head IS the
gate this programme has been tuning -- and it is the same head that receives
`object_loss = zero` for every event in training.

**Measured 2026-09-29** (`tools/train/probe_object_score_separation.py`, eb17-best, val
event corpora, threshold 0.1), labelling a decoded instance GOOD when its `anchor_span`
matches a gold anchor span resolved exactly as `compute_group_loss` resolves it:

| sample | groups | good | spurious | mean good | mean spurious | **AUC** |
|---|---|---|---|---|---|---|
| smoke | 19 | 51 | 25 | 0.7040 | 0.2955 | **0.8196** |
| 23x larger | 446 | 898 | 860 | 0.6925 | 0.3045 | **0.8272** |

**AUC 0.83 with zero gradient.** The head inherits usable signal from the shared
representation, so training `object_loss` in natural mode starts from something rather than
from noise -- this is NOT the head-init situation that has bitten this programme before.

**UNRECONCILED, AND IT BOUNDS THE CLAIM.** The spurious:good ratio here is 0.96:1; the
blind test's extra-trigger accounting gives 3.78:1. Candidate explanations, none verified:
this probe counts decoded INSTANCES while the eval counts trigger SPANS after formatting
and dedup (one instance can emit several); this probe labels a match by anchor SPAN within
the group while the eval requires event TYPE and trigger TEXT; and the populations differ
(event corpora val here, the full 26,724-record test split there). Until that is resolved,
AUC 0.83 is evidence the head ranks, NOT a prediction of how much precision training buys.

### The ratio gap is ACCOUNTING, not population -- so the AUC is measured on an easier task

Ran the EVAL's own Ortmann accounting on the PROBE's own data (400 val event records,
control vs anchor 0.1, CPU, both arms chunked identically so the ratio is unaffected):

| category | control | anchor 0.1 | delta |
|---|---|---|---|
| COR | 190 | 239 | +49 |
| FN | 284 | 210 | -74 |
| FP | 125 | 455 | **+330** |
| BES+BEO+BEL+LBE+LE | 26 | 51 | +25 |

74 recovered = 49 exactly correct + 25 boundary/label errors, to the unit, exactly as on
the blind test.

| view | ratio spurious : recovered gold |
|---|---|
| probe, decoded INSTANCES matched by anchor span | 0.96 : 1 |
| **eval accounting, SAME data** | **4.46 : 1** |
| eval accounting, full blind test | 3.78 : 1 |

**VERDICT: accounting.** The population is not the difference -- the same records score
4.46:1 under the eval and 0.96:1 under the probe. The probe counts decoded INSTANCES and
calls one good when its anchor SPAN matches a gold anchor in that group; the eval counts
trigger SPANS after formatting and dedup and requires the event TYPE and trigger TEXT to
match. The probe's labelling is roughly 4.6x more forgiving.

**CONSEQUENCE, and it downgrades the earlier reading.** AUC 0.8272 says the object head
ranks instances well *under the probe's loose labelling*. It is NOT an estimate of the
separation the metric actually poses, where the spurious class is ~4.6x larger per
recovered gold. The ranking signal is real; its size against the real task is UNMEASURED.

**BEFORE FUNDING A TRAINING RUN**, re-measure the AUC with eval-equivalent labelling:
score emitted trigger spans, require type + text match, after dedup. If the AUC survives
near 0.83 the instance-rejection experiment is well-founded; if it falls toward chance the
head is not separating what the metric counts and the experiment is not yet justified.

### CORRECTION: the 4.46 vs 0.96 comparison was a marginal ratio against an absolute one

The section above compared the eval's **DELTA** ratio (new FP per recovered FN, control ->
0.1) with the probe's **ABSOLUTE** ratio (spurious instances / good instances at 0.1).
Those are not the same quantity and the "4.6x more forgiving" conclusion drawn from it is
withdrawn. Measured properly on the SAME 400 val event records at anchor 0.1:

| quantity | value |
|---|---|
| eval trigger predictions (COR 239 + FP 455 + partials 51) | 745 |
| probe decoded instances | 759 |
| **predictions per instance** | **0.98** |
| eval absolute FP : COR | 1.90 : 1 |
| probe absolute spurious : good | 0.96 : 1 |
| probe AUC on these records | 0.8351 |

**Three causes, correctly apportioned.** POPULATION: not a cause, these are the same
records. GRANULARITY: not a cause either -- 745 predictions from 759 instances is ~1:1, so
one decoded instance yields about one trigger prediction. STRICTNESS: real, and about
**2x** -- the probe calls 388 instances good where the eval counts 239 correct, and the
eval finds 455 FP where the probe finds 371 spurious. The headline 4.6x was mostly an
artifact of the mismatched comparison.

**WHAT THIS CHANGES FOR THE DECISION.** The ~1:1 granularity is the useful finding: the
unit the intervention operates on (reject a whole instance) maps almost exactly onto the
unit the metric counts (a trigger prediction), so an instance-level gate translates
directly into trigger precision. AUC 0.8351 is measured with a positive class ~1.6x more
generous than the eval's COR, so it overstates -- but by around 2x in class balance, not
by 4.6x, and the ranking signal at 0.835 has margin above chance to absorb that.

**STILL WORTH DOING BEFORE A TRAINING RUN**, now a small change rather than a rebuild:
label each instance by whether the eval counts its trigger as COR, and re-run the AUC.
That removes the last estimate from the chain.

### Eval-equivalent labelling: dedup is a no-op, and the residual is NOT explained

Added the metric's own unit to the probe -- trigger predictions deduped by anchor span,
strongest score kept. On 400 val event records at anchor 0.1:

| view | good | spurious | ratio | AUC |
|---|---|---|---|---|
| instances | 382 | 374 | 0.98 : 1 | 0.8426 |
| **deduped (eval's unit)** | **382** | **374** | **0.98 : 1** | **0.8426** |

**Identical.** No two instances in a group decode the same anchor span, so dedup was never
the difference. Two hypotheses are now eliminated:

- **dedup** -- measured, a no-op;
- **the emitted trigger differs from the anchor span** -- refuted in code. `decode_group`
  (records.py:816-819) puts `rec.anchor_span` into the trigger field with `rec.score` as
  its score, so for natural-mode records the emitted trigger IS the anchor span and the
  probe's unit IS the metric's unit.

**THE RESIDUAL IS UNEXPLAINED AND I AM NOT GUESSING AT IT.** The probe counts 382 good on
these records; the eval counts COR 239 and FP 455 (absolute 1.90:1 against the probe's
0.98:1). Remaining candidate, unverified: the two runs do not process the same physical
units -- the probe uses the TRAINING collator with `sliding_window`, the eval run used
`chunk_size=512` plus its own split hygiene, so "the same 400 records" may be different
numbers of scored windows with different menus.

**AUC IS STABLE ACROSS FOUR INDEPENDENT SAMPLES: 0.8196, 0.8272, 0.8351, 0.8426.** The
ranking signal is not in doubt. What is not established is the exact class balance of the
task a trained gate would face, and one estimate therefore remains in the chain.

## Emission does not scale with window size (2026-09-29)

Three arms over 300 `cc_news_long` documents (median 9,697 tokens, 100% multi-window),
same model, same docs, threshold 0.3, differing only in chunking.

| head, strict micro F1 | A: 4096 win, ov 0 | B: 4096 win, ov 64 | C: 200 words, ov 50 |
|---|---|---|---|
| entity | 0.0668 | 0.0686 | **0.1407** |
| event_type | 0.3663 | 0.4135 | **0.9427** |
| event_trigger | 0.0036 | 0.0024 | 0.0078 |
| structure | 0.0100 | 0.0100 | 0.0192 |
| event_argument | 0.0000 | 0.0000 | 0.0013 |
| relation | 0.0000 | 0.0000 | 0.0000 |

**A vs B -- OVERLAP IS A WASH.** Identical window, only overlap differs, and entity moves
0.0668 -> 0.0686. The provisional overlap 0 stands; boundary coverage is not what long
documents are losing.

**THE REAL FINDING: the model emits a roughly FIXED number of spans per window, whatever
the window holds.** Gold is 22.7 entities per 1,000 tokens. Arm A emits **2.4** per 1,000
tokens (10.6% of gold), arm C emits **12.4** (54.6%). Per window that is ~7.7 emitted
where ~73 exist at 4096 tokens, against ~2.4 where ~4.5 exist at ~260 tokens. Recall per
window is ~11% at the model's own window and ~55% at the small one.

| arm | predictions | correct | FN | recall | precision |
|---|---|---|---|---|---|
| A | 6,893 | 2,459 | 60,963 | **3.7%** | 0.353 |
| B | 7,187 | 2,534 | 60,784 | 3.8% | 0.349 |
| C | 35,930 | 7,232 | 47,326 | **11.0%** | 0.199 |

**DENSITY IS RULED OUT, and it inverts the obvious explanation.** cc_news_long is 22.1
entities per 1k tokens; `cc_news_haiku45` -- the SAME SOURCE, in training -- is **36.1**,
and assorted test corpora are 52.4. The eval corpus is SPARSER than what the model trained
on. The only variable left is DOCUMENT LENGTH: median 457 tokens in training against 9,511
here.

**CONSEQUENCE FOR PRODUCTION.** Recall degrades with document length because emission does
not grow with the window. On a 9,697-token document read at the model's own 4096 window,
eb17 finds under 4% of the entities; the same document read in 200-word windows finds 11%.
The EKF pipeline's 200-word band was not a tuning quirk -- it was compensating for this,
and the recorded "16/16 vs 15/16 at the library default 384 on Turkiye" is consistent.

**CONSEQUENCE FOR THE UNIFORM-WINDOW POLICY.** "One window everywhere at the model's
configured size" is the WORST setting measured for long-document recall. The policy is
right that training, eval, blind test and inference should agree; it is wrong that the
agreed value should be the training window, at least until emission scales.

**WHAT IS NOT ESTABLISHED.** Why emission is capped -- candidate budget, decode threshold
behaviour at length, or attention dilution -- is unmeasured, and C wins here on F1 while
LOSING precision (0.199 vs 0.353), so "smaller is better" is a recall trade, not a free
win. A/B-vs-C also varies window AND overlap together; only A-vs-B is clean.

### CORRECTION: that was an ENTITY result, and events want the OPPOSITE window

The section above concluded "window size is everything" and that the training window is
the wrong uniform value. That generalised an ENTITY measurement onto events the run never
had the signal to judge: `event_argument` came back 0.0000 / 0.0000 / 0.0013 across the
three arms -- noise. The run adjudicated entities only.

Measured on cc_news_long (300 docs, 3,392 events, 11.3/doc, 1.55 per 1k words), trigger to
argument distance in WORDS:

| | value |
|---|---|
| median | **446** |
| mean | 1,333 |
| p90 | 4,112 |
| max | 15,782 |

| window | argument links inside | **whole events fitting** |
|---|---|---|
| 200 words (arm C) | 42.0% | **30.5%** |
| 384 words (library default) | 48.4% | 36.6% |
| 2730 words (arms A/B) | 81.4% | **74.8%** |

**A 200-word window cannot see 69.5% of events.** The median argument sits 446 words from
its trigger, more than twice that window, so arm C is CUTTING event structure, not merely
trading precision for recall. Its nominal event edge (0.0013 vs 0.0000) is noise and must
not be read as a win.

**THE REAL SHAPE IS A TENSION, NOT A SINGLE ANSWER.** Entities are LOCAL and emission-bound,
so small windows help them (recall 3.7% -> 11.0%). Events are LONG-RANGE, so small windows
destroy them. No single window optimises both, which is a different objection to the
uniform-window policy than the one made above: not that the training window is too large,
but that one value cannot serve both tasks.

The EKF pipeline's 200-word band is well matched to ITS job -- casualty figures, which are
local and entity-like -- and would be a poor choice for long-range event structure.

**COVERAGE CAVEAT.** 2,074 of 7,472 arguments (27.8%) and 347 triggers do not appear
verbatim and are excluded, so the distances describe ~72% of argument links. If the missing
surfaces are systematically abstractive the distribution shifts, though not enough to bring
30.5% near 100%.

## The cap: candidate enumeration does not scale with the window (2026-09-29)

`candidate_budget` is 128 and eb17 leaves it there. The config calls it "an inference-width
decision" and sizes it from query-group statistics on corpora whose median document is 457
tokens. It is a FIXED enumeration per query, independent of how much text the window holds.

Measured with `ProposalStats.gold_hit_without_injection` -- the proposals BEFORE gold
injection, which is the inference-time question -- on cc_news_long:

| window (tokens) | gold | in candidates | coverage |
|---|---|---|---|
| 128 | 559 | 355 | 63.5% |
| 256 | 1,069 | 506 | 47.3% |
| 512 | 1,645 | 521 | 31.7% |
| 1024 | 2,104 | 446 | 21.2% |
| 2048 | 4,838 | 685 | 14.2% |
| **4096** | **11,797** | **853** | **7.2%** |

**Gold grows 21x while the candidate set captures a nearly FLAT 355-853.** That is a cap,
not a gradient. At the model's own 4096 window **92.8% of gold entities never enter the
candidate set**, so no threshold, no calibration and no decode change can recover them.

**THIS EXPLAINS THE WINDOW ARMS ENTIRELY.** Arm A's 3.7% entity recall sits under a 7.2%
ceiling; arm C's 11.0% sits under ~47%. Arm C did not win because small windows read better
-- it won because 16x more windows means 16x more independent 128-candidate budgets. The
"tension" between entity-wants-small and event-wants-large is an ARTEFACT of the cap, not a
property of the tasks.

**WHY THE REGULAR CORPORA LOOK FINE.** At 483 median tokens a window holds ~11 gold
entities against a budget of 128, so the cap never binds -- consistent with entity strict
F1 0.5736 on the blind test. The cap is a LONG-DOCUMENT failure and is invisible on every
corpus this programme normally measures.

**THE FIRST VERSION OF THIS PROBE WAS VACUOUS.** It counted positives in
`candidate_pair_loss` and reported 100% coverage at 256/512/1024, because
`gold_injection_prob` defaults to **1.0** -- training INJECTS gold into the candidate set,
so the answer was fixed before the model ran. Three consecutive 100% readings are what
prompted the check. `gold_hit_without_injection` already existed for exactly this and needs
`collect_diagnostics`.

**NEXT, AND NOT YET DONE.** Whether raising `candidate_budget` with window length recovers
the recall is unmeasured; so is its cost (enumeration is quadratic-ish in boundaries, and
the config notes 1024 costs 6.4x the enumeration of 160 for no gain on SHORT documents).
Nothing here touches events: `global_decode` was off in every run so far, so the event
numbers remain unmeasured rather than negative.

### The cap decomposed: half configuration, half the boundary scorer

Paired sweep, identical documents, `gold_hit_without_injection`:

| window | baseline (16/16, budget 128, alpha 0) | bumped (128/128, 2048, alpha 0.05) | aggressive (256/256, 8192, alpha 0.25) |
|---|---|---|---|
| 512 | 29.7% | 64.9% | **73.7%** |
| 1024 | 19.9% | 43.6% | 50.5% |
| 2048 | 14.8% | 30.4% | 32.8% |
| 4096 | **8.1%** | 19.7% | **21.5%** |

**Capacity is a real lever that SATURATES.** Baseline -> bumped roughly doubles coverage at
every width. Bumped -> aggressive (2x top-k, 4x budget, 5x alpha, 4x k_max) buys only
19.7% -> 21.5% at 4096. And coverage still falls 73.7% -> 21.5% across widths AT the
aggressive setting, so configuration explains about half the collapse and cannot close it.

**WHERE THE REST GOES.** `ProposalStats.start_hit`/`end_hit` separate boundary selection
from pairing. At the aggressive setting:

| window | pair coverage | start boundaries found | end boundaries found |
|---|---|---|---|
| 512 | 74.6% | **89.2%** | **84.6%** |
| 4096 | 22.1% | **50.3%** | **57.9%** |

Pair coverage tracks the PRODUCT of the two (0.892 x 0.846 = 75.5% vs 74.6% observed;
0.503 x 0.579 = 29.1% vs 22.1%), so `ends_per_start` costs ~1 point at 512 and ~7 at 4096
-- real but minor.

**THE DOMINANT FAILURE IS BOUNDARY RANKING, AND IT IS NOT CONFIGURATION.** At 4096 the
scorer ranks only 50.3% of gold starts into a top-k with room for ~1024 of ~4096 positions.
Capacity is 4x what is needed and half the gold boundaries still never make the list.

**LIKELY CAUSE, NOT YET PROVEN.** eb17 trains at `max_len: 4096` with `window_stride: 3072`,
but its corpora have a median document of **457 tokens** -- so the boundary scorer has
almost never seen 4096 tokens of real content, and its ranking at that scale is untrained
rather than broken. That is a TRAINING-DISTRIBUTION hypothesis and the way to test it is a
long-document training mix, not a config change.

**WHAT THIS SETTLES.** The earlier "entities want small windows, events want large" tension
is an artefact: arm C won by getting 16x more independent budgets. Raising the budget
recovers half of that for free at eval/inference time, and the remainder needs training.

### AMENDMENT: lifting the cap is NET HARMFUL at every threshold tried

Item 1 raised gold coverage at a 4096-token window from 8.1% to 18.7% and I presented
that as recovering half the long-document loss. It does not. Measured on 60 cc_news_long
docs, global_decode on, the model's own window, one variable per row:

| arm | entity P | entity R | **entity F1** | FP per COR |
|---|---|---|---|---|
| **S03 shipped / t 0.3** | **0.3551** | 0.0366 | **0.0664** | **0.77** |
| W03 wide / t 0.3 | 0.0113 | 0.0426 | 0.0179 | 70 |
| W05 wide / t 0.5 | 0.0099 | 0.0350 | 0.0155 | 79 |
| W07 wide / t 0.7 | 0.0090 | 0.0295 | 0.0138 | 86 |
| W09 wide / t 0.9 | 0.0090 | 0.0253 | 0.0133 | 88 |

**No WIDE arm beats the incumbent at ANY threshold.** The width buys +0.0060 entity recall
for -0.3438 precision, and it gets WORSE as the threshold rises: from t 0.3 to 0.7 the
threshold removed 14% of false positives while destroying 31% of correct ones, so FP per
COR climbs 70 -> 86. The extra candidates are scored ABOVE the marginal correct ones --
the confidence ordering is inverted on exactly the candidates the wider proposal admits.

**SO THE CAP WAS DOING REAL WORK.** The low threshold WAS compensating for the cap, but the
compensation is not recoverable: the discrimination needed to exploit the extra coverage
does not exist. Coverage is an upper bound on reachable recall and the model cannot
approach it. Item 1 is a coverage result with no metric behind it.

**DO NOT QUOTE `event_type` FROM THAT TABLE.** It reads 0.3851 -> 0.9982, which is a gold-
menu artefact: under a menu built from each document's own gold, `event_type` precision is
pinned at 1.0000 by construction, so F1 = 2R/(1+R) and emitting MORE instances raises it
mechanically with no precision penalty available. The wider proposal emits more instances.

**CONVERGES WITH THE SCORER FINDING.** Only 50.3% of gold starts rank into a top-k with 4x
the room needed; raising capacity just admits more candidates the scorer mis-ranks. No
configuration fixes that, which leaves the long-document training mix as the only lever --
and its hypothesis is now better supported, since the scorer has barely seen 4096 tokens of
real content (median training document 457 tokens).

### The anchor optimum is NOT a cap artefact

Two full anchor sweeps on the same validation split, shipped vs WIDE proposal width, no
global_decode (matching how the original pick was made). `event_argument_strict`:

| anchor | SHIPPED | WIDE | delta |
|---|---|---|---|
| 0.5 | 0.0342 | 0.0005 | -0.0337 |
| 0.3 | 0.0551 | 0.0006 | -0.0545 |
| 0.2 | 0.0756 | 0.0006 | -0.0750 |
| **0.1** | **0.0966** | 0.0007 | **-0.0959** |
| 0.05 | 0.0853 | 0.0006 | -0.0847 |
| 0.02 | 0.0529 | 0.0006 | -0.0523 |
| 0.01 | 0.0317 | 0.0005 | -0.0312 |

**THE CONTROL REPRODUCED BIT-FOR-BIT.** The shipped arm returns the published curve exactly
-- optimum 0.0966 at anchor 0.1, entity flat at 0.5115 across all seven points -- through a
day that changed the config gate, S14's flag, the resync helper and the eval windowing
default. So the harness is intact and the WIDE arm is interpretable.

**THE OPTIMUM DOES NOT MOVE: 0.1 either way.** The low anchor was NOT compensating for the
proposal cap, so the +0.0579 event_argument blind-test win stands on its own terms.

**BUT DO NOT READ WIDE'S ARGMAX AS AN OPTIMUM.** Its curve is FLAT at 0.0005-0.0007 across
the whole range, so "optimum at 0.1" is noise. What the column says is that the wider
proposal destroys the event head at EVERY anchor -- 138x worse at 0.1 -- which matches the
entity result rather than complicating it.

**THE OVERRIDE WAS CONFIRMED LIVE FROM INSIDE THE RUN**, which earlier attempts could not
establish: `[eval] boundary_head overrides applied: {'record_anchor_threshold': 0.01,
'record_anchor_threshold_wins': True, 'start_top_k': 128, 'end_top_k': 128, ...}`. The
eval-time allowlist and `resync_derived_settings` both work; the earlier silence was stdout
block-buffering through tee, not a decorative override.

## Item 3 priced: oversampling is a modest lever, supply is the binding constraint

The scorer ranks within a window, so exposure is measured in FULL-WIDTH WINDOWS, not
records and not raw tokens. Under `sliding_window: true, max_len: 4096, window_stride:
3072`, measured on the real training mix (79,182 records, 6,000 sampled):

| oversample of docs > 4096 tok | full-width windows | **% of windows** | % of tokens | compute/epoch |
|---|---|---|---|---|
| **1x (today)** | 2,336 | **2.90%** | 10.0% | 1.00x |
| 2x | 3,682 | 4.5% | 18.2% | 1.10x |
| 5x | 7,720 | 8.7% | 35.8% | 1.40x |
| **10x** | 14,451 | **14.4%** | 52.7% | **1.90x** |
| 20x | 27,912 | 22.9% | 69.1% | 2.91x |

**TOKEN SHARE AND WINDOW SHARE DIVERGE, AND WINDOW SHARE IS THE ONE THAT MATTERS.** At 10x,
long documents are 52.7% of TOKENS but only 14.4% of WINDOWS -- a long document yields a
few large windows while a short one yields a single small window. Quoting the token share
would overstate the exposure by ~3.7x.

**OVERSAMPLING ALONE CANNOT REACH A BALANCED MIX.** Full-width windows hit 50% only at
**153x** (16.3x compute per epoch, each of 937 documents seen 153 times). That is
memorisation, not training.

**SUPPLY IS THE BINDING CONSTRAINT**: 937 of 79,182 records (1.18%) exceed 4096 tokens, and
you cannot reweight past that. `cc_news_long` adds 689 documents, taking the pool to ~1,626
-- a 73% increase that roughly doubles what oversampling can reach.

**SO ITEM 3 IS: 10x oversampling PLUS cc_news_long, ~2x compute, for maybe 20-25% full-width
windows against 2.90% today.** Whether that moves a scorer currently ranking 50.3% of gold
starts is UNKNOWN -- there is no dose-response curve for long-context exposure, and this
programme's head-init curves have knees rather than being linear, so the honest position is
that the intervention is affordable and its effect is unmeasured.

**CORRECTION TO AN EARLIER FRAMING.** "Median training document is 457 tokens" is true and
misleading: those 1.18% of long records carry **8.36% of all training tokens**. The record
count understates long-document presence by ~7x.

## The event_argument headline is a CHINESE-CORPUS number (2026-09-30)

**THE QUESTION THAT EXPOSED IT.** Event arguments read 0.0000 in every long-document arm,
even with global_decode on. Before a long-document training mix, I set out to establish
whether events are decodable on long documents at all. The question was mis-posed.

**WHAT A TRACE SHOWED.** On cc_news_long the model emits NO arguments -- 0 predicted
argument keys against 95 gold over six documents -- and its few triggers are wrong (0 of 7
exact or even overlapping; `said, "` is a malformed span). Then:

- truncating the SAME documents to ~350 words still yielded 0 arguments, so it is not length;
- `batch_extract` and `batch_extract_long` both yield 0 on short RAMS docs, so it is not the
  long-document code path;
- the real entry point, `evaluate_checkpoint`, also yields 0 on those docs, so it is not my
  harness.

**PER CORPUS, 25 documents each, `evaluate_checkpoint`, eb17-best, threshold 0.3:**

| corpus | argument gold | event_argument strict | trigger strict |
|---|---|---|---|
| WikiEvents | 515 | **0.0000** | 0.0159 |
| CASIE | 588 | **0.0000** | 0.0403 |
| RAMS | 56 | **0.0000** | 0.0938 |
| **CMNEE** (Chinese) | 169 | **0.2019** | **0.7143** |

0 matches on 1,159 English argument gold across three corpora. At even 1% true recall that
would be ~12 matches, so this is not sampling noise.

**AND THE BLIND TEST IS COMPOSED ALMOST ENTIRELY OF THE ONE THAT WORKS.** Argument gold in the
eb17-best test split, by corpus:

| corpus | argument gold | share | text |
|---|---|---|---|
| **cmnee** | **18,414** | **88.4%** | Chinese |
| casie | 2,413 | 11.6% | English |

Only two corpora carry argument gold at all; WikiEvents and RAMS contribute none to the test
split. So `event_argument` on the blind test -- 0.1843 at training time, 0.1752 re-scored,
and the recorded **+0.0579 anchor-0.1 win** -- is a measurement of **CMNEE argument
extraction**, with an English minority that scores zero.

**WHAT THIS DOES NOT SAY.** It does not say the +0.0579 is wrong: it is real, validated, and
reproduced bit-for-bit. It says what it is a result ABOUT. Nor does it touch the EKF casualty
pipeline, which extracts through record structures rather than `event_argument` and has its
own held-out measurements.

**WHAT IT DOES SAY.** English event argument extraction is not demonstrated anywhere in this
programme's current metrics, and cc_news_long's zero is explained by that before any of the
cap, window or type-vocabulary findings are needed. Those findings stand on their own; they
are simply not why events read zero.

**THE MEASUREMENT FIX IS CHEAP AND OVERDUE.** `eval_by_language: false` in eb17 hid this, and
the blind-test JSON carries no per-label argument breakdown. Any event_argument number should
be reported per corpus, or at minimum per language, beside the aggregate -- otherwise a
headline built 88% from one corpus is read as a general capability.

### CORRECTION: the census and oversampling pricing measured the wrong population

"Item 3 priced" above, and the corpus length table in the a9725ac commit message, were
computed on the 8 EVENT FILES ONLY (79,182 records). The scripts read `data.train_files`,
which eb17-best.yaml does not have -- its keys are `corpora`, `train_only`,
`partial_annotation`, `event_files` -- so the training corpora were silently empty. Every
number in that section is withdrawn. "Median training document 457 tokens", repeated all
day, was `cc_news_haiku45`'s median, not the training mix's.

Re-measured with train.py:1549's own resolution (`_split_files(corpora, "train",
train_only) + _event_split(event_files, "train")`), 18 files, 202,211 records. **The gate
that makes this trustworthy: the sample yields 1.0061 windows per record against the
training run's own logged 201,072 -> 202,469 = 1.0069.**

| | withdrawn | **corrected** |
|---|---|---|
| median document | 457 / 596 | **362 tokens** (p90 1,362, p99 3,158, max 18,012) |
| records > 4096 tok | 1.18% | **0.43%** (859 records) |
| tokens in docs > 4096 | 8.36% | **4.93%** |
| full-width windows today | 2.90% | **1.24%** |

| oversample docs > 4096 | % windows full-width | % tokens long | compute/epoch |
|---|---|---|---|
| 1x | 1.24% | 6.0% | 1.00x |
| 5x | 3.63% | 24.1% | 1.24x |
| 10x | 6.36% | 38.9% | 1.54x |
| 20x | 11.11% | 56.0% | 2.14x |

50% full-width windows needs **415x** (not 153x). **THE CONCLUSION SURVIVES AND SHARPENS:**
oversampling cannot build a long-document regime, and supply is the binding constraint.
cc_news_long's 689 documents against 859 existing long records would roughly double the
pool. Long documents are rarer in the real mix because its Chinese corpora are short,
sentence-level records.

## English arguments are UNDER-LEARNED, not broken -- and what eb18 changes (2026-09-30)

**The question.** eb17 scores 0.0000 on English event arguments (WikiEvents, CASIE, RAMS) while
CMNEE scores 0.2019. Before buying English data, establish whether the English path is
BLOCKED (a defect data cannot fix) or merely UNDER-TRAINED.

**Traced on CASIE's own TRAINING documents** (eb17-best, `batch_extract`, whole documents, no
windowing -- CASIE's longest document is ~1,700 tokens, under one 4,096 window, so neither the
window nor `global_decode` is involved):

| threshold | mentions | arguments | correct strict | correct ignoring trigger | gold |
|---|---|---|---|---|---|
| 0.3 | 4 | 0 | 0 | 0 | 69 |
| 0.1 | 20 | 18 | 0 | 2 | 69 |
| 0.03 | 63 | 152 | 0 | 7 | 69 |
| 0.01 | 127 | 782 | 2 | 14 | 69 |

CMNEE, same code, emits and matches arguments (5 of 10, 1 of 3 on the first docs). CASIE's
arguments ARE scored -- lowering the threshold releases them -- but far below 0.3 and ranked
near-randomly (`dominate` offered as a Place and a Victim). The data is clean: 100% of CASIE,
CMNEE and RAMS triggers and arguments occur verbatim in their documents. **Verdict: English
arguments are under-learned.** Contributing: English arguments are 17% of training argument
gold (17,817 vs 85,540), all from CASIE's 798 documents, and are longer (2.25 words mean vs
CMNEE's 1.03). The check that distinguishes data from defect after eb18 trains: argument
recall on CASIE's TRAINING documents. If it stays near zero with 2-3x the English
supervision, something other than data is wrong.

### Reachability: how much gold can the proposer even offer?

`probe_candidate_coverage.py`, now SEEDED (training-mode preprocessing randomly drops labels,
so unseeded gold totals drifted 8,620-8,921 on identical data) and with dropout off. Gold
reachable before injection, 150 training docs from 10 corpora:

| arm | 1,024 tokens | 4,096 tokens (training window) | gold starts found @4,096 |
|---|---|---|---|
| per_query, start/end_top_k 16 (eb17) | 27.0% | 21.5% | 15.6% |
| per_query, 128 | 51.4% | 38.2% | 39.4% |
| shared, 64 starts / 384 spans (default) | 17.1% | 12.4% | 27.7% |
| shared, 128 / 768 | 22.1% | 15.8% | 39.8% |

Every arm scored the identical gold set (8,976 / 8,926) -- the seeding gate. Per corpus at
1,024 tokens: **CMNEE 87.9%, DocEE 58.6%, cc_news 38.8%, CASIE 25.3%** -- the same language
split as the scores. So the ceiling is mostly the BOUNDARY SCORER ranking English gold
poorly, not only the cap: even at 128 starts per query, only 39.4% of gold starts rank in.

**CORRECTION to "The cap" above.** That section says the cap is "invisible on our normal
corpora -- at 483 median tokens a window holds ~11 gold against a budget of 128". The budget
compared was the TOTAL candidate budget; the binding limit is 16 STARTS PER QUERY. Measured
through the real collator, 20.2% of gold spans sit in queries holding more than 16.

**shared cannot be reach-matched at sensible size.** At 128/768 it finds the same gold starts
as per_query 128 but keeps under half the spans: one pool per window versus up to ~384 slots
per query. Zeroing its untrained pairing layer moved nothing (12.4 -> 12.8%).

### CORRECTION: shared WAS A/B'd, on 2026-09-21

I told the user today that `candidate_pool: shared` had "never been A/B'd", repeating a stale
comment in `eb17-best.yaml`. It was: **attempt 2 (catalog, 2026-09-21), REFUTED, entity
-0.0745**, still -0.0435 after 4 epochs, events within noise. That verdict stands for entities:
#180 (below) lives in the shared RECORD loss, not the mention path the entity head uses. Only
the event heads could move in a rerun, now that #180 is fixed. What `shared` is, from the code
(`pool.py`): one deduplicated span pool per encoded SEQUENCE ("once per document" in upstream's
terms -- under our sliding window that is one WINDOW), scored by every query. It is not
document-level; `global_decode` is the only cross-window mechanism.

### Gold capacity: the true maximum is 619

Full training mix, 231,094 samples, 1.2M queries, training-time dropping OFF (it would
undercount): max gold spans per query **619** (WikiEvents); at the current cap 256, **11
samples (0.005%)** overflow and `skip_sample` drops each whole -- 9 of WikiEvents' 200 train
docs. eb17's own comment records why that is worse than wasteful: an emptied gold mask is
POSITIVE supervision to abstain (`abstention_loss`). At 768 nothing overflows.

### The annotation menu was binding

The Aug-2026 cc_news annotation offered each document a seeded random **6 of 56** event types.
Reconstructing each document's offered set from its seed: **5,116 of 5,116 recorded events
fall inside the offered 6, zero outside** -- and the check could fail, because the validator
accepts all 56 types. So cc_news's "sparse events" (18.4% of documents, 0.64 arguments per
document) was the menu, not the news. The new purchase offers all 56 (verified: 56/doc vs
6/doc). Pilot batch `msgbatch_01134CegRtA7QfdcbxPZkC34` (300 docs) measures the true yield.

### Decisions recorded for eb18-balanced (`tools/train/config/base/eb18-balanced.yaml`)

- **Recall-first operating point** (user): select the threshold and checkpoint as the MAXIMUM
  RECALL AT PRECISION >= X, not strict F1. X unset; the selection code does not exist yet.
- **Balance on argument gold**, ~50/50 eng/zho; human gold kept whole; LLM-annotated English at
  60 real-text : 40 synthetic-text (user's ratio). Supply of real-text English is binding.
- `global_decode: true`, `per_query` at start/end_top_k 128, gold capacity 768 with
  `truncate_with_warning`, `eval_by_language: true`. The config lists six pending items and is
  not launchable until they close.
