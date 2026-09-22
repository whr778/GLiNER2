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
