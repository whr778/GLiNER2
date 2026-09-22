# Event arguments: why they fail, and what has been measured

> ## *** 2026-09-21: THE NEGATIVES LEVER, MEASURED CLEANLY AT LAST ***
>
> `absneg2` is the first legal comparison of absent-negatives-in-the-denominator:
> **event_argument strict 0.1840 -> 0.2215, +0.0376**, against a seed floor of 0.0009-0.0020.
> So this diagnosis's central head DOES move, by a mechanism that costs `classification`
> -0.1977 -- a head it has no route to. The lever is real and the price is currently
> unacceptable; see [[LABEL_NEGATIVES_PLAN]].
>
> Two earlier attempts at this measurement were void (injector never wired; then checkpoint
> selection), so any number for this lever dated before 2026-09-21 should not be quoted.

> ## *** 2026-09-21 (later): ALL FOUR OPTIONS ARE NOW CLOSED ***
>
> Option 2 was the last one open. Its trained margin is refuted on measurement -- see §Option 2
> -- so the structural menu this document opened with is exhausted:
>
> | option | verdict |
> |---|---|
> | 1 `candidate_pool: shared` | NEGATIVE (entity -0.089, argument null) |
> | 2 typed role constraints | **trained margin REFUTED (0.28% mean loss effect); decode-time a null on F1** |
> | 3 constrain by predicted entity type | NEGATIVE (-0.074 F1) |
> | 4 roles remapped into the NER space | NEGATIVE (event_argument -0.0300, trigger -0.0541, 0 up / 10 down) |
>
> **The only lever that has ever moved `event_argument` outside the floor is absent negatives
> (+0.0376), and it is not shippable at -0.1977 on classification.** That is the honest state
> of this line. The next question is therefore not "which option", it is whether the
> classification collateral can be repaired or routed around -- see [[LABEL_NEGATIVES_PLAN]].
>
> A structural reading worth carrying: options 2 and 3 both assumed the model ranks
> type-incompatible fillers too highly. Measured, it does not -- disallowed candidates hold
> w_S ~ 0.003 of the probability mass. The type signal is real as a DESCRIPTION of correct
> versus wrong arguments (88% vs 68% carry a predicted type) and useless as a CORRECTION,
> because the model has already applied it.

Companion to [[JOINT_IE_SCALING]] (Tier 2) and [[PAPER_0_FOUNDATION]] §10.

**This document grew by accretion and is ordered by when things were learned, not by how
confident they are.** §1-§4c are the original diagnosis; §4d onward are dated findings, some
of which retract earlier ones. Read this summary first — it is the only part kept current.

> **The title used to read "Why `event_argument` sits at 0.118".** That figure, and the whole
> "threshold 0.5" framing of the original status block, are **retracted** — the incumbent's
> model card says `Decision threshold: 0.3`, and `0.1178 / 0.5783 / 0.6051 / 0.7545` match
> neither the card nor any measured pass. §4d has the provenance. Inline `> *Retracted*`
> markers below flag every place the old figures still appear; they are left in place because
> they are the stated motivation for runs that really happened.

> **§4h and §4i are now being acted on.** The cause they identify — no label negatives, for
> any head but classification — has an implementation plan and working code:
> [`LABEL_NEGATIVES_PLAN.md`](LABEL_NEGATIVES_PLAN.md). Absent label queries now reach the
> model (0 → 306 of 865 queries on real data), and the numbers below are expected to be
> superseded by a model trained with them.

## WHERE THIS STANDS

**The incumbent, re-baselined.** `whr778/gliner2-eb16-rebuild-tr`, 18,786-record blind test,
validation-selected threshold 0.2: `event_argument` strict **0.0991** / relaxed **0.5884**,
`event_trigger` **0.5984**, `event_type` **0.8650**. (§4d)

**The diagnosis holds, and is now directly observed rather than inferred.** The mention path
keys event instances by TYPE, so N events of one type in a document pool into one. On twelve
blind-test documents each containing >=2 gold events of one type, the incumbent emitted two
instances in **0/12** and pooled multiple triggers into a single instance in **11/12** — it
finds 2-4 triggers per document and merges them every time, unioning their arguments. That is
why strict argument F1 sits near 0.10 while relaxed sits near 0.59. (commit `c840425`)

**No content-derived key fixes it.** `event_type` collapses 69.7% of gold instances,
OneIE's `trigger` span 39.9%, and the two together exactly 39.9% — adding the type buys
nothing. Only an *index* — one addressable slot per instance, which is what
`event_records: true` allocates — has a structurally guaranteed 0% collapse. (§4c,
`tools/data/event_multiplicity.py`)

**The record head works, on a 40%-trained checkpoint.** 5/12 multi-instance, 0/12 pooled, and
`event_argument` strict *precision* up **1.3-1.7x** over the incumbent at every matched
threshold. Its recall is 4-10x lower and climbing (+41% in one epoch), so the floor is
training, not structure — but the trajectory does not reach the incumbent's relaxed recall,
and that recall is bought by emitting 4,613 predictions for 927 gold triples anyway. **Judge
the finished model on strict F1 at its own calibrated point, not on relaxed recall.**
(§4e, §4f)

**Threshold is not the lever.** Sweeping it moves argument *recall* a lot — never-proposed
falls from ~48.8% to ~29.4% — and no F1 at all, because precision pays for it. (§4d)

**Label negatives: diagnosed here, BUILT since, and still not clean.** The training menu was
built from each document's own gold, so the model was never shown a type it must reject —
given a schema of only absent types it fires on **63%** of documents (incumbent) / **54%**
(event-records). Training and eval shared the blind spot, which is why no metric caught it.
(§4i) **Status 2026-09-22:** the injector exists and eb17 uses it, but three defects were
found the same week — the "0 absent queries" that motivated it was a HARDCODED PRINT (the
measured figure is 0.19%), the committed pools file was stale and gave **23.6% of eb17's
records zero negatives** (now `negative_pools: auto`), and `partial_annotation` never
reached the trainer. The realised negative ratio is **14.1%** against GLiNER v1's ablated
optimum of 50% — a dose sweep is TODO 17, unrun.

**The argument→trigger binding IS trained, and the headline spread predates it.**
`record_loss_weight` defaults to 1.0 and `compute_group_loss` seeds a record instance from
the ANCHOR, supervising field losses into it — so strict's trigger requirement is an
optimisation target, not merely a scoring one. OneIE trains the same binding in a different
form (pairwise (trigger, entity) classification, negative-saturated by construction, at
sentence scope) and then discards it at scoring time: **their metric is looser than ours.**
Crucially the 0.1178 / 0.5783 spread was measured WITHOUT `event_records`, on a record head
that had never seen an event and so had no event binding objective at all — it is not
evidence about the architecture eb17 trains. (§4c-i)

**`event_type` has never actually been scored.** Its precision is 1.0000 *by construction* —
the eval builds the type menu from the document's own gold, so no wrong answer is on offer,
and `F1 = 2R/(1+R)` in 12 of 12 readings. Against the corpus's real 8-type menu the incumbent
scores precision **0.5521**, not 1.000, and the event-records model is worse at **0.3352**.
Quote `event_type` as recall, never as F1. (§4h)

**Instrument defects found along the way, all fixed:** the blind test is NOT affected by the
gold-capacity cap but in-training eval IS (§4g); the checkpoint persists 92 `boundary_head`
keys and auto-loads them (§4g, correcting §4e); and `max_gold_per_query: 32` with
`skip_sample` was teaching the model to abstain on its richest documents — 3,774 firings in
the first event-records run, which was stopped and relaunched for it
(`tests/processing/test_gold_capacity.py`).

---

## 1. The model finds the arguments. It cannot attach them to the right event.

Same checkpoint, same predictions, three scoring keys:

> *Retracted — see §4d. Not at threshold 0.5; the card says 0.3.*

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

## 3. Was `event_records: true` ever configured? NOT UNTIL 2026-09-15 — and that was the answer to the obvious objection

> *Historical. It is configured now, trained, and measured — §4e, §4f. This section records why the question mattered.*

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

> *Retracted — see §4d. Not at threshold 0.5; the card says 0.3.*
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

> **This section was read from the PAPER and is kept for its quotations. For the settled
> position, read §4c-i, which was read from the SOURCE.**

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

> *Retracted — see §4d. Not at threshold 0.5; the card says 0.3.*

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

> *Retracted — see §4d. Not at threshold 0.5; the card says 0.3.*
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

### 4c-i. SETTLED, from OneIE's SOURCE (2026-09-22)

Everything above in §4c was read from the PAPER. The source was read on 2026-09-22
([`GerlinGreen/OneIE`](https://github.com/GerlinGreen/OneIE), a mirror of the Blender Lab
release). **This subsection is the current position; read it instead of the two above.**

#### The conclusion, in one line

**Both OneIE and this project train the argument->trigger binding. Their SCORING is looser
than ours. The forms of the objective differ, and the number everyone quotes was measured
on a model that had no event binding objective at all.**

#### What OneIE does, verified in code

```python
# graph.py -- one node per trigger SPAN; an argument is an EDGE to a specific trigger
trigger = (start_offset, end_offset, label_idx)
role    = (trigger_idx, entity_idx, label_idx)

# model.py -- arguments get their OWN loss, over (trigger, entity) CANDIDATE PAIRS
classification_loss = entity + event + relation + role_criteria(...) + mention
loss = classification_loss - entity_label_loglik.mean() - trigger_label_loglik.mean()
if use_global_features:
    loss = loss + (top_scores - gold_scores).clamp(min=0).mean()

# scorer.py -- and the metric then DISCARDS the binding
args.add((arg_start, arg_end, trigger_label, role))     # event TYPE, not trigger SPAN
```

#### What we do, verified in code

`record_loss_weight` defaults to **1.0** (`configuration.py:180`), and `compute_group_loss`
supervises `natural` mode by seeding an instance FROM THE ANCHOR:

```python
anchor_qid   = group.spec.anchor_query_id
anchor_f_idx = group.field_query_ids.index(anchor_qid)
seed_to_inst = {seed[1]: i for i, seed in enumerate(group.instance_seed)
                if seed is not None and seed[0] == anchor_f_idx}
for record in records:
    aft = record.field_for_query(anchor_qid)
```

The anchor span seeds the instance; field losses are supervised INTO that instance.
`record_object_loss` and `record_field_loss` both sum into the boundary loss.

#### Side by side

| | OneIE | ours |
|---|---|---|
| binding in the REPRESENTATION | `trigger_idx` edge | the record anchor |
| binding in the LOSS | yes -- pairwise classification over (trigger, entity) candidates, null role included | yes -- anchor-seeded instance assignment |
| binding in the METRIC | **no** -- keyed on event TYPE | **yes** -- strict is `(type, role, entity, trigger_key)` |
| negatives for that objective | **saturated by construction**: every non-argument pair is a null-role negative | from candidate sampling, not from the pair set |
| scope | sentence -- the pair set is small | document, 4096 tokens -- a pair set is quadratic |

#### The consequence that actually matters

**The 0.1178 strict / 0.5783 relaxed spread was measured WITHOUT `event_records`** -- on a
record head that, per §3, "was supervised on `json_structures` only and has never seen an
event". That model had no event binding objective at all. That is a sufficient explanation
for the spread, and it is **not evidence about the architecture eb17 trains**.

So the open question is not "add an edge objective". It is: **does the binding objective we
already have move strict argument F1, now that events reach the record head?** eb17 was
bought to answer that and has not yet. See [[TODO]] item 16; step 1 is still the free Arg-C
metric, open since 2026-09-15 and confirmed missing from `eval_metrics.py` on 2026-09-22.

*How this conclusion moved, including the two wrong turns taken to reach it: [[PROJECT_HISTORY]], 2026-09-22.*

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

**3. IT HAS ITS OWN MULTI-INSTANCE RESIDUAL — BUT IT IS SMALLER THAN OURS, AND THIS
PARAGRAPH USED TO SAY OTHERWISE.** *"Multiple events per trigger"* is a named category in
OneIE's remaining-error distribution, and the first version of this section concluded that
the span representation "trades one failure for another". **Measured, that is wrong.** It is
not a trade; OneIE's key is strictly better on our own data, and still not sufficient.

Any decoder addresses instances by *some* key, and two gold instances sharing a key value
collapse into one. So the cost of a design is readable off the gold before a model runs.
`tools/data/event_multiplicity.py` prices all three keys on the same blind test — the % is
gold instances that share a key with another instance in the same document:

| corpus | instances | key = `event_type` (ours) | key = `trigger` (OneIE) | key = both |
|---|---:|---:|---:|---:|
| cmnee | 6,553 | 67.5% | 44.3% | 44.3% |
| casie | 938 | 96.3% | 16.0% | 16.0% |
| mendeley_ed | 156 | 0.0% | 0.0% | 0.0% |
| **ALL** | **7,647** | **69.7%** | **39.9%** | **39.9%** |

Three things fall out of that table.

*The trigger key is strictly better.* 69.7% → 39.9% overall, and on casie 96.3% → 16.0%, a
six-fold reduction. The claim that OneIE merely moves the problem sideways does not survive
contact with the numbers.

*Adding the type to the trigger buys exactly nothing* — the last two columns are identical
to the decimal. Whatever the trigger does not separate, the type does not separate either.

*And 39.9% is still a plurality of the gold.* One node per trigger span is a better key, not
a solution. Worse, **39.9% is an upper bound on the collapse that flatters OneIE's side in
one direction and understates it in another**: our corpora store triggers as SURFACE STRINGS,
not offsets, so two separate occurrences of the same word are one key here where OneIE would
have two nodes. The true span-keyed residual on this data is *lower* than 39.9% and cannot be
measured without offset annotation we do not have — which is itself the cost of adopting the
design.

**The conclusion the table actually supports is that no content-derived key is the answer.**
`event_type` collapses 69.7%, `trigger` collapses at most 39.9%, and both fail for the same
reason: the key is a *property* of the event rather than its *identity*. The only key with a
structurally guaranteed 0% collapse is an index — one addressable slot per instance,
allocated by position and carrying no semantics. That is what `event_records: true` does, and
this table is the strongest quantitative argument on file for the line this project is
already building.

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

**A side-by-side of OneIE, line 2 and line 3 — with each approach's principal issue — is
in [[EVENT_LINE]] §4i.**

**Net:** OneIE is the right thing to have read and the wrong thing to copy wholesale. What
transfers is the insight that **trigger instances should be individuated by span rather
than by type** — which `event_records: true` achieves within the record head, without a
closed tag set and without giving up the document. What does not transfer is the tagger,
the sentence scope, or the hand-written feature templates.

---

## 4d. THE SWEEP RAN. The re-baseline, and two premises it broke

Run 2026-09-15. Validation grid on `whr778/gliner2-eb16-rebuild-tr`, winner picked on
relaxed argument F1, blind test scored **once** at that threshold — 18,786 records, supports
identical to the model card's (event_type 9,862 / event_trigger 14,041 / event_argument
strict 20,827), so it is the same test.

**Validation grid** (this is val, not test — the two are not comparable):

| threshold | arg relaxed F1 | arg relaxed R | arg strict F1 | trigger strict | type strict |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.5584 | 0.6568 | 0.2255 | 0.4818 | **0.9201** |
| **0.2** | **0.5875** | 0.5604 | 0.2789 | 0.5361 | 0.8254 |
| 0.3 | 0.5794 | 0.4954 | 0.3037 | 0.5421 | 0.7623 |
| 0.4 | 0.5555 | 0.4393 | 0.3115 | 0.5349 | 0.7185 |
| 0.5 | 0.5229 | 0.3882 | 0.3152 | 0.5197 | 0.6788 |

**Blind test at the val-selected threshold 0.2**, against the model card's own published
figures at its calibrated 0.3:

| head | card @ 0.3 | measured @ 0.2 | |
|---|---:|---:|---|
| event_argument strict F1 | 0.110 | **0.0991** | — |
| event_argument relaxed F1 | 0.606 | **0.5884** | — |
| event_argument relaxed **recall** | 0.632 | **0.7059** | **+0.074** |
| event_trigger strict F1 | 0.612 | **0.5984** | — |
| event_type strict F1 | 0.820 | **0.8650** | **+0.045** |
| event_type precision | 1.000 | **1.0000** | unchanged |

### The pre-registered question, answered: substantially CALIBRATION

§4b registered the reading rule in advance — *does relaxed recall move materially?* It does.
Relaxed argument recall runs **0.632 → 0.7059**, so the arguments *never proposed at all*
fall from the quoted **48.8% to about 29.4%** — a 40% relative reduction in missing
arguments, bought with nothing but a threshold. That much of the deficit was never a
capability problem.

**But it does not improve any F1**, because the recall is paid for in precision (0.582 →
0.5045) and strict F1 moves the wrong way (0.110 → 0.0991). Which is exactly what the
pooling diagnosis predicts: proposing *more* arguments cannot help when the failure is
**binding them to the right instance**. Calibration moves recall; only addressability moves
strict.

### Premise 1 broken: the incumbent was never at 0.5

This sweep was motivated by "every event number this project quotes is a single-point
reading at threshold 0.5". **The model card says `Decision threshold: 0.3 (calibrated
against the validation set)`.** The 0.5 premise was wrong, and the figures §1 quotes
(`0.1178` / `0.5783` / trigger `0.6051` / type `0.7545`) match neither the card at 0.3 nor
this run at 0.2. Treat them as of uncertain provenance and stop quoting them.

### Premise 2 broken: the validation pick did NOT transfer

On validation, 0.2 beat 0.3 on relaxed argument F1 by **+0.008** (0.5875 vs 0.5794) — inside
every run-to-run noise floor this project has measured. On test it does not beat the card's
0.3 either: 0.5884 against 0.606.

**Weight those two halves differently.** The val margin being noise is clean evidence. The
test comparison is *code-confounded* (see the caveat below), so it corroborates rather than
proves. The known code change — the `field_dtypes` cardinality repair — touches the RECORD
head, and the incumbent runs events through the MENTION path, so the event heads *plausibly*
ran identical code; that has not been proven and no other event-path commit has been ruled
out. Either way the protocol did its job: because the test was scored once at the
val-selected point we know this, instead of having fitted a number to it. **A threshold
sweep is not the lever here.**

### ~~The one free thing that IS on the table~~ — RETRACTED, see §4h

~~`event_type` precision is exactly 1.000 at threshold 0.2 ... that is F1 available for no
training, and nobody has looked.~~

**Struck.** The precision is 1.000 *by construction*, not because the head is conservative:
the eval hands the model a menu built from the document's own gold, so no wrong type is on
offer. Lowering the threshold raises recall toward 1.0 and F1 with it while measuring
nothing — the same failure as scoring a gate on firings rather than hits. §4h has the proof
and the real number (precision 0.5521, not 1.000).

### Caveat, because the comparison is card-against-now

The card was generated by an older checkout. `structure` strict moved 0.164 → 0.2595 between
them and structure is scored at the RECORD thresholds, not the global one, so that delta
cannot be the threshold — the code changed (the `field_dtypes_list` cardinality repair is
the known candidate). So the 0.3-vs-0.2 rows above are threshold *plus* code version, not
threshold alone. The single unconfounded statement is the measured column: **at threshold
0.2 on current code, the incumbent scores event_argument strict 0.0991 / relaxed 0.5884,
event_trigger 0.5984, event_type 0.8650** — and that, not `0.1178`, is what the
event-records base has to beat, at a threshold chosen the same way.

---

## 4e. THE EVENT-RECORDS BASE AT EPOCH 1: the mechanism works, the model is 40% trained

Run 2026-09-15 while the base was still training (epoch 1.9 of 4, step 21k/54.8k). Same
protocol as the incumbent — validation grid, pick on relaxed argument F1, score the blind
test **once**. Denominator verified identical: `scoring against 18786 records` in both logs.

Two config traps had to be disarmed first, and either would have produced a confident wrong
answer. The training config adds `professorbob_re`, `scierc`, `paraloq_json`, which carry
**1,816 test records** — scoring through it would have measured 20,602 against the
incumbent's 18,786. ~~And the checkpoint's own `config.json` carries no record keys at all, so
`event_records` comes from the YAML.~~ **STRUCK — see §4g.** That was read off the top level
of `config.json`, missing the nested `boundary_head` dict, which in fact holds 92 keys
including `event_records`. The checkpoint carries its own setting and the loader honours it.
The denominator above is the real and sufficient reason for the separate eval config.

### What improved: BINDING. Consistently, at every matched threshold.

`event_argument` **strict precision**, validation, same grid, same val set:

| threshold | incumbent P | epoch-1 P | ratio | incumbent R | epoch-1 R |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.1997 | **0.3455** | 1.73× | 0.2590 | 0.0621 |
| 0.2 | 0.2997 | **0.4501** | 1.50× | 0.2608 | 0.0462 |
| 0.3 | 0.3749 | **0.5499** | 1.47× | 0.2553 | 0.0384 |
| 0.4 | 0.4350 | **0.6177** | 1.42× | 0.2426 | 0.0287 |
| 0.5 | 0.4968 | **0.6444** | 1.30× | 0.2308 | 0.0219 |

Strict precision requires the argument to be bound to the **right instance**. It is higher at
every point on the grid, by 1.3–1.7×, on a model that is 40% trained and worse at everything
else. That is the record head doing the job it was added for, and it agrees with the
independent mechanism probe (`tools/train/probe_event_multiinstance.py`, commit c840425): 5/12 multi-instance and
0/12 pooled, against the incumbent's 0/12 and 11/12.

### What did NOT improve: everything else, because the model is undertrained

Strict **recall** is 4–10× lower. Blind test at each model's own val-selected threshold
(incumbent 0.2, epoch-1 0.1): event_type 0.8650 → 0.6271, event_trigger 0.5984 → 0.3124,
event_argument relaxed 0.5884 → 0.2358, and off the event heads classification 0.5800 →
0.1306, relation 0.2152 → 0.0068. It is behind across the board, as a 40%-trained model
should be.

### The claim NOT to make, and I nearly made it

On the blind test, `event_argument` **strict F1 reads 0.0991 → 0.1588, +0.0597** — the one
head the whole line targets, apparently improving. **Do not quote that as a win.** The two
numbers are at *different thresholds* (0.2 against 0.1), and the matched-threshold validation
comparison points the *other* way: the incumbent's strict F1 is higher at all five grid
points (0.2255 vs 0.1053 at 0.1, and worse from there). Precision carries the strict F1 up on
test only because recall is low enough to flatter the harmonic mean.

**What survives is the precision signature, not an F1 win.** Binding is better; the model
cannot yet propose enough arguments for that to become a score.

### One caveat that makes the epoch-1 numbers a floor

The validation pick landed on **0.1, the edge of the grid**, with every metric still climbing
as the threshold fell. The optimum is below the grid and was never bracketed, so every
epoch-1 figure here understates a properly calibrated epoch-1 model. When the run finishes,
**sweep below 0.1** — the incumbent's `event_type` head has the same untested tail.

---

## 4f. IS THE LOW RECALL UNDERTRAINING, OR A MISSING MECHANISM?

§4e left `event_argument` relaxed recall down 4-10x against the incumbent and could not say
whether that was a model at 40% of its schedule or a decoder that structurally cannot propose
more. Those have opposite consequences, so the question was settled rather than argued.

**The discriminator is a CURVE, not a level.** Same 150 cmnee documents, 927 gold
(type, role, entity) triples, matched threshold 0.1, two checkpoints of the SAME run
(`tools/train/probe_argument_recall.py`):

| checkpoint | recall | precision | F1 | predicted | hits |
|---|---:|---:|---:|---:|---:|
| incumbent (mention path) | 0.6839 | 0.1374 | 0.2289 | 4,613 | 634 |
| event-records **epoch 1** | 0.1165 | 0.2714 | 0.1630 | 398 | 108 |
| event-records **epoch 2** | **0.1640** | 0.2386 | **0.1944** | 637 | 152 |

### Answer: undertraining. It is climbing, and nothing structural is capping it.

Recall rose **0.1165 -> 0.1640 in one epoch, +41% relative**, and raw emissions rose 398 ->
637 (+60%). That is not a floor.

Three structural candidates were checked and cleared:

- **The record path has NO capacity cap.** `max_records`, `max_fields` and `max_spans` in
  `boundary_preprocessing.py:186-204` are each `max(...)` over the batch's own data, not
  constants. Nothing truncates record gold.
- **Per-role cardinality is a small ceiling, not this one.** 16.0% of gold event instances
  hold more than one filler for a single role, but that is only **7.5% of gold arguments**
  (1,743 of 23,252) — and only the 8 of 40 compiled field specs that are scalar are exposed
  to it at all. It cannot explain a 4-10x gap.
- **The one real truncation is on the MENTION path, which `event_records` moves events off.**
  `on_capacity_exceeded = "skip_sample"` drops a sample's gold **entirely** when ANY query
  exceeds `max_gold_per_query = 32` (`processing/targets.py:429-436`) — it fired **3,520
  times** in this run, on overflows of 34-66. It teaches the model to emit nothing on exactly
  the richest documents, and it plausibly explains the entity regression (0.5340 -> 0.4343);
  it should not touch event arguments under `event_records: true`. **Worth fixing on its own
  merits, separately from this line.**

### But do not read "climbing" as "will catch up"

Two reasons to expect the gap to persist, and they matter more than the trend:

1. **The trajectory does not reach it.** Linear extrapolation from +0.0475/epoch puts epoch 4
   near **0.26**, against the incumbent's 0.684. Training alone, at this rate, does not close
   it in the two epochs that remain.
2. **The incumbent's recall is bought by carpet-bombing, and is not a target worth matching.**
   It emits **4,613 predictions for 927 gold triples** — a 5:1 over-emission at precision
   0.1374. Relaxed recall rewards exactly that, which is this project's standing lesson about
   form gates scored on firings rather than hits. The record head emits 637 at precision
   0.2386, **1.7x better**, and strict scoring is what punishes the difference.

**So the honest statement is: the recall floor is training, the remaining schedule will not
erase it, and matching the incumbent's relaxed recall would mean adopting its over-emission —
the precise behaviour the strict metric exists to penalise.** The comparison to watch at the
end of the run is strict F1 at each model's own calibrated point, not relaxed recall.

---

## 4g. DOES THE GOLD CAP ALSO CORRUPT EVALUATION, AND DOES INFERENCE NEED A CHANGE?

Three questions, answered from the code rather than assumed.

### 1. The blind test is NOT affected. In-training eval IS.

`evaluate_checkpoint` -> `compute_metrics` (training/eval_metrics.py) contains no reference
to `pad_target_graphs`, `mention_mask`, `build_targets` or `collate_fn`: it scores model
predictions against the dataset's own gold. **Every blind-test number this document quotes is
therefore unaffected by the cap.**

In-training evaluation is a different path and it *is* affected. `trainer.py:670-681` passes
the SAME `on_capacity_exceeded` and `max_gold_per_query` into `collate_fn_inference` as the
training branch passes into `collate_fn_train` — the non-training branch deliberately drops
only `error_policy`. So a val sample whose query overflows has its gold cleared during eval
too, and the mention metrics take their denominator straight from that mask
(`training/metrics.py:54-55`: `total = targets.mention_mask.sum()`).

**The bias is optimistic and it is silent.** A skipped sample leaves both the numerator and
the denominator, so the densest documents are not scored as failures — they are not scored at
all. Per-epoch val numbers from any run using `skip_sample` are therefore measured on an
easier subset than the corpus, and "best checkpoint" selection inherits that.

### 2. Inference needs no `max_gold_per_query` change — but `candidate_budget` is its analogue

`max_gold_per_query` is a TARGET-building parameter; pure inference builds no targets, so it
cannot bite at decode. The inference-side ceiling is **`candidate_budget`, which defaults to
128** — the number of candidate spans enumerated per query. A query whose gold exceeds it
cannot have all of its spans proposed no matter how well the model scores them.

Measured on these corpora, 0.037% of (doc, label) query groups exceed 128, so it is a real
but small ceiling. Raising it costs decode compute for every consumer of the model, which is
why it is left at the default here and why the casualty configs likewise train wider than
they decode. `evaluate_checkpoint` accepts `boundary_overrides` if a specific eval needs it.

### 3. YES, the YAML is persisted into config.json and auto-loaded — AND A CORRECTION

A trained checkpoint stores **92 `boundary_head` keys**, `max_gold_per_query`,
`candidate_budget`, `training_candidate_budget`, `enable_records` and `event_records` among
them, and `AutoExtractor.from_pretrained` loads them. Consequences:

- The incumbent and the epoch-1 checkpoint both carry **`max_gold_per_query: 32`** baked in.
  The relaunched run will bake in 256 / 384.
- `event_records` is `True` in the epoch-1 checkpoint and `False` in the incumbent's — which
  is why the mechanism probe could tell them apart with no YAML involved at all.

**CORRECTION to §4e and to commit `ab13385`.** Both state that "the checkpoint's own
config.json carries no record keys at all, so `event_records` comes from the YAML", and that
evaluating through the incumbent's config would have run a record-trained model through the
mention path. **That is wrong** — it came from inspecting only the TOP level of config.json
and missing the nested `boundary_head` dict. The eval-config surgery was still right, but for
one reason rather than two: **the denominator**. The training config's three extra corpora
carry 1,816 test records and would have scored the model against 20,602 where the incumbent
used 18,786. The decode-path argument should be struck.

---

## 4h. `event_type` PRECISION IS 1.0000 BY CONSTRUCTION, and the head has never been scored

Noticed because "exactly 1.0000" held while recall moved from 0.5229 to 0.7621, across two
models, two splits and five thresholds. Real classifiers do not do that.

### The cause

`_schema_from_gold` (training/eval_metrics.py:367) builds the event menu from **the
document's own gold** — `schema["events"]` holds exactly the types present in that record and
no others. The model is asked *"which of these types are here?"* where every option is there
by construction. **It cannot emit a wrong type because no wrong type is offered.**

### The consequence, proved rather than argued

With precision pinned at 1, `F1 = 2R/(1+R)` exactly — no free parameter. Checked against
every `event_type` reading on file, two models × (five validation thresholds + one blind
test): **12 of 12 match to 1e-4.** Every `event_type` F1 this project has quoted is a
reparameterisation of recall and carries no independent information.

Entities, relations and arguments are handed a gold-restricted menu too, but they must also
land the **span**, so their precision stays free to be wrong — on the same blind test,
classification 0.5862, entity 0.5314, relation 0.3168. `event_type` is the only head with
nothing else to get wrong.

### What the blind test cannot see, measured

`tools/train/probe_event_type_fp.py` offers the corpus's FULL taxonomy — all 8 cmnee event
types, which is what a deployment does — over 150 blind-test documents at threshold 0.3:

| model | precision | recall | F1 | types invented |
|---|---:|---:|---:|---:|
| incumbent `eb16-rebuild-tr` | **0.5521** | 0.8373 | 0.6654 | 142 of 317 |
| event-records, epoch 2 | **0.3352** | 0.8708 | 0.4840 | 361 of 543 |

**The incumbent's real event-type precision is 0.55, not 1.00** — 45% of its type predictions
are wrong once wrong answers are available — and its honest F1 against a real menu is
**0.6654**, against the 0.8650 the blind test reports.

And a finding about the new line rather than the metric: **the event-records model is
markedly worse at type discrimination** — 0.3352 precision, 361 inventions against 142, while
emitting 543 type predictions where gold holds roughly 209. It over-generates types at a
threshold already *above* its calibrated point. That is a candidate cost to set against the
binding gain in §4e, and nothing in the standard eval would have shown it.

### What to do

- **Report `event_type` as RECALL**, never as F1, wherever the gold-only menu is in play, and
  state the menu whenever an event-type number is quoted.
- Treat the full-menu figures above as the honest ones.
- Watch the over-generation on the finished model.
- This does **not** touch `event_argument`, which must land the entity span, so its precision
  is real.

---

## 4i. THERE ARE NO LABEL NEGATIVES. Not for events, and not for NER either.

Asked directly: *were negatives ever implemented for events, the way they are for NER?* The
answer is that they were never implemented for **either**, and this is the cause behind §4h
rather than a separate issue.

### What the training menu actually is

`InputExample.from_dict` (training/data.py:1138) is the whole story:

```python
entities = output.get("entities")                    # gold keys ONLY
...
for evt_data in output.get("events", []):            # one Event per GOLD event
    events.append(Event(event_type=evt_data.get("event_type", ""), ...))
...
classifications.append(Classification(
    task=..., labels=cls_data["labels"],             # the FULL menu
    true_label=cls_data["true_label"]))               # answer kept separate
```

**Classification is the only head that sees a label it must reject.** Entities and events are
handed a menu built from their own gold, so every option on it is correct. Confirmed in the
data as well as the code: across fourteen corpora, **zero entity labels map to an empty
list** — there is not one explicit negative anywhere in `data/`.

The negatives that *do* exist are on the SPAN axis — `hard_negatives_per_positive: 5`,
`minimum_hard_negatives: 8`, selected by `select_hard_negative_candidates`. Those teach *"this
span is not a `Person`"*. Nothing teaches *"`Person` is not here"*.

### The behavioural consequence, measured

100 blind-test cmnee documents, each given a schema containing **only event types the document
does not have**. A model that learned to reject fires on approximately none:

| checkpoint | documents | fired on an ABSENT type | rate |
|---|---:|---:|---:|
| incumbent `eb16-rebuild-tr` | 100 | 63 | **63.0%** |
| event-records, epoch 2 | 100 | 54 | **54.0%** |

**The model cannot say "no" to a type that is not there, roughly three times in five.** That
is the same defect §4h measured from the other side (real precision 0.5521, 142 invented
types) and it explains it completely: a model never shown a wrong type has no reason to
reject one.

### Why it stayed invisible

`_schema_from_gold` builds the EVAL menu from gold too (§4h). So training never presents a
negative and evaluation never tests for one — the blind test cannot express this failure, and
reports `event_type` precision 1.0000 while the model fires on absent types 63% of the time.
**Training and eval share the same blind spot, which is why no metric on file caught it.**

### Why this is probably the largest lever on the table

Every other intervention in this document moves a fraction of a point. This one addresses a
precision of 0.55 where the reported number is 1.00. It is also cheap: sampling absent types
from the corpus taxonomy into each training schema is a data/collate change, not an
architectural one, and the same applies to entity labels.

The abstention head is the mechanism that would learn it — `abstention_loss` trains a
per-query gate whose target is 1 for an absent query (boundary/losses.py:601). With menus
derived from gold, that target is essentially never 1, so the gate has almost no positive
class. **NOT YET VERIFIED DIRECTLY**: an attempt to measure the absent-query rate through
`collate_fn_train` returned empty batches (wrong input shape) and was abandoned rather than
reported as zero. The claim rests on the code path and the data, not on that measurement.

---

## 4j. BLIND-TEST DENSITY, and why the row count is not what it looks like

Items per document in the test split, so an event number can be read against how much event
there is to find:

| corpus | docs | ent/doc | rel/doc | evt/doc | arg/evt | cls/doc |
|---|---:|---:|---:|---:|---:|---:|
| maven | 355 | 0 | 0 | **27.17** | 0.00 | 0 |
| casie | 107 | 19.96 | 0 | **8.77** | 2.60 | 0 |
| cmnee | 2,724 | 0 | 0 | **2.41** | 3.18 | 0 |
| mendeley_ed | 156 | 0 | 0 | 1.00 | 0.00 | 0 |
| biored | 45 | 17.04 | 10.91 | 0 | 0 | 0 |
| chfinann | 3,204 | 10.52 | 0 | 0 | 0 | 1.00 |
| docee | 2,744 | 6.95 | 0 | 0 | 0 | 1.00 |
| turkish_event | 3,090 | 5.98 | 0 | 0 | 0 | 1.00 |
| sentence_rex | 4,283 | 0 | 1.00 | 0 | 0 | 0 |
| **ALL** | **18,901** | **4.17** | **0.29** | **0.91** | **1.34** | **0.53** |

`arg/evt` is the one to keep: cmnee carries **3.18 arguments per event** and casie 2.60, so
the argument head's ceiling is set by two corpora. maven has 27 events per document and
**zero** arguments — it is trigger-only, and it dominates raw event counts while contributing
nothing to the argument metric.

### The raw row count is inflated by the CONFIG, not by the corpora

`[split hygiene] REPAIRED ... test 18786 records (dropped 10829 exact duplicate(s))` reads
like a repetitive corpus. It is not. **Seven corpora are listed in BOTH `data.corpora` and
`data.event_files` with the identical test path** — casie, chfinann, cmnee, docee, docfee,
events_biotech, text2json — so every one of their rows is loaded twice: **10,714 rows**, which
with ~115 genuine repeats gives exactly the 10,829 reported, and 18,901 + 10,714 = **29,615**,
exactly what the loader read.

Consequences:

- `REPAIRED` on this config is EXPECTED and is not a data-quality signal.
- **The protection is load-bearing.** `training.split_hygiene: warn` or `off` reproduces a
  pre-gate run — which here means scoring the blind test with **10,714 duplicated documents**,
  double-weighting seven corpora in every micro-averaged metric. The gate is silently holding
  that up.
- Do NOT "fix" it by deleting the `corpora` entries without checking: the two listings are
  identical for TEST but differ for VAL (`event_files` points at
  `data/scaling_joint/<corpus>.val.jsonl`, `corpora` at `data/<corpus>.val.jsonl`), so the
  edit is not the no-op it appears to be.

### Are the duplicates the same ANNOTATION? Both kinds exist, and both are handled right

Counted within each file loaded ONCE:

| | count | what it is |
|---|---:|---|
| config double-load | **10,714** | byte-identical, same file read twice — waste |
| exact repeats inside a corpus | **115** | same text AND same target — waste |
| **same text, DIFFERENT target** | **728** | **schema conditioning — PRESERVED** |

The 728 sit where the design expects: **text2json has 872 rows over only 186 distinct texts**
(619 variants) — one document presented with up to eight extraction schemas, which is the
signal that corpus exists to teach — plus sentence_rex 106 and events_biotech 3.

The arithmetic closes exactly: distinct `(text, target)` across all files loaded once is
**18,786**, which IS the scored blind test, and 115 + 10,714 = 10,829, which IS what the
loader dropped.

So **728 texts are scored more than once, each against a different schema** — intended, not
contamination. This is why the within-split rule is text AND target rather than text alone:
deduplicating on text would silently discard 728 supervised examples.


---

## 4k. LINKING NER TO EVENT ARGUMENTS — three options, and what the data supports

> **A FOURTH OPTION EXISTS AND IS THE ONE RUNNING (added 2026-09-19).** Options 1 and 3 here
> were both measured and both NEGATIVE — option 1 (`candidate_pool: shared`) moved the target
> not at all while costing entity −0.089, and it mattered because it was a PRE-REGISTERED
> rescue for option 3, so budget competition is not the binding constraint. Option 2 remains
> prototyped and unmeasured. **Option 4 — reframe an argument as a typed span the ENTITY head
> extracts — is the only one that adds EXTRACTION supervision rather than constraining
> binding, and so the only one that can move a span that was never proposed at all.** It is
> built and its A/B is running: see [[OPTION_4_ROLE_TO_ENTITY]] and [[EXPERIMENT_CATALOG]].
> Read the three options below as the state of the question before that was written.

In OneIE an argument **is an entity node**: the graph joins a trigger to an *entity mention*
via a role edge, so role classification is conditioned on entity type. In GLiNER2 an argument
is its own role query with no connection to the entity head. This section records whether that
link can be recovered here.

### The data says the structure is there — in exactly one corpus

Measured across every corpus carrying both entities and events:

| corpus | docs with both | argument mentions | also entity gold |
|---|---:|---:|---:|
| **casie** | 798 | **17,992** | **17,992 (100.0%)** |
| every other corpus | 0 | — | — |

**Every casie argument is an entity mention**, exactly OneIE's assumption. And roles are
strongly typed by entity type:

| role | entity types observed |
|---|---|
| `Time` | Time **100%** |
| `Vulnerability` | Vulnerability **99%** |
| `Attacker` | Person 80%, Organization 19% |
| `Compromised-Data` | PII 51%, Data 49% |
| `Victim` | Person 41%, Organization 35%, System 12% |

**But casie is the ONLY corpus with both**, and cmnee — 85.7% of the blind test's event mass —
has **zero** entity gold (§6). docee, chfinann and turkish_event have entities and no events.
So a role-to-type constraint learned from GOLD would train on 798 documents and could not fire
where the metric is decided.

### Option 1 — `candidate_pool: shared`. Config flag, zero new code.

`boundary_head.candidate_pool` is `"per_query"` by default and accepts `"shared"`
(`configuration.py:95`). Shared makes entity and argument queries score over ONE candidate
pool rather than enumerating per query, which gives the representational half of what OneIE
gets structurally, without any role-to-type constraint. Cheapest thing on this list and it
works everywhere, including corpora with no entity gold.

**ATTEMPT ONE (2026-09-17) MEASURED NOTHING, and the reason is a finding in itself.** Both
treatment arms died within a minute of starting:

    [config] boundary_head keys ['candidate_pool'] are structural and cannot be
    overridden on the `pretrained` path -- the checkpoint's modules are already built.

The guard is a good one and it was doing its job; `candidate_pool` simply should never have
been on its list. It sat in the hand-written first group of `_STRUCTURAL_BOUNDARY_KEYS`,
above the block marked "Measured 2026-09-05" — assumed structural, never checked. Measured
now by building the head both ways: **340 tensors under `per_query`, 340 under `shared`,
none added, none removed, none reshaped.** `model.py:246` builds `shared_pool_builder`
unconditionally and explains why in a comment — "present for checkpoint transparency even
while the default per-query path is selected". The flag selects a forward path at runtime.

The contradiction was sitting in this project's own prose. The A/B config header called the
flag STRUCTURAL and, in the same sentence, said its tensors "exist in every checkpoint". A
flag whose tensors always exist cannot be structural — nobody read the two halves together.
The loader's own error message names `candidate_pool` among "parameter-changing flags" as
well, which is a hint that can never fire, since a flag that changes no tensor cannot cause
a state-dict mismatch. All three texts are corrected.

**What the failed attempt cost, and what it bought.** One A10 hour, of which the `perquery`
control was a complete and reusable arm. It bought a narrowed guard, and a gate: the
treatment now has to prove itself through the gradient, because `shared_pool_builder` is
present in every checkpoint and an arm that failed to switch is indistinguishable from an
arm that switched and did nothing. Measured locally on one real backward —

| `candidate_pool` | shared-pool grad norm |
|---|---|
| `per_query` | 0.000e+00 |
| `shared` | 8.890e+00 |

— and `tools/lambda/pool_ab.sh` now aborts the run, before spending on a second arm, if an
arm's log does not show a non-zero norm.

**THE HANDICAP, which decides how a null is read.** Those shared-pool tensors receive no
gradient under `per_query`, so a warm-started shared arm begins from a freshly-initialised
pool while the control begins fully trained. A POSITIVE is therefore strong; a null at
matched steps is uninformative, which is why a `shared-long` arm at double the samples runs
beside it.

**ATTEMPT TWO (2026-09-17): THE TREATMENT APPLIED AND THE PATH DIVERGES.** With the guard
narrowed, both arms started, the gate read the shared pool as live — 64 tensors receiving
gradient — and the gradient was **`nan`**. Both arms then died identically:

    [pool] candidate_pool=shared  shared-pool grad norm nan over 64 tensor(s)
    FloatingPointError: 196 non-finite micro-batch loss(es) were zeroed

**196 is 49 optimizer steps x 4 accumulation — every micro-batch, from the first.** This is
not data-dependent, not a rare sample, and not the numerical-instability signature the event
line has seen before (sdpa+bf16 on mmBERT): FA2 was confirmed loaded at bootstrap and
`GLINER2_STRICT_ATTN=1` was set, so the known NaN cause is excluded.

**It is DETERMINISTIC, not reproduced — and the difference matters.** Both arms failing
identically looked like two independent confirmations. They were one: cmnee+casie train is
**10,079 records**, and the arms' `max_train_samples` of 13,000 and 26,000 are both ABOVE the
corpus, so neither cap bound. Both trained the same 10,079 records for the same 2 epochs — the
logs agree line for line, 1,260 optimizer steps each — so this is one configuration executed
twice. `shared-long` was supposed to be the longer arm and never was; length has to come from
`num_epochs`, and the config is corrected. A sample cap cannot manufacture samples the corpus
does not contain, and a second arm that silently equals the first is the same class of defect
as a treatment that silently equals its control.

**What has been ruled out, all locally and free:**

| repro | result |
|---|---|
| tiny head, fp32 and bf16 | finite, shared-pool grad 7.6 |
| checkpoint's real settings, random weights | finite |
| checkpoint's TRAINED weights, fp32 | finite |
| trained weights, bf16, 512 tokens | finite |
| trained weights + REAL cmnee records through the real collator, `event_records: true` | **finite** (per_query 1.73, shared 4.16) |
| the checkpoint's 64 shared-pool tensors | all finite, absmax ~0.09 |

Three further candidates were then tested and all stayed finite:

| further repro | result |
|---|---|
| `gold_injection_prob = 1.0` (its value at step 0; `per_query` never calls the module at all) | finite |
| bf16 **autocast** rather than `.to(bfloat16)` — what `bf16: true` actually does | finite |
| the full **backward**, not just the forward | finite: shared grad 6.03, total 83.4 |

Pool-based negative injection is excluded by configuration rather than by test: both arms set
`negative_labels_per_dim: {}`, and `_negative_labels` treats an empty dict as disabled, so
`NegativeLabels` never ran. The absent queries the log reports come from `negative_query_ratio`
in the boundary settings, which the control had equally.

**Every CPU-reachable variable is now finite, so the divergence is CUDA-side** and a Mac
cannot bisect further. One observation worth carrying into that debug: the shared arm's total
gradient norm is **83.4 against the control's 21.5**, about 4x, which is what an untrained
module in the middle of a trained network should look like — suggestive, but not itself a NaN.

**LOCALIZED 2026-09-17 (A10, ~15 min, ~$0.35).** `tools/train/debug_shared_pool_nan.py` hooks
every submodule and flags those whose OUTPUT is non-finite while ALL THEIR INPUTS WERE FINITE --
the conjunction matters, because once a NaN exists it propagates and "contains a NaN" names
dozens of modules. Both arms were run; `per_query` is clean at every step and proves the
harness.

| step | `per_query` | `shared` |
|---|---|---|
| 0 | loss finite, 0 non-finite grads | loss finite, shared-pool grad **5.40**, 0 non-finite grads |
| 1 | loss finite, 0 non-finite grads | **loss STILL FINITE (3.995), backward grad `nan`, 225 parameters non-finite** |
| 2 | loss finite, 0 non-finite grads | weights already corrupted; the embedding table itself now emits NaN |

**THE FORWARD IS HEALTHY. THE BACKWARD DIVERGES, AT STEP 1.** That is why every earlier repro
missed it — all of them were single-step, and step 0 is clean in both arms. By step 2 the
optimizer has written NaN into the weights, which is why `encoder.embeddings.tok_embeddings`
appears to "manufacture" a NaN: its weight is one. The 225 poisoned parameters include the
entire encoder, so one bad step destroys the whole model, not just the pool.

**It is CUDA-specific.** The same script run for FOUR steps on CPU stays finite throughout
(shared-pool grad 6.33 / 2.27 / 6.55 / 5.71), so step count was never the missing variable.
What CUDA adds is **bf16 autocast**: the GPU log shows `shared_pool_scorer.length_projection`
taking a finite fp32 input and returning bf16, i.e. the path runs under autocast, and the CPU
runs did not.

**ANSWERED 2026-09-17 (A100, ~2 minutes). IT IS A PRECISION FAULT, NOT A PATH FAULT.**

| precision | step 0 | step 1 | steps 2-4 |
|---|---|---|---|
| **fp32** | clean, shared grad 5.90 | clean, 2.45 | clean -- 7.52, 5.62, 3.85; ZERO non-finite parameters throughout |
| **bf16** | clean, shared grad 6.57 | **grad `nan`, 225 parameters poisoned** | model destroyed |

**`candidate_pool: shared` is sound. bf16 autocast breaks its backward**, and it does so at
step 1 with a finite forward, which is why every single-step and every CPU repro missed it.

Two consequences, and the second is the one that changes what to do:

1. The fix is a targeted `autocast(enabled=False)` around the offending op in the shared-pool
   backward -- but the op still has to be named, which needs a backward-side instrument
   (`register_full_backward_hook`) rather than the forward hooks used so far.
2. **The treatment can only train in fp32 today, which makes the existing `perquery` arm
   unusable as its control** -- that arm is bf16 AND ran on an A10, so it confounds precision
   and card with the thing under test. It is kept as a rough reference only. A `perquery-fp32`
   control now runs on the same box, same precision, same data, same steps.

**Why an fp32 A/B is worth ~50 minutes even though fp32 is not shippable:** it decides whether
the bf16 bug is worth fixing at all. If `shared` is a wash even in fp32, the autocast fix buys
nothing and this option closes; if it wins, the fix is justified and the real bf16 comparison
follows.

**The standing reading of this, until it is resolved:** `candidate_pool: shared` has never
been trained in this project — checked, not assumed. The guard only ever blocked the
`pretrained` path, so a from-`encoder:` config could always have selected it; searching the
whole config tree and all of git history for `candidate_pool: shared` returns only the three
A/B configs written for this experiment on 2026-09-17. The flag is reachable, its modules are built
and saved in every checkpoint, and the architecture paper documents it as an option — but
being listed as structural meant no training run could ever select it, so nothing exercised
its backward pass. The loss is 2.4x the control's on identical CPU input, which is consistent
with an untrained module and tells us nothing about whether it would converge. **Option 1 is
therefore not "unswept"; it is unimplemented in practice**, and pricing it means debugging a
path with no training history, not running a config flag.

### Option 2 — extend `joint_ie`'s typed constraints to event roles.

> *** RESOLVED 2026-09-21: the TRAINED margin is refuted; the DECODE constraint is a null. ***
> Option 2 was the last of options 1-4 still open, and it is now closed on measurement rather
> than on argument. `typed_margin_mask` was built and the ceiling measured on held-out val:
> the mask fires (0.339% of candidate cells, 91 queries carrying a disallowed competitor) but
> the probability mass on disallowed candidates is **w_S median 0.00003, mean 0.00284**, so
> the loss effect at delta=ln2 is **0.00% median / 0.28% mean**. The model already scores
> type-incompatible fillers near zero -- it was trained on this data -- so the margin spends
> its effort on candidates that were never competing. The prediction below that a model
> "TRAINED with the constraint" would beat filtering after the fact is NOT supported.
> Full numbers and the three brackets in [[OPTION_2_TYPED_ROLE_CONSTRAINTS]].

The machinery already exists **for relations**: `relation_specs` carry `head`/`tail` entity-type
constraints defaulting to `entity_types` (`joint_ie/engine.py:281-289`), and the joint beam
already decodes records through role edges. Relations get this naturally because head and tail
ARE entity spans; event roles are the same shape and simply are not wired in.

A real but bounded change. Its payoff is limited to corpora carrying both, i.e. casie — so it
would be measurable but not decisive on the current mixture.

> *Updated 2026-09-19 — of options 1-3 this is the ONLY one still open, and the evidence for
> it got STRONGER while the others failed.* Option 1 measured negative (entity −0.089, argument
> null) and option 3 measured negative (−0.074 F1). But the 800-document sweep that killed
> option 3 also found the type signal is real and **role-dependent**: 88% of correct arguments
> carry a predicted entity type against 68% of wrong ones, and `Location`/`Date` discriminate
> where `Subject` does not. Its own conclusion names what is not ruled out — "a model TRAINED
> with the constraint rather than filtered after the fact" — and that is precisely option 2,
> because `joint_ie`'s typed constraints live in the BEAM rather than in a post-hoc filter.
> The sweep also says how: **per role, not globally.** Bounded payoff, unbuilt, and now the
> best-supported of the three. Implementation plan: [[OPTION_2_TYPED_ROLE_CONSTRAINTS]].

### Option 3 — constrain argument candidates by the model's OWN entity predictions at decode.

The closest analogue to OneIE that this data can support. OneIE's argument candidates *are*
entity mentions; we do not need entity GOLD in the event corpus for that, only an
entity-capable MODEL — and ours is, trained on 89,040 entity-bearing records. At decode,
restrict or re-rank argument candidates by what the entity head scores highly. **This works on
cmnee despite cmnee having no entity annotation**, and it is an EVAL-TIME change over one
trained checkpoint, so it needs no retraining.

> *Retracted 2026-09-19 — the status below is superseded by the 800-document sweep later in
> this section ("SWEPT AT 800 DOCUMENTS: the route is NEGATIVE"). Option 3 WAS measured:
> no menu and no filter beat the no-menu baseline, best configuration **−0.074 F1**, and
> merely adding an entity menu cost 30-35% of argument recall BEFORE any filter. The
> prototyping note is kept because it records the decode-path trap that delayed the
> measurement, which is worth knowing on its own.*

**Status: prototyped, NOT yet measurable.** A first attempt scored an oracle bound on casie
(filter predicted arguments to gold entity surfaces, which by the 100% result above should
cost zero recall). It returned 4 true positives over 60 documents — because the probe calls
`extract_events`, which builds an EVENTS-ONLY schema, while every argument number in this
document comes from `compute_metrics` + `_schema_from_gold`, which builds entities, events and
classifications together and yields ~20x more arguments on the same corpus. **The filter has
to be hooked into `compute_metrics`, not layered over `extract_events`** — the two decode
paths are not interchangeable, which is worth knowing independently of this feature.

### Option 4 — REMAP ROLES INTO THE NER SPACE, and inject argument spans as entities

Proposed 2026-09-17. Options 1-3 all constrain BINDING — which span fills which role. This one
**bypasses binding entirely**: an argument reframed as a typed span is extracted by the ENTITY
head, needing no trigger link. Given that strict argument F1 is 0.10 against relaxed 0.59, and
the gap IS binding, sidestepping it is a different and possibly larger lever than constraining
it.

**Precedent exists and is half of this already.** `tools/data/events_to_entities.py` maps
`event_type -> entity label` using the TRIGGER surface, which is how `maven_ner` was built.
Option 4 is the other half: `role -> entity label` using the ARGUMENT surface.

**The supply is large, and concentrated where it is most needed.** 484,424 argument mentions
across 953 distinct roles, including the corpora with NO entity gold at all:

| corpus | argument mentions | distinct roles | docs with entity gold |
|---|---:|---:|---:|
| casualty_events | 124,561 | 4 | **0** |
| cmnee | 62,573 | 11 | **0** |
| duee | 28,875 | 121 | **0** |
| rams | 17,026 | 65 | **0** |

Top roles overall are already entity-shaped: `Subject` (60,255), `location` (53,779), `Date`
(41,812), `Equipment` (24,043), `Object` (22,844).

**THREE GUARDS, each from something already measured:**

1. **PARTIAL ANNOTATION IS THE SERIOUS RISK.** Non-argument entities in those documents stay
   unlabelled. With label negatives live, a derived corpus would be marked as annotating
   entities and would draw entity negatives against gold it never annotated — precisely the
   contradiction `build_negative_pools.py`'s within-dimension rule exists to prevent,
   self-inflicted at scale. **The derived corpus must be flagged PARTIAL: a source of
   positives, never of negatives.**
2. **ROLE IS NOT ENTITY TYPE.** casie measures `Victim` -> Person 41%, Organization 35%,
   System 12%. Mapping a polysemous role onto an existing NER type is wrong most of the time.
   Keep the ROLE NAME as the label — deterministic by construction — and map only the clearly
   entity-typed ones (`location` -> `Location`, `Date` -> `Date`) through `labels_file`, which
   is what that machinery is for. This also keeps the rewrite out of the corpora, per the
   standing rule to unify in the CONFIG.
3. **INVENTED-LABEL CORPORA MUST NOT DRIVE IT.** `zh_multitask` shows **791** distinct roles
   and `mix_natural` 156 — the "annotator invented labels per document" problem CLAUDE.md
   already names. Only taxonomy corpora contribute, measured first.

**And a precedent to avoid:** docee, chfinann, docfee and turkish_event were converted
DESTRUCTIVELY — events replaced by entities plus classifications — which is why they train zero
events today (§6). Option 4 must be ADDITIVE: a derived corpus alongside the original, never a
replacement.

**Why this is worth doing before options 2 and 3:** both of those need entity gold to coexist
with events, which today means casie alone at 798 documents. Option 4 MANUFACTURES that
coexistence for cmnee, duee, rams and casualty_events — so it is also the enabler for the other
two, not merely an alternative to them.

**BUT OPTION 2 HAS A CIRCULARITY TRAP, and it constrains how option 4 must label.** Option 2's
value is constraining "role R is filled by entity type T" where T is determined INDEPENDENTLY.
If option 4 derives the entity label FROM the role (`Victim` role -> `Victim` entity label),
then "`Victim` role <- `Victim` entity" is a tautology: it runs and teaches nothing. The two
guards pull against each other —

| labelling choice | safe? | enables option 2? |
|---|---|---|
| keep the role name as the label | yes, deterministic by construction | **no, tautological** |
| map onto a real NER taxonomy | **no** — `Victim` -> Person 41% / Org 35% / System 12% | yes |

The casie measurement resolves it: **roles whose NAME IS A TYPE NAME are deterministic** —
`Time` -> Time 100%, `Vulnerability` -> Vulnerability 99%, and `location` (53,779) and `Date`
(41,812) are the high-volume cases. **Roles naming a semantic FUNCTION are not** — `Victim`,
`Attacker`, `Subject`, `Object`. So map the type-named roles onto the canonical taxonomy and
keep the function-named ones as role labels; option 2 then gets real constraints over the
deterministic subset, which is also the largest by volume.

**And option 2 has a second route that does not need option 4 at all:** use the model's
PREDICTED entity types as the type signal instead of gold — option 3's mechanism feeding option
2's constraint. That works on cmnee today, with no data derivation.

### MEASURED 2026-09-17: the predicted-entity-type route works, modestly

The cheapest route — option 3's mechanism feeding option 2's constraint, using the model's OWN
predicted entity types instead of gold, so it runs on cmnee today with no data derivation.
Probe: `tools/train/probe_predicted_entity_types.py`, 120 cmnee documents through the EVAL path
(`_schema_from_gold` schemas via `model.batch_extract`) with a 14-type entity menu added, which
is the intervention — cmnee has no entity gold, so the eval schema normally carries no entities
and the head is never queried at all.

| | correct arguments | wrong arguments |
|---|---:|---:|
| has a predicted entity type | 312 (**88%**) | 92 (**68%**) |
| no entity type predicted | 44 (12%) | 44 (**32%**) |

"No entity type" is **2.6x enriched among wrong arguments**. The naive filter — drop arguments
the entity head does not recognise as anything — moves precision **0.7236 -> 0.7723 (+0.049)**
for **−12.4% relative recall**. Real, and the same precision-for-recall shape as the negatives.

**The per-role detail reproduces the casie split from independent data:**

| role | correct | wrong |
|---|---|---|
| `Date` | Date:31, Time:25 | Date:6, Time:2 |
| `Location` | Location:11 | **none at all** |
| `Subject` | Aircraft:247 | Aircraft:81 |

`Date` and `Location` — TYPE-NAMED roles — get consistent, discriminating types. `Subject` — a
FUNCTION-NAMED role — gets `Aircraft` for correct and wrong alike, so type identity carries
nothing (cmnee is military news; subjects are aircraft either way).

**That is the casie gold split arrived at from model predictions on a corpus with no entity
annotation whatsoever.** Two independent measurements agreeing on WHICH roles are typeable is a
far stronger basis for option 2 than either alone, and it says the constraint should be applied
PER ROLE rather than globally: enforce it where the role is type-named, leave it off where it
is function-named.

### SWEPT AT 800 DOCUMENTS: the route is NEGATIVE, and the pilot had no baseline

The 120-document pilot above reported "+0.049 precision" and was **wrong about the
conclusion**. Swept properly — four menus x three filters, 800 cmnee documents, same eval path:

| menu | filter | P | R | **F1** |
|---|---|---:|---:|---:|
| **none (baseline)** | — | 0.5430 | **0.2495** | **0.3419** |
| minimal 6 | none | 0.5875 | 0.1639 | 0.2563 |
| minimal 6 | global | 0.6684 | 0.0259 | 0.0498 |
| curated 14 | none | 0.5835 | 0.1591 | 0.2500 |
| curated 14 | global | 0.6543 | 0.1277 | 0.2136 |
| wide 28 | none | 0.5967 | 0.1724 | 0.2675 |
| wide 28 | global | 0.6632 | 0.1202 | 0.2035 |
| wide 28 | per-role | 0.5928 | 0.1506 | 0.2402 |

**No menu and no filter beats the baseline.** The best menu configuration is −0.074 F1.

**FINDING 1 — merely ADDING an entity menu cannibalises argument extraction.** Recall falls
0.2495 -> 0.16-0.17, a 30-35% relative loss, BEFORE any filter. Precision rises a little
(0.543 -> 0.59) and F1 falls hard. Offering entity queries beside event queries competes for
candidate budget, which is exactly what **`candidate_pool: shared` (option 1)** addresses — so
this raises option 1's value rather than lowering it.

**FINDING 2 — the pilot had no baseline, and that is the error.** It compared filter-on against
filter-off WITHIN the menu condition and never against no-menu at all. The menu was costing a
third of recall the whole time, invisibly, because only the expected comparison was
instrumented. **This is the second time in two days that a result looked positive because the
control was absent** (the other: the negatives A/B, until §5d's base reference). The rule worth
keeping: **when an intervention has two parts — add a menu, then filter on it — the baseline is
NEITHER, not the first part.**

**What survives.** The correlation is real: 88% of correct arguments carry a predicted type
against 68% of wrong ones, and `Location`/`Date` discriminate where `Subject` does not. The
signal exists; under post-hoc filtering with `candidate_pool: per_query` it costs more than it
returns. Two things would change the verdict and are NOT ruled out: a model **trained** with the
constraint rather than filtered after the fact, and `candidate_pool: shared`, which targets the
cannibalisation directly.

### The ceiling on options 1-3, and why option 4 is not bound by it

`event_argument` relaxed recall is 0.42 and strict 0.30, so roughly a third of the loss is
arguments **never proposed**. Typing constrains what IS proposed; options 1-3 attack the
binding half and none of them recovers the missing third.

**Option 4 is the exception**, which is the argument for it: routing argument spans through the
entity head adds EXTRACTION supervision rather than constraining binding, so it is the only one
of the four that can move the never-proposed third. It also gives up the event grouping for
those spans — a span extracted as an entity is not attached to an instance — so it is a
complement to the record head, not a replacement for it.

## 5. What follows, in order

> *Written before §4d-§4g. Item 1 is kept current; items 2-4 are the plan as it stood
> and are superseded in part -- item 3 in particular is now answered: relaxed and strict
> move in OPPOSITE directions under threshold (§4d), so reporting both is not optional.*

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
   **STOPPED at epoch 2.4 and RELAUNCHED the same day.** The first attempt ran under
   `max_gold_per_query: 32`, which with `on_capacity_exceeded: skip_sample` cleared gold
   for every query in any document where one query overflowed -- 3,774 firings, 3.65% of
   samples, each one supervised to ABSTAIN rather than merely left unsupervised (§4g).
   Record targets survive that, so the event heads were not corrupted, but the decision
   was to stop-loss rather than pay twice for the same model. The relaunch runs at
   `max_gold_per_query: 256` / `training_candidate_budget: 384` with **zero** firings,
   at 16.4 samples/s against the old run's 15.7 -- the raised cap is free in wall clock.
   NOTE the incumbent control is still a cap-32 model, so the comparison now carries two
   differences; rebuild the control with the same cap when the line is next re-based.
   Checkpoints from the aborted run are kept: `whr778/gliner2-eventrecords-ep1`.

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
- **AND THE MIXTURE IS NARROWER THAN THE CONFIG LOOKS.** `eb16-rebuild-tr.yaml` names nine
  event corpora. Counted record by record, **only three of them carry a single gold event
  instance in the blind test** — cmnee 6,553 (85.7%), casie 938 (12.3%), mendeley_ed 156
  (2.0%). Every event number this project quotes is therefore, to within a rounding error,
  a cmnee number.
- **Five of those nine train zero events too.** `docee`, `chfinann`, `docfee`,
  `turkish_event` and `events_biotech` carry no `events` key in **train or test** — 90,103
  training records between them, converted to `entities` + `classifications`, so the event
  head never sees them. They are event corpora by provenance and not by content. That is
  worth knowing before concluding anything about how much event data the model has had:
  `data/` holds 343,115 gold event instances across 25 corpora, and this config reaches a
  small fraction of them.
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
