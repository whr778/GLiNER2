# Line 3 — full event support, and what could sink it

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

## 5. What "proves out" means

The line continues if, on an identical `event_argument` denominator:

1. **strict rises materially toward relaxed while relaxed HOLDS** — the binding signature;
2. **the guard heads do not collapse** — `event_type` 0.7545 and `event_trigger` 0.6051 are
   the things a record-head switch could plausibly damage;
3. **the per-role pattern is not uniform** — §4f.

It does not continue on a strict rise alone if relaxed rose with it: that is more data or
longer training, not better binding.

## 6. Deliberately deferred

Backfill is accepted as a cost of moving forward, and is listed so it is a decision rather
than an omission: the single-variable `event_records` run (drop the three corpora, ~$28), a
second event corpus in the blind test (§4c), and the argument-recall problem (§4b). None
blocks the next step; all block a defensible claim.
