# Project Journal — Global Inference on Boundary-Head Candidates

> **⚠ 2026-08-18: every joint_ie / 137k number in this document predates a data repair
> and is superseded.** 45 corpora shipped overlapping train/val/test, and the scaling
> configs additionally paired regenerated train files against frozen val slices
> (252 train-in-val, 22 val-in-test). All were repaired, the slices rebuilt, and the
> four scaling points now gate CLEAN. The curve is being re-run from scratch — see
> `JOINT_IE_SCALING.md` and [[lambda-137k-curve-restart]]. Numbers below are kept as
> the record of what was measured and believed at the time; do not compare new results
> against them.

A chronological record of what was decided, why, and what later proved wrong. The other
working papers state conclusions; this one states the *path*, including the reversals,
because several of the most useful results in this program are corrections to earlier
results in this program.

Dates are commit dates. Numbers are measured unless marked as an estimate.

Companion documents: [[RESEARCH_PROGRAM]] (thesis + paper map), [[EKF_MHT_DESIGN]]
(tracker line), [[JOINT_IE_SCALING]] (curve line), [[BOUNDARY_ARCHITECTURE]] (the head),
[[PAPER_0_FOUNDATION]] (foundation draft).

---

## Origin — September 2024

The idea does not start in the commit log, and a journal built only from `git log` would
miss it by two years.

**September 2024** — Weaver's *"Exposing the Power of the Kalman Filter"* (Towards Data
Science, 7 Nov 2023) is read. It walks the predict/update cycle from first principles and
ends by motivating the **Extended** Kalman filter for nonlinear systems. The question it
raises — could a Kalman filter track a real-world quantity reported across a stream of
documents? — is the one this whole programme exists to answer.

**September 2024 – August 2026** — turned over, not built. The first commit of this line
lands 2026-08-07, roughly two years later.

**The question, in the form it was actually asked:** *can a Kalman filter track real-world
events reported in text, and **diarize** them into separate streams?* Two halves — track,
and decide which stream each observation belongs to. The second is not a downstream detail;
it is half the question, and it is the half that has failed in every real-event test so far.

That ordering is worth recording because it is not recoverable from the code, and it
explains the shape of everything below: **the filter came first and the extraction second.**
Document-level event extraction, the boundary head, the counting layer and the entire
joint_ie line are not the goal — they are what the original question turned out to require
before it could be asked properly. It is also why [[EKF_MHT_DESIGN]] §1 scopes its claim so
narrowly (*streaming, quantitatively-evolving events*, explicitly NOT "EKF helps event
extraction") — that scope is the original question, kept honest.

---

## Phase 0 — inherited baseline (to mid-July 2026)

The repository begins 2025-07-07 as GLiNER2 proper: a span-architecture extractor with
entity, relation, classification and JSON-structure heads driven by query markers. Nearly
all of that predates this research program. The program starts where the foundation paper
starts.

---

## Phase 1 — foundation, and the head-init finding (16 Jul – 5 Aug)

**Threshold calibration became a first-class step** (16 Jul). A post-training sweep picks
the operating point rather than assuming 0.5. This choice pays off, badly and twice, much
later — see the matched-threshold entries below.

**Document-level extraction was scoped** (17 Jul): `DOCUMENT_EXTRACTION_PLAN.md`, a global
event decoder, and an opt-in `global_decode` path. The idea that a greedy per-query decode
leaves something on the table is the seed of the whole joint_ie line.

**Infrastructure, deliberately, before results** (18–23 Jul): spot-resume, "fail loud on
missing `metric_for_best`", event converters (ChFinAnn, DocFEE, DuEE, Mendeley), the
extraction viewer, HF push tooling. The `metric_for_best` guard is worth noting — it was
added defensively and later caught a real failure.

**Decision: report fair span diagnostics** (16–17 Jul), following Ortmann 2022, instead of
a single micro-F1. This is the first instance of a habit that recurs throughout: prefer the
measurement that can embarrass you.

**The head-init finding** (31 Jul, §10.6): training *from an encoder* with fresh heads
behaves very differently from warm-starting them. Follow-ups on 3 Aug recorded a
**negative** A/B and a sanity run showing fresh heads cannot bootstrap from synthetic data
alone.

**The mmBERT head-init scaling curve** (4–5 Aug): 10K / 40K / 100K base points, RAMS
argument-strict F1 **0.050 → 0.115 → 0.158**. Knee between 10K and 40K, still climbing at
100K. Recorded then as a data-volume law about mmBERT.

> **Later overturned.** See 9 Aug: the same curve on the *boundary* head is nearly flat
> (0.177 → 0.202). The deficit was a property of the **span head**, not of mmBERT or of
> data volume. The original reading was not wrong about its own numbers; it was wrong
> about what they generalised to.
>
> **And 15 Aug: "nearly flat" is really "flat within noise".** A control re-run of the
> published 137K recipe scored +0.023 above it, so single-run variance is ≥±0.02 and the
> whole boundary spread (0.177–0.215) fits inside it. The span-head conclusion survives
> and is strengthened — span climbs 0.108, five times the variance — but no shape should
> be read off the boundary row.

---

## Phase 2 — the architecture pivot (24 Jul – 6 Aug)

**joint_ie decoding lands** (24 Jul), then **the boundary architecture** (27–28 Jul) — a
head with no span-width cap, replacing the span head's 20-width lattice.

**Decision: adopt boundary as the architecture for this program** (JOINT_IE_SCALING
decision 1). Rationale: dense document-level relations are exactly where a 20-width cap
bites, and the whole thesis is about output that does not fit a per-query greedy decode.

**Merge of origin/main** (5 Aug) brought the boundary rewrite in as the baseline, with our
features re-ported on top over seven commits. A parallel decision from the same week
(21 Jul) kept our DDP implementation over main's while adopting main's tests.

**Counting layer** work (6 Aug) with a documented 19-instance cap.

---

## Phase 3 — the EKF/MHT line, built in a day (7 Aug)

Seventeen commits, and the shape of the argument matters more than the count.

**Decision: synthetic streams before real news.** A parametric disaster-stream generator
with regimes, so the tracker could be developed against ground truth that actually exists.
Cost: $0.

The sequence was deliberately adversarial to its own thesis:

1. tracker + baselines harness
2. **MoE gate (Reading B) — "beats both baselines and the EKF"**, recorded as such
3. harder-regime ablation + an estimate-scaled `R` fix, where the EKF earns its keep
4. learned gate, cross-regime transfer, union gate
5. text → observation extraction and normalization
6. Sonnet-5 realizer (real prose from synthetic streams) + a one-sided innovation gate
7. model extraction arm; **confidence-as-soft-R tested and recorded as a negative**
8. the `missing`-role probe: the weak role was a **confidence-cut selection bias**, an
   extractor problem, not a normalization problem
9. fine-tune on 250 realized streams — end-to-end EKF **0.291 → 0.165**, `missing`
   **0.458 → 0.122**

**Decisions with reasons, recorded as negatives:** a symmetric 3σ gate is catastrophic on
rising tolls (use the one-sided, dynamics-aware gate); confidence-as-soft-R does not beat
a hard cut, because zero-shot errors are gross false positives rather than graded noise.

**Venezuela 2026 scaffolding** was built the same day as the intended *blind* test — an
event after the model cutoff. It remains the only genuinely blind test available and is
still unrun.

---

## Phase 4 — joint_ie design, and a blocker taken seriously (7–8 Aug)

The boundary→joint_ie adapters were built and unit-tested: mentions, relation pairs, then
`joint_decode` end to end.

**Decision 4b, which blocked Phase A** (8 Aug): the beam modelled only nodes (mentions) and
edges (relations) — there was no record/instance concept, so events would bypass the beam
entirely and decode greedily. With RAMS in the warm-start, events were on the evaluated
path. Rather than proceed and quietly compare a beam arm against a partly-greedy arm, the
blocker was declared and the design extended (events as trigger node + role edges;
anchorless structures).

**Decision: every scaling point carries relation data** (8 Aug) at the pool's own 73/27
ratio, "removes a real confound" — otherwise volume and *whether the relation head was
warmed at all* would move together.

**Decision: drop SciERC on licensing** (8 Aug), not leakage. It contributed 265 of 37,237
relation records (0.19%) and alone forced every model card to "research use only". A
negligible data cost for a materially better license posture, taken because the models are
intended to be public.

**Decision: exclude DocRED** because Re-DocRED re-annotates the same documents — leakage,
checked rather than assumed (a 3,000-sentence `sentence_rex` sample had zero verbatim
overlap with Re-DocRED's 3.2M characters).

---

## Phase 5 — the bug week (8 Aug)

A dense run of defects, each found by trying to run the thing rather than by reading it.

- **Boundary layout omitted event-role queries** — `[V]` was missing from the extractive
  markers. Fixed, with a test asserting layout/marker parity for all five task types.
- **Evaluation aborted on any unalignable entity surface.** `error_policy` (malformed
  records) and `on_missing_surface` (alignment) are different knobs; the latter was never
  forwarded, so it sat at "raise" and killed runs at the first epoch-end eval.
- **FA2 raised on every inference call** — weights load fp32 by design and training wraps
  forward in autocast, but inference never did.
- **Blind test loaded boundary checkpoints as span**, dying on `config.max_width` *after*
  training had fully completed and saved.
- **Pushed models had no model card at all.**
- **NaN root cause** (8 Aug): single-variable matrix on one H100, 60 steps, deterministic —
  sdpa+bf16 goes non-finite at step 15 at 2.0 samples/s; **FA2+bf16 is clean at 22.0**.
  Either variable alone fixes it, so it is the interaction. Never DDP, never the data,
  never the head.

**Decision: `finalize_run.py`** — replay the post-training tail (blind test + card) against
an existing checkpoint, so a job whose tail crashes does not require retraining.

---

## Phase 6 — the curve runs (9–10 Aug)

**Events emitted nothing, and had never emitted anything** (9 Aug). Every event metric read
0.0000 at every threshold down to 0.01. The boundary engine skipped non-entity queries
assuming the record head would decode them — but the record head is **inert for events**
(an events schema produces no `record_metadata`, so `compile_record_specs` returns `{}`).
The models had learned trigger and role spans all along; nothing assembled them. Two
lessons recorded: a metric that cannot leave 0.0 is indistinguishable from a model that
learned nothing, and `metric_for_best` had been pointing at exactly such a metric.

**The blind test contained no events despite 73% event training data** (9 Aug). No event
corpus declared a `test:` key — and fixing that was not enough, because `_event_split`
filters on `Path(p).is_file()` and the slices had never been copied to the box. A
missing-path filter that drops silently is indistinguishable from "this corpus has no test
data".

**First boundary point**: 10K on the boundary head beat the span curve's 100K on arguments.

**The matched-threshold rule, first use** (9 Aug): warm starts *appeared* to regress with
more base data. They had not; the points were calibrated at different thresholds. A
cross-model warning was added to `model_card.py`.

**The curve completed, 12/12 arms, ~23h on one H100** (10 Aug). Event-argument F1, all
points threshold-matched at 0.3:

| | 10K | 40K | 100K | 137K |
|---|--:|--:|--:|--:|
| mmbert (cold base) | 0.010 | 0.039 | 0.079 | **0.098** |
| rams (warm start) | 0.177 | 0.191 | **0.202** | 0.192 |

**Finding, then same-day retraction.** It was written up as "warm-starting and base-scaling
saturate at different points". A support check run afterwards showed the mmbert arms
straddle the 9 Aug blind-test fix: 10k/40k/100k scored on 3,527 argument instances, 137K on
20,845. So the cold-base row is not a curve, and the cross-row comparison never shared a
test set. **Only the RAMS row survives** (support 2,016 throughout): it saturates from 100K,
and the dip is saturation not decline — one seed, ~5% relative.

The failure is the matched-threshold lesson one level up. Thresholds were checked because
they had already overturned a result; **support** was not, and it is the same class of
error. `compare_capabilities.py` now prints support beside every metric and flags a change.

**The matched-threshold rule, second use.** Re-DocRED looked erratic (relation 0.176 /
0.136 / 0.207 / 0.176). Its thresholds *alternate* 0.1/0.3/0.1/0.3; split by threshold,
both halves rise monotonically. Not noise — artefact. Two process fixes recorded:
`metric_sweep: true` is right for shipping one model and wrong for a curve, and
`test_metrics.json` does not record its threshold, so comparability cannot be checked from
the metrics alone.

All 12 arms are on HF (private), plus 26M of training logs pulled before the box was
destroyed.

---

## Phase 7 — real events (9–10 Aug)

### Turkiye–Syria 2023

**Decision: pre-register before running**, including predictions, because the outcome was
already known to the model driving the pipeline (Feb 2023 predates the cutoff). Ground
truth was sourced from one Al Jazeera tracker page sampled through the Wayback Machine,
one archive URL per point — **search-result summaries were rejected outright** after they
contradicted each other on dates (31,643 for 9 Feb *and* 17,134 for 10 Feb).

**Decision: truncate the series at 21 Feb.** The page froze at 41,000 and reported it into
April while the real toll reached 53,537 — staleness of the source, provable because Al
Jazeera itself published 44,218 on 24 Feb. Left in, a tracker that simply stopped updating
would have scored perfectly for six weeks.

**Result: a negative one.** Extraction read the trajectory nearly point-for-point;
`est_last_value` **beat** the EKF (0.136 vs 0.208); the 1999 Izmit toll of 17,500, quoted
in the article's history section, was tracked as a 2023 figure in every configuration; and
Syria was never recovered, because the association key was computed per *document*.

**A pre-registered prediction failed**, and the reason outlived the experiment:
range-normalized RMSE never noticed that 12 of 20 readings came from a different
earthquake, because 17,500 sits mid-range of a 1,014 → 41,000 trajectory. **A badly wrong
observation scored well.** Only carrying the baseline exposed it.

**Then attribution was solved** — by framing, not by machinery. Record extraction has an
inverted-U response to context volume: below ~750 characters the record head does not fire;
at 1,000–1,500 it emits exactly two records and binds **both** countries 16/16; above
~2,500 it fires normally but binds the wrong pairs. Asking the record head for the
`location` it already knew, at the right window, gave turkey 0.107 and syria 0.075.

**Correction recorded the same day**: an earlier claim that the encoder "physically cannot
see the whole document" past 512 tokens was **wrong** — DeBERTa-v3 uses relative attention
only, GLiNER2 does no chunking, and entities were demonstrably found at character 6,038.
The mechanism is degradation past the trained length, not truncation.

**And the benchmark cannot score the filter.** With clean attribution `last_value` hits
0.000 — necessarily, since the ground truth was read from the same sentence the extractor
reads. The baseline is an oracle by construction. Turkiye tests extraction and attribution;
it cannot test the tracker.

**`extract_long` supersedes the lead-window fix**: GLiNER2 already ships overlapping-chunk
long-document extraction, and at `chunk_size=200` it gets 16/16 on both countries over the
*whole* article, with no dependence on where a publication puts its historical section.

### Hurricane Helene 2024

**Decision: split the sources.** Ground truth from Wikipedia's per-state casualty table
(31 dated snapshots, 7 state streams + Total, with genuine downward revisions), feed from
AP wire prose — so `last_value` is no longer an oracle.

**Result: the Turkiye-solved configuration does not generalise.** It fragments one event
into **18 association keys** spanning every geographic granularity, and the event type
varies per article. Turkiye had two clean country-level places; Helene has a hierarchy.

**The required fix is the opposite of Turkiye's.** There the pipeline had to *split*; here
AP reports the national total and rarely breaks it down, so it must *pool*. The extracted
values in time order are the national total trajectory. **Neither "always split" nor
"always pool" is right — correct granularity is a property of what the source reports**, and
nothing infers it.

Scored as one national stream: EKF 2.042 vs `last_value` 2.125. The EKF edges a
non-oracle baseline for the first time, but both are worse than predicting a constant, so
the win is not claimed. Recall is the limit: 25 observations from 70 articles.

**Standing position: three real-world attempts, zero clean wins for the tracker.** Turkiye
could not test it; Helene cannot yet reach it.

---

## Phase 8 — the warm-start run (10 Aug)

**Diagnosis first**: `mmbert-137k` cannot do `[C]` record extraction at all — `None` at
every threshold down to 0.01, with `enable_records: true` and a real `record_decoder`
present. The cause is the mixture. It contains a corpus **named `text2json` whose 7,817
records all supervise `entities`, not structures**, so the record head was never taught the
task. The same audit explains the weak NER: those records were the *only* entity
supervision, 5.7% of a 137K mix.

**Decision: warm-start with replay at 70/30** (user's ratio, from experience). Replay is
sampled across both old task families at the pool's own 73/27 ratio, and the mixture is
pre-shuffled at the example level so every batch is a mixture rather than a block. NER is
split evenly between `pile_ner_def` and `nuner_full` rather than by pool size, because pool
size reflects how big a dataset someone published, not how useful it is.

**Decision: heterogeneous field types.** The casualty corpus gains a string `location`
beside its numeric roles, because the previous fine-tune saw only numeric fields and
collapsed to "emit a digit" — asked for a `location` it returns the number.

**Two false starts, recorded because the second correction reverses the first:**

1. The run trained correctly but at **4.6 samples/s against the curve's 22**, ETA 14h. It
   was killed after ~$6 rather than paying 4.6× an estimate.
2. **Profiling blamed document length.** It found a real effect — cost is superlinear in
   sequence length, and `text2json` again (30.6% over 4,000 chars, max 102,068) sits in
   the *replay* list. A 2,000-char cap gives 1.87× for no loss in record count.
3. **But that was a secondary factor.** The dominant cause was **FA2 silently not
   loading**. `transformers` was unpinned, a fresh box resolved 5.13 against a `kernels`
   pin matching 5.6, the ranges are disjoint, and the encoder fell back to sdpa — which is
   both ~11× slower and numerically fatal on bf16 ModernBERT. With FA2 restored: **56.9
   samples/s, zero non-finite losses, ~70 minutes**.

**The profiling was run on CPU, where FA2 does not exist**, so it was structurally incapable
of finding the real cause. A secondary factor was presented as the answer.

**Fixes**: `transformers` pinned alongside `kernels` with a note that the ranges move
together; `GLINER2_STRICT_ATTN=1` turns the first fallback into a load-time error so a run
that requires FA2 fails before spending GPU hours.

---

## Phase 9 — the record head finally learns (10 Aug, late)

The warm start was rerun as a two-arm A/B after the first attempt produced correct
multi-instance records with `location: None` every time.

**Diagnosis, confirmed.** The corpus emitted no `record_metadata`, so
`compile_record_specs` returned nothing and the record head was never supervised. Declaring
`mode=` in `build_multievent_corpus.py` fixed it: **`location` fills 6/9 (4/9 correct)**
against 0/9 in every prior configuration.

**`anchorless` learns nothing** (1/9 instances). The earlier evidence had been explicitly
discounted -- a model never trained on anchorless failing to decode it says nothing about
the mode -- so it was run rather than assumed. Training on it did not rescue it.

**The two arms explain each other.** `natural` costs relation -0.037; `anchorless` costs
-0.002 and is flat everywhere. One arm learned a task and displaced capacity; the other
learned nothing to displace anything with. That reframes the regression as a **price**, not
a defect.

**Two self-inflicted costs worth recording.** A loader change made earlier the same day was
verified by LOADING a checkpoint but never by running a forward -- the hub FA2 repo form is
CUDA-only and raised at first forward on CPU, which is worse than the fallback it replaced.
And the 2xH100 box ran ~40 minutes past the point every artifact was local, roughly $5-6,
because analysis was treated as the task and the machine as background. Second cost lapse
of the day; the first was launching a run whose throughput had never been sanity-checked.

## Phase 10 — the beam arm was never runnable (10 Aug, close)

A question about the mention-key format ("if there truly are duplicate spans, can we dedup
them?") turned out to be aimed at a real defect and at an imprecise description of it.

**The description was wrong first.** The note said the collision came from `role_name` being
`'head'`/`'tail'`. Relation field names are arbitrary and binding is positional, so that was
incidental. The defect is that the **relation type is dropped**: `spec["task_name"]` exists
at `model.py:1260` and `query_types` is built from `field_name` alone at `engine.py:516`.

**Dedup was the wrong fix, and the probe is why.** On `deaths_in {head,tail}` +
`injured_in {head,tail}`: 512 mentions, 271 unique keys, 241 colliding — and **0 of 241
collisions had matching logits**. Each relation type is its own schema group with its own
query embeddings, so each scores the span independently. These were duplicate *keys*, not
duplicate *mentions*; merging them would have scored one relation's role edges against
another relation's mention evidence. Running-but-wrong, which is worse than crashing.

**Two things the reproduction found that reasoning had not.** The raise is
*threshold-dependent* — no raise at 0.5, raises at 0.05 and 0.01 — so a schema looks fine
until the model gets confident. And `TypedEndpoints` had the same root cause: with bare role
names every relation declares `("head",)/("tail",)`, so endpoint typing was **vacuous across
relation types**. The plan for number-to-place attachment depends on exactly that constraint
discriminating, so dedup would have quietly forfeited the thing it was meant to enable.

**Fixed by qualifying the node type per query** (`qualified_query_type`, `models/base.py`) at
all three sites at once — `query_types`, `_query_type` for the pair endpoint keys, and the
`TypedEndpoints` construction — because a one-sided fix does not raise, it drops every edge
through the `keep_ids` filter and returns empty, which reads as "the model found nothing".
That is the §3 empty-layout failure a second time.

So "no raise" was refused as the acceptance test. Measured after the fix: 0 collisions,
**256/256 edges resolving to nodes**, 18 surviving threshold 0.05. The regression tests were
then checked by reverting the fix and confirming both fail.

**Consequence for the papers.** The beam arm has never run on a schema with two relation
types. Every number in the 12-arm curve is the greedy arm. The papers call the comparison
"unmeasured"; until today it was *unrunnable*.

## Phase 11 — Phase A finally runs, and answers a different question (10 Aug, close)

With the collision fixed, the beam arm ran for the first time: Re-DocRED, 96 relation types,
one trained model, eval-time `decode_mode` switch.

**The headline number is not the finding.** Joint beat greedy 0.1803 to 0.0740 on relation
F1 at threshold 0.5 — and §4b's own table already had greedy at 0.176 at threshold 0.3.
Threshold 0.5 is near greedy's worst operating point, so the "win" was mostly an operating
point. **Third time the matched-threshold rule has overturned a conclusion here**, which
is enough times that it stops being a lesson and becomes a standing rule.

**Beam width should be 1, and that is the real result.** Seven widths, monotone decline
(0.2406 at W=1 to 0.2058 at W=64), entity metrics byte-identical throughout. Widening drops
40 predictions of which 18 were correct — better precision, worse F1. The beam maximizes the
objective better as it widens (it even keeps greedy as a floor, so score is monotone), and
the objective is not F1. **A better search on a mis-specified objective is worse output.**
So the gain over greedy is the *formulation* — constrained joint selection — not the search,
and Phase A's "greedy vs beam" framing is mis-specified.

**Checking a remembered fact paid.** The recollection was that OneIE used beam width 3; the
paper says θ=10, the released package defaults to 5. But the instinct that OneIE used a
*small* beam was right, and the reason is structural — their β=2 caps branching at 2 per
step, so θ=10 is a wide search relative to their space. Worth recording that the correction
and the intuition were both useful: the number was wrong and the direction was right.

**Then the arm exposed the actual bug.** Joint recall barely moved across thresholds
(0.1498 → 0.1591) while greedy's moved 9× (0.0461 → 0.4134). Cause: `joint_decode` never
passed `decision_threshold`, so it sat at 0.5 while utilities were centered there — edge
selection ignored `--threshold` entirely. I had described that flatness as "threshold
insensitivity, a useful property" one message earlier. It was a plumbing bug.
`JOINT_IE_SCALING` had *predicted* this exact issue in its arm-comparability caveat and
called it "moot for a single-threshold eval". It was the dominant confound.

**A rejected design, recorded because the reasoning generalizes.** A OneIE-style β label cap
was considered and dropped: β is a pruner, and the joint arm sits at P=0.61 / R=0.15, so it
targets the axis already being won. Span-dimension caps already exist, and compute was never
binding. On events it would bite only on list roles, where the known failure is
under-generation. *Match the knob to the failure mode, not to the paper it came from.*

## Phase 12 — a summarizer, tested before it was built (10 Aug, close)

Proposal: run a purpose-built summarizer to split text into self-contained bullets so
number-to-place binding becomes local, with a verbatim-number guard against fabrication.

**Finding the real cases killed my own framing first.** I had been asserting Helene's hard
case was an intra-sentential aggregate ("120 in NC, 17 in TN, 227 total"). Searching the
feed found **5** multi-number casualty sentences and none of that shape — the example came
from a probe, not the corpus. What is actually there: a 140-mile distance, a 30-year career,
a town's 6,000 population, 30.5 centimetres of rain, the year 2004, and four deaths
belonging to **Hurricane Ivan**. The failure modes are non-casualty numbers and cross-event
leakage, not aggregate splitting.

**The premise test said no.** Raw text 3/5; free bullets 2/5 with 2 fabrications; extractive
bullets 3/5 with 0. Restructuring is neutral at best. The test also caught a flaw in itself
mid-run — the first version joined the bullets back into one string, rebuilding the very
ambiguity the split existed to remove.

**The guard and the summarizer are in direct tension**, which is the durable finding. The
summarizer's highest-value act is normalizing implicit prose into digits — "they died
together" → "2 people died" — and that is precisely what a verbatim guard rejects.
Constraining it to extractive-only resolves the conflict and costs real recall: a firefighter
whose death is never quantified becomes unreachable.

**And the result redirected the work.** Raw extraction's one error on that sample was a
*scope* error, not an attachment error — the national 225 filed under South Carolina. Second
independent signal pointing at aggregate scope.

**Also caught: I reported the wrong arm of my own experiment.** A `vector_state_test` run
looked like a new catastrophic result until the delta (+1.3379) exactly matched the number
the design doc had already recorded for the *isotropic* case. The default is isotropic; the
doc's table uses `--q-prop 0.15`. The doc was right and the run was mis-flagged.

## Phase 13 — the attachment blocker, mostly solved (10 Aug, close)

Number-to-place attachment had been the top open item for a day, and the fix turned out to
be neither a better extractor nor a bigger model.

**Counting first changed the target twice.** I proposed routing *unlocated* totals to
`__aggregate__` and argued it should go first because it had no bootstrap dependency. It
touches **4 of 106** observations. Then dumping the state streams showed what was actually
in them: contamination that is **always upward**, never once downward — Florida (truth 26)
receiving 300, North Carolina (truth 96) receiving 250. A one-directional error with a
2–10x separation is gateable; that is the whole reason this worked.

**Judge against a larger scope, and classify three ways.** Gating a stream against its own
running scale fails on its early history, where a toll legitimately jumps 6 → 25 faster than
any ratio tolerates. And a two-way keep/reroute split destroyed the national stream
(0.402 → 2.110), because North Carolina's **1400** is not a national total — it is not a
casualty count at all. Three outcomes were needed: keep, reroute, drop.

**On Helene: per-state 5.247 → 0.591, and the aggregate improved too.** Flat across ratios
1.5–2.5, so not a knife-edge. The control that made me believe it: removing the same 25
observations *at random* gives 4.427, so the gate is selecting rather than thinning a filter
into looking better.

**Then held-out testing took some of it back.** On Turkiye-Syria the gate as specified
**cannot run** — there is no `__aggregate__` stream to judge against. Generalizing the
reference to a global maximum makes it run: Syria, 65% contaminated with Turkey's tolls,
improves 3.7x; Turkey, already clean, **degrades 2.3x**, because the global max is dominated
by Turkey's own values and Turkey ends up judged against a reference it defines itself. It
rerouted 1,014 at t=12.5h — Turkey's true value at that moment.

The finding is sharper than the win: **the gate needs a declared scope hierarchy, not just a
magnitude.** Turkey's 41,000 filed under Syria and Turkey's 41,000 filed under Turkey are
identical to any ratio test. Helene has the hierarchy declared in `rollup.json`; Turkey does
not. Both numbers belong in the writeup — the Helene one alone is the misleading half.

## Phase 14 — the day measurement corrected four of my own claims (11 Aug)

No new capability shipped. What happened instead is that five things I had asserted were
checked, and four were wrong.

**"Recall is the limit: 25 observations from 70 articles."** That count came from the
superseded `tracked_lead` run. `extract_long` had already taken the same feed to 106
observations across 44 of 70 articles — **4.2x** — and the feed contains only ~86 sentences
carrying both casualty language and a digit, so the pipeline *over*-extracts. I had repeated
the stale figure in PIPELINES.md that morning and built a priority list on it.

**"North Carolina's 1400 is a non-casualty number."** It is **Hurricane Katrina's** toll
quoted inside a Helene article; the 250 is a Typhoon and the 230 is Milton. That relabel has
teeth: the scope gate removes them *because they are large*, not because they belong to
another event, so it silently keeps Bosnia's 16 and Mexico's 2.

**"0 false positives."** Measured on 83 Helene observations. On 250 gold training positives
the same rule false-rejects **3.6%**. Small-sample luck.

**"A model cannot referee a boundary it does not know."** The self-guide scores **82.5%**
against a 25% chance baseline, on a *harder* boundary than the one that defeated it.

**MHT was closed by an oracle rather than an argument.** Perfect association is worth
**+0.055**, and the gate already beats a perfect two-way assignment on two states because it
has a third option the oracle lacks: *drop*. A subsystem competing for a 9% residual is not
the next thing to build.

**What survived.** Type energies solve the unit-error half cleanly — 4/4 at 0/83 — and the
competing-type SET is the whole design: `quantity`, described as "a count of things that are
not people", is semantically adjacent rather than incompatible and takes false positives from
0% to 21.7%. *Compete against what a value cannot be, not what it resembles.* Cross-event
resisted all three signals tried, because the type is right there and only the event is wrong.

**Then the same discipline caught three scoping errors in the GIST work before any compute
was spent**: gold cannot be the guide (0.23% coverage); same-record rivals must never be
vetoed because gold is authoritative there (22.2% of material); and uniform rival sampling
returns noise, since a hard negative is a *near* neighbour and 12 draws from 17,131 types
never contain one.

**And the tests caught two silent failures in code I had written an hour earlier** — swapped
axes that vetoed nothing rather than erroring, and a shadowed `floor` local that disabled the
abstention check. Both are pinned by guard tests now. Neither would have raised.

Nothing is wired into `model.py`; the day's output is inputs, measurements and four
corrections.

## Phase 15 — the metrics were lying, twice (12–14 Aug)

A phase about the instrument rather than the model. Two defects in checkpoint selection and
one in scoring, all of which had been silently shaping results.

**`metric_for_best` fell back to `eval_loss` when its key was absent** (`c0ab89c`). The
fallback swapped both the quantity *and its direction*: a run configured to maximize an F1
maximized loss instead, selecting the worst checkpoint. This is what pinned MAVEN to epoch 1
of 10. It now raises. The sibling defect in `make_sweeping_compute_metrics` — a missing key
defaulting to 0.0, which scored every threshold identically and kept the first grid point —
was the real cause of 15 `[eval sweep]` lines reading 0.0000 (`3d21eba`).

**Fair evaluation moved to the reference tool's weights** (`d6debaf`), the paper's Eq. 6/7
rather than Eq. 5. Boundary errors now earn 0.5 TP, so the weights move F1 rather than only
P and R. See [[PAPER_0_FOUNDATION]] §8.

**Evaluation dropped `entity_descriptions` — and the first fix was a no-op.** Corpora that
name their types `e_0`/`e_1` and put the meaning in a parallel map were scored by asking the
model to find "e_0" with an empty description. The tell was a baseline that could not be
true: pristine `fastino/gliner2-base-v1` reading 0.1351 strict entity F1 on `pile_ner_def` at
recall 0.085.

The first fix (`bbacce6`) put the map under `schema["entities"]` as the values. That
type-checks and reads as correct, and changes nothing the model sees — those values are label
targets, while the prompt is built from `schema["entity_descriptions"]`. It was caught only
because a with/without control returned **identical numbers to four decimal places**. Real
fix is `7586411`; on 100 `pile_ner_def` val records, strict F1 0.0174 → **0.5381**.

Worse, the test shipped with `bbacce6` asserted the dict shape I had assumed, so it locked
the defect in. Replaced with one that pushes the schema through `processor._infer_from_json`
and asserts the description text reaches the prompt — *assert against the consumer, not
against your model of the consumer*.

**The blast radius was measured rather than assumed, and it is small.** Selection was never
affected: every config but the preservation one selects on `eval_loss`, which the trainer
computes from the forward pass. Blind-test reach is 49.3% for `mmbert-base` and ≤0.5%
everywhere else; the record-mode A/B stands at 0.2%, because `mix_natural.test.jsonl` is
empty and its 35.5% description share is a val-split property. No paper number needed redoing
— which is the point of measuring instead of assuming.

**The synthetic-corpus arm, and a trade measured in both directions.** Fine-tuning base-v1 on
a 5K Haiku-generated multi-task corpus produced large in-distribution gains — event strict
**0.0083 → 0.5467**, a schema the base model essentially cannot do — against a **−0.118
strict F1 (−22% relative)** loss on general-domain NER (0.5320 → 0.4136 on 6,016 records),
each arm at its own swept threshold, so the gap is not a threshold artifact. At a fixed 0.5
it reads −0.121; sweeping moves it by 0.003. The check was worth running because the
fine-tune's own sweep had picked 0.7 on synthetic val, so neither arm was at its optimum.

The FairEval decomposition says *what* was lost, and it is not the span machinery: boundary
errors went **down** (BES 3398→2706, BEL 1725→1272) while label errors rose (LE +1302,
LBE +579) and FN rose 8,977. The encoder and span head held; the labeling head reorganized
onto the synthetic type vocabulary. That is catastrophic forgetting of a label distribution,
not of a representation — which predicts a cheap fix (5–10% replay) rather than an expensive
one. Expected in direction, and now quantified.

**Also caught, before it cost a GPU run:** `uv pip install -e .` resolves transformers 5.13,
because the `transformers>=5.6,<5.7` pin lives in the `[local]` extra while
`kernels>=0.12,<0.13` is a core dependency. The two ranges are disjoint; kernels never hooks,
mmBERT falls back to sdpa, and bf16 goes non-finite around step 50 after running 11x slow,
with one `RuntimeWarning` as the only signal. This is documented in `pyproject.toml` *because
it already cost the 2026-08-10 run*, and it recurred anyway on a fresh box. Install
`.[local]`, and set `GLINER2_STRICT_ATTN=1` so a fallback raises.

## Recurring lessons

1. **Report the baseline every time.** Run B of Turkiye reads as a success at 0.208 without
   `est_last_value` beside it.
2. **A silent filter is worse than a loud failure.** Three separate incidents: `is_file()`
   dropping event test slices, a cached fetch failure making a transient 429 permanent, and
   an FA2 fallback that only warned.
3. **Check the threshold before reading a curve.** It has now changed a conclusion
   THREE times. Promoted from lesson to standing rule: no arm or curve comparison is
   readable until every arm sits at its own swept threshold.
4. **Stale state reads as current state.** A frozen tracker page, a cached failure, a
   leftover `FAILED_1` marker firing a waiter early.
5. **A misleadingly-named corpus cost two separate diagnoses.** `text2json` supervises
   entities and holds the longest documents in the mix.
6. **Profile where the phenomenon lives.** CPU profiling cannot find a GPU-kernel problem.
7. **Verify the operation you care about, not the one that is easy.** A checkpoint that
   LOADS is not a checkpoint that runs a forward; a metric that moves is not a metric on
   the same test set; a schema that is valid is not a schema that can decode.
8. **Terminate the box before analysing.** Twice in one day the machine outlived its
   usefulness because the interesting part was what came next.
9. **A better search on a mis-specified objective is worse output.** Beam width hurt F1
   monotonically while improving the score it was built to maximize.
10. **Match the knob to the failure mode, not to the paper it came from.** A OneIE β cap is
   a pruner; the arm that needed help was short on recall, not precision.
11. **Find the real failure cases before designing for them.** Two designs today were aimed
   at an example sentence that does not occur in the corpus.
12. **Check which arm you ran.** A "new" catastrophic result was the untuned default of an
   experiment whose tuned table was already written down.
13. **Count before designing.** Two fixes proposed as "do this first" were no-ops at 4/106
   and at 3.8% of the data. One dump of the actual values redirected a day of work.
14. **A control that removes the same amount at random.** Gating improved nRMSE 9x; random
   removal of the same count improved it 1.2x. Without that comparison the result was
   indistinguishable from thinning the filter until it stopped moving.
15. **Re-check the provenance of any number you are about to build on.** "25 observations"
   came from a superseded run and survived into three documents before anyone asked.
16. **Compete against what a value cannot be, not what it resembles.** A semantically
   adjacent competitor took false positives from 0% to 21.7%; physically incompatible ones
   left them at 0%.
17. **A measurement that contradicts your labels is usually right.** Twice the probe sorted
   cases my hand labels had mis-assigned, on both the "230"s and the "two"s.
18. **Assert against the consumer, not against your model of the consumer.** The test that
   shipped with `bbacce6` asserted the dict shape the fix produced, so it passed while the
   fix did nothing. The replacement pushes the schema through the processor and checks the
   text reaches the prompt.
19. **Identical output from a with/without control is a result, not a coincidence.** Two eval
   runs agreeing to four decimal places is what exposed a fix as a no-op. Run the control
   even when you are confident, and *especially* when the change is one you just wrote.
20. **A hazard documented in the source is not a hazard prevented.** The transformers/kernels
   pin drift is commented at length in `pyproject.toml` because it cost a GPU run — and it
   recurred on the next fresh box, because `-e .` does not install the extra that carries
   the pin. Prevention needs a check that runs, not a comment that explains.

## Phase 16 — the blind test was never blind (15 Aug)

Started as "separate the event loss signal", ended by invalidating the reference every
other number is quoted against. Both halves are worth keeping.

**The loss half.** The flat `task_loss_weights` dose sweep (0.5/1.0/2.0/4.0) was null on
every metric. The temptation was to conclude loss balance is not a lever here. Bucketing
the loss by task instead — `probe_task_losses.py`, reconciling to ~1e-7 with a residual of
exactly 0 — showed why it *had* to be null: `query_weights` reaches only start/end/pair,
**18.5%** of the loss, so `w=4` moved events from 6.6% to 10.6% of the gradient while
three quarters of it sat untouched. A null result from a lever that cannot reach is not
evidence about the hypothesis.

I got the first version of that measurement wrong in a way worth recording: I bucketed
four terms, found "events 1.6%, entities 17.2%, task-blind 76.4%", and reported it. The
76.4% was not task-blind, it was **unmeasured** — bucketing the other five terms moved
events to 6.6% and entities to **77.2%**. The correction changed the recommendation, not
just the decimals: the imbalance is entities dominating, not events starving.

**Regime beats dose.** Event positive fraction is 0.562 at convergence but **0.052** at
cold-start init. `pos_weight` fixes negative-dominated imbalance; at warm start that
imbalance is already gone, so `k>1` creates the opposite one. The mechanism belongs in the
cold-start rebuild, which is not where I was about to spend the GPU.

**The contamination half.** Chasing a structure-data gap led into
`convert_text2json.py`, which emits the flat key->value shape — `{"tournament_code":
"ROL-2024", "winner": "Sofia Petrova"}` — as *entities*. That is a record. 97% of the
corpus. Two consequences, both measured: the record head got **zero** supervision from a
corpus named text2json (`json_structures` was 0.0% of the cold-start gradient), and the
entity head learned **6,203** pseudo types, 731 of them appearing exactly once, from what
the mix audit calls its *only* entity supervision.

Then the splits. `SplitWriter._route` drew one random **per row**, so a document emitted
more than once scattered across train/val/test. text2json's val was **99.0%** contained in
its train. Aggregated over a whole config, **1,080 documents — 7.03% of the blind test —
were in train**. Every number anchored to the 137k reference was measured that way.

Fixes, in the order they mattered: group splits by document (and make it the **default**,
because opt-in meant 20 of 21 converter write sites silently didn't); gate a config's
aggregated splits with `check_leakage.py --config`; run the same gate inside
`train.py` before a single step, with test authoritative and never modified. Blind-test
contamination fell **2,942 -> 21** documents on the cold-start config, 299 -> 4 on
warmstart.

Two of my own fixes were wrong on the first try and caught only by re-measuring: the group
key wasn't normalized the way the checkers normalize (case/whitespace variants scattered
anyway), and `build_warmstart_mix.py` carved its val slice positionally, generating 27
contaminated documents itself.

**Lessons.**

21. **A null result from a lever you have not measured the reach of is not a result.**
    Measure what fraction of the objective a knob can touch before concluding the knob
    does nothing.
22. **"Unmeasured" and "not attributable" look identical in a partial decomposition.**
    Bucketing four of nine terms produced a confident, wrong story about where the
    gradient goes.
23. **Correctness invariants must be defaults, not options.** Grouped splits existed as an
    opt-in argument for one commit; 20 of 21 call sites didn't use it.
24. **Verify the fix with the same instrument that found the bug.** Both follow-up defects
    were invisible to inspection and obvious to a re-run of the checker.

### Same day, second half — the reach was the whole story, and then the control was

Re-running the null sweep with the reach fixed (`task_loss_weight_scope: all`, 94.3% of
the loss instead of 18.5%) gave **+0.013 event strict on both doses**, above the measured
floor. Entities went *up* rather than down, so it is the plan's "free win" branch, not the
predicted trade; the one consistent cost is **event_type −0.019** at the higher dose,
monotone in the dose. First positive result on this line, and it only appeared because the
null was diagnosed rather than believed.

Then a RAMS experiment — does an intermediate `mix_natural` stage help the event
downstream? — produced a wash (arguments span 0.005 across three arms) and, incidentally,
**the most consequential number of the day**. Its control was the *published* 137K recipe
re-run unchanged: it scored **0.2151** against the published **0.192**. +0.023 from a
re-run alone.

That single number retired a claim I had made two hours earlier from the same data. I had
read the 137K point (0.192, below 100K's 0.202) as "the head-init curve turns at 100K".
It does not turn. −0.010 is half of one run's variance. And since every point on that
curve is one seed with no measured floor, **no point-to-point difference on it is
interpretable** — including the 40K→100K rise I had been quoting as a trend.

What survives is what was never marginal: boundary beats span 3.5× at 10K (0.177 vs
0.050), and the *span* curve climbs 0.108 across its range, five times the variance. The
thesis — head-init scaling is a property of the span head, not of mmBERT — is strengthened,
because the boundary head turns out not to climb *at all* within measurement.

25. **A baseline you did not produce is not a baseline.** Three times today a delta
    dissolved once the control was run on the same code, the same data, the same day.
    The cost of the control is always less than the cost of the wrong conclusion.
26. **One seed is a measurement of one seed.** Every curve in this project has one point
    per configuration. That was fine while the effects were 3.5×; it is not fine now that
    they are 0.01, and the shape of a one-seed curve is not evidence of anything.

## Open

- **Clean re-baseline + `scope: all` arms running** as of 15 Aug on one 2xH100: two control
  seeds (new noise floor on the rebuilt `mix_natural`) then `evwide2`/`evwide4`, which test
  whether magnitude matters once the weight's reach is fixed at 94.3%.
- **Four converters still split row-wise** (`docee`, `docfee`, `cmnee`, `mendeley_ed`) — they
  do not use `SplitWriter`. 21 documents of residual contamination, removed by the trainer
  gate every run but not yet fixed at the source.
- **`data/scaling_joint/` val files are frozen from 8 Aug**, built from the pre-fix corpora.
  Rebuilding them would also rebuild the j10k/j40k/j100k slices the scaling curve rests on,
  so it is deferred rather than done.
- **Structures are never scored by the blind test** — `_schema_from_gold` builds no schema
  for `json_structures`, so structure-only records are skipped (35.1% of `mix_natural`'s val).
- Replay mix for the synthetic fine-tune — 5–10% of the original labeled data, to hold the
  general-NER label distribution while keeping the event gain. Predicted cheap by the
  Phase 15 error decomposition; unrun.
- `mmbert-base`'s blind-test entity row is understated (49.3% of its test set was scored
  without descriptions). No live claim depends on it; recompute if one ever does.
- Venezuela 2026 — the only genuinely blind test, still unrun.
- Helene needs administrative rollup + `extract_long` before it is a usable instrument.
- The 12 HF models carry `attn_implementation: flash_attention_2`, which silently falls
  back to sdpa: harmless for fp32 inference, a trap for bf16 fine-tuning from them.
- Phase B joint training, still gated on Phase A being positive. Phase A ran (Phase 11)
  but its headline is confounded by threshold; best-vs-best is the deciding run.
- `joint_beam_width` default is still 16; the measurement says 1.
- `RequiredRoles` fill-vs-reject trap, recorded in the registry and deferred.

---

## Phase 17 — two builds lost, and the reference set turned out to be a ghost (19–20 Aug)

Four corrections, three of them to claims made earlier in the same two days.

**MHT was priced wrong, in our favour.** The `+0.055` that rejected MHT came from a *two-way*
oracle — each observation goes to its own place or to the national total — which has no home
for a figure belonging to no scope in the event, and therefore scores Katrina's 1,400 as badly
as the shipped gate does. The tell had been sitting in the results and was read as a
curiosity: the gate *beats* the perfect oracle on two states, because it can *drop*. Track
birth/death **is** a null hypothesis, so with a reject option the ceiling moves to **+0.111
(18.8%)** — double.

**So we built the cheapest piece that delivers one, and it lost.** M5 track birth by
innovation gating, tracks advanced jointly in time order, no ground truth: best 0.608 against
the magnitude ratio's 0.591 — and the per-place number alone was the flattering half. With the
Total column restored it is 2.115 against 0.316, a 6.7× degradation of the one stream the
project calls honest. It had reproduced a bug `gate()`'s three-outcome design was written to
prevent. Two causes, both measured: judging a stream against its own track is **circular**
(removing the self-reference is worth more than every other knob), and the innovation is
uninformative about scope on a rising toll. The circularity is the same one that killed the
implied-maximum reference on Türkiye — two independent mechanisms, one cause. It is also the
first evidence *for* deferred assignment, which addresses it directly.

**The data-side route was built, trained, and beaten by a threshold.** `casualty_loc_muted`
withholds an interfering event's records while keeping its text. Two arms, four epochs, one
A10, ~$2.35. The suppression is real — precision up, recall down, 15 of 20 large Helene false
positives removed, ungated error 46.844 → 19.822. Then a **declared per-event plausibility
ceiling**, one threshold with no model behind it, recovered and exceeded the whole gain: at a
ceiling of 2,000 the *control* wins both ungated (5.853 vs 6.194) and gated (3.336 vs 3.729),
carrying 81 more observations. The ceiling removes only junk; muting removed real signal too.

**And it fixed the class it was not built for.** Chasing a 94,000 that had wrecked Tennessee
found Asheville's population — emitted as `dead` *and* `injured` *and* `missing` at confidence
1.0. None of the large values was a Helene toll: populations, **FEMA flood-insurance
policies** (129,933), wellness checks, power crews, troops, churches, and years read as tolls.
The sharpest is a 15,000 whose sentence exists to warn against that exact error. Both genuine
cross-event tolls survive muting *and* the ceiling. A correction inside the correction: troops
and crews were first filed as "counts of non-people" — they are people, just not casualties,
and that distinction is the point, because entity typing reaches insurance policies and stops
dead at power crews.

**The reference set behind all of it cannot be regenerated.** `tracked_rollup.json`, written
2026-08-10 and the source of every published Helene number, reproduces from no committed
state: the `--rollup` flag did not exist in any commit before the file was written, and the
rollup file was not in the tree either. Both were uncommitted working-tree state, committed
later in a form that does not reproduce it. Comparisons among the published figures stand —
one frozen artifact — but no new model can be placed on their scale, which is what blocked the
muting arm's pre-registered guard. `run_pipeline.py` now records its full invocation and a
`-dirty` git marker in every output. Second time provenance has stopped this line; Türkiye is
stalled the same way.

**What is actually open, unchanged by two builds:** cross-event contamination. Katrina's 1,400
and Maria's 3,000 survive every mechanism tried. A ceiling low enough to catch them is just
the magnitude gate again, rejecting a figure for being large rather than for belonging to
another storm.


---

## Phase 18 — the critical path turned out to be the extractor (20 Aug)

Three association mechanisms had to be built and lost before this was visible.

**A router proposal that finally builds a representation.** Take `min(start)..max(end)` over
an event's own trigger and arguments, embed that block, match against live filters. It fixes
the objection that sank clustering — it *produces* a representation instead of assuming one —
and it is per-event and local, which is the discourse attachment that proximity and type both
failed at. Verified on the hard case: in the Katrina passage the only named event found is
`Hurricane Katrina`, so the block is Katrina-local.

**It could not be run.** The span architecture emits a bag of triggers all sharing one role;
no threshold works, the Katrina block being either the bare name without its own 1,400 or a
sweep that swallows Helene. The boundary base yields nothing above threshold 0.3 on English
disaster copy and nonsense at 0.1.

**The cause, and a correction to a claim made an hour earlier.** A first pass read the model
card and said 68% of its event supervision was Chinese. Wrong, and the truth was worse: DocEE,
ChFinAnn and DocFEE are not events at all — `entities` + `classifications` — so counting them
flattered both sides. Corpora that actually bind arguments to triggers give **798 English rows
against 20,884 Chinese**, a 3.7% English share, with CASIE the entire English side and MAVEN
and Mendeley trigger-only. An argument F1 of 0.506 is a Chinese-only number.
*(0.506 is a RELAXED own-test figure — see Phase 19 for the category error this nearly caused, and for the rebuild's outcome.)*

**So the front end is being rebuilt** — cold start, 189,284 records, English trigger→argument
798 → ~39,800, Chinese kept because it is why the head works at all. The risk is declared up
front: 72% of the new English data is synthetic, on a line whose recurring failure is
in-domain-good and real-news-zero, so the gates are on AP prose.

**Method notes from the day.** The smoke measured 18.5 samples/s against an extrapolated 12.1,
so every cost estimate before it was 34% high — and `num_workers`, the obvious suspect for
28–81% utilisation, was refuted (18.4 at 0, 18.5 at 4); the idling is variable sequence length,
and memory sits at 10.6 GB of 40, so batch_size is the real untested lever. The gate harness
was validated by running it against the incumbent and confirming it *fails*, because a gate
that passes everything measures nothing. And rams finally got a val split, carved by document,
which found 101 duplicate rows in its test set alone.

## Phase 19 — the rebuild won every benchmark and appeared to lose the job (21–23 Aug; conclusion overturned in Phase 20)

The front end from Phase 18 trained cleanly and self-terminated: 35,484 steps, 6 epochs,
17.3 h, ~$35, eval loss falling every epoch 1.2672 → 0.8917 and still falling at the end.
Then scoring it produced three reversals in a row.

**Reversal 1: the harness had never swept anything.** `frontend_gates.py` set
`Schema().events(trigger_threshold=…, argument_threshold=…)` — values read only by the
*span* engine (`inference/runtime.py`). The boundary greedy decode this model runs gates on
the single global `extract(threshold=)` and never looks at them. So all five rows of a
five-point sweep ran at the default 0.5. Proven both directions: via the Schema, thresholds
0.999 and 0.01 return byte-identical output; via the global knob, the Katrina case is empty
at 0.05 and above and emits at 0.01.

The Phase 18 note above says the harness "was validated by running it against the incumbent
and confirming it *fails*". That validation is exactly why the defect survived a month. The
harness returned the expected answer, so nobody asked how. **A gate that fails the thing you
expect it to fail is not thereby verified.**

**Reversal 2: corrected, the incumbent passes the gate the rebuild fails.** Over the
pre-registered 0.1–0.5 range, on the same 60 Helene windows: the rebuild forms a trigger
plus ≥1 bound argument on **25.0%** of them, the incumbent on **65.0%**. The claim that
justified the rebuild — "the incumbent is ~0 at every threshold" — was true only at 0.5.
Its real curve is 0.0 / 0.0 / 8.3 / 20.0 / 65.0 across 0.5→0.1. The Phase 18 premise
survives (nothing usable at 0.3+, nonsense at 0.1); the strengthened version, which is what
was actually acted on, does not.

Corrected at all three sites that carried it — harness docstring, the config's
pre-registration block, and TODO — by dated correction rather than erasure.

**Reversal 3: on held-out corpora the rebuild beats the incumbent on every single head.**
Both scored by one command on one A100, same 11 files / 15,456 rows, threshold pinned 0.5
($1.83). entity +0.0158, relation +0.0593, classification +0.0160, structure +0.0096,
event_type +0.0155, event_trigger +0.0145, event_argument +0.0133, event +0.0043; fair
entity/trigger/argument +0.0278 / +0.0147 / +0.0570. Structure swept to the record head's
own thresholds (0.178 max object probability, so 0.5 is unreachable): 0.1184 vs 0.1132.

**So gates 1–2 FAIL and gates 3–4 PASS.** Every benchmark up, the target behaviour down.

> **This reading is WRONG — see Phase 20.** Gate 1 counts firings, not correct ones. The
> rebuild's bindings are right 67–100% of the time against the incumbent's 0–7.7%, so the
> mix change worked. The paragraph is kept because how it failed is the lesson.
This is the strongest case the programme has produced for pre-registering gates on the real
distribution: scored the normal way, this model is an unambiguous improvement and ships.

**A category error we came within one step of committing.** Gate 3 was written as
"event_trigger ≥ 0.710 and event_argument ≥ 0.506 (the incumbent's)". Those are **relaxed**
figures from the incumbent's *own* test set. The rebuild's strict argument F1 is 0.214 — set
against 0.506 that reads as a halving, when like-for-like it is a doubling (0.1046 vs
0.0913 strict on the shared split). Fixed in the harness and the config.

**Why the pre-registered remedy is declined.** The config fixed "downsample
`casualty_events` first" as the response to a gate-1 failure. Two measurements say it cannot
work. `_decode_events` returns a single-element list per event type and pools every trigger
and argument into it — its own docstring says the mention path "carries no instance
dimension" — so a passage naming two hurricanes returns `n_event_instances=1` with Helene's
246 and Katrina's 1,400 both bound as `dead` on the same event, at 0.1 and at 0.01. Gate 2
takes min..max over that one instance, so it rewards sparsity rather than binding. (Not
impossible: the incumbent does produce a local Katrina block at 0.01, and that passage
carries one Hurricane-typed event, not two extracted ones. Fragile and threshold-lucky,
which no corpus fixes.) And `casualty_events` carries 8 event types and no named identities,
so it holds no same-type discrimination to teach. **The next constraint is the instance
dimension — the record head — not another corpus.**

**A silent-truncation bug, caught by arithmetic.** Provisioning the eval box, the blind test
printed "scoring against 3 files" where a count verified an hour earlier said 11.
`_event_split` dropped any event file not already on disk — no fetch, no warning — while the
`corpora` path fetched from the Hub. On a fresh box that silently removed every event corpus,
i.e. the entire subject of gates 3–4, and reported the remainder as a completed blind test.
It could only ever appear on a box, because locally those files exist. Fixed to fetch, and to
name anything it still cannot resolve.

**Housekeeping with a real finding in it.** Eight run memories still read LIVE against zero
running instances. Reconciling them turned up the 4th cell of the real-vs-synth 2×2, measured
but never written down: the real+synthetic **mix preserved worst of all three arms, 0.3267,
−38.6% vs base** — worse than either arm alone, so mixing did not split the difference. It had
been cited second-hand as "−39%" and now has a primary source. Those JSONs lived only in
gitignored `out/`; they are committed now.

## Phase 20 — the gate was wrong, and the free measurement caught it (24 Aug)

Phase 19 closed with "eight benchmarks up, the target behaviour down" and a recommendation
built on it. Asked to defend that recommendation, it came apart in two steps.

**First, the recommendation only addressed one of the two failing gates.** Gate 1 asks for
a trigger plus ≥1 bound argument; one event instance satisfies it, so pooling is irrelevant
to it. Gate 2 is the one pooling breaks. "The next constraint is the instance dimension"
therefore answered gate 2 and left gate 1's regression unexplained — while the
pre-registered remedy it declined, downsampling the synthetic corpus, was aimed precisely
at gate 1.

**Second, and worse, gate 1's numbers do not mean what they were read to mean.** The two
models' curves cross: the rebuild leads at 0.5 and 0.4, ties at 0.3, and loses only as the
threshold opens up. The incumbent's 65% is entirely a low-threshold phenomenon, and gate 2
had already shown that its low-threshold bindings are nonsense. Gate 1 counts FORM.

So we measured the thing neither gate measures, free, on CPU: locate each window's gold
death toll by character offset, and count a `dead` argument as a hit when its span overlaps.

| threshold | rebuild fired → hit (prec) | incumbent fired → hit (prec) |
|---|---|---|
| 0.50 | 3 → 3 (100%) | 0 → 0 |
| 0.40 | 4 → 4 (100%) | 0 → 0 |
| 0.30 | 5 → 4 (80%) | 4 → 0 (0%) |
| 0.20 | 6 → 4 (67%) | 10 → 0 (0%) |
| 0.10 | 12 → 9 (75%) | 39 → 3 (7.7%) |
| 0.05 | 20 → 16 (80%) | 55 → 16 (29%) |

**The incumbent's gate-1 win is 39 firings carrying three correct death tolls.** It binds
`dead` to `"car Hurricane Helene"`, `"Mexico"`, `"Pacific coast"`, `"Carolinas"`. The
rebuild matches or beats its yield at every matched threshold and triples it at 0.1, off a
third of the firings.

**So the rebuild worked.** The mix change did what it was built to do; there is no
real-news regression to repair; and Phase 19's headline — which had already reached the
paper's abstract, TODO, MODEL_LINEAGE and the pushed model card — was wrong. Corrected in
all of them.

**The lesson is about the gate, not the model.** A form-only criterion scored
best-over-a-threshold-range rewards a model for firing indiscriminately. That is the same
error MODEL_LINEAGE's matched-threshold caution exists to prevent; we had applied it to A/B
arms and never to the gates those arms are judged by. It cost a day and would have cost a
training run. **Every form gate needs a correctness companion before it is used to compare
two models** — and the companion here was twenty minutes of local CPU, cheaper than any of
the reasoning built on top of the wrong number.

Three of this programme's own instruments have now been the error rather than the model:
the inert threshold sweep (Phase 19), the strict-versus-relaxed gate 3 (Phase 19), and gate
1 itself. In each case the instrument returned a plausible answer, which is why none was
audited until something else contradicted it.

**What still stands.** Gate 2 fails for both models; the decode emits one instance per
event type and pools every span into it, so the router still has no per-event input. Yield
is 15% at 0.1 — the rebuild is the better extractor and not yet a sufficient one. The next
constraint remains the instance dimension, now for a clean reason rather than a confused
one: not "the mix failed" but "the mix succeeded and this is what is left".

## Phase 21 — the process-noise question, asked properly and closed (24 Aug)

A review of the pipeline turned into a question about the filter: are the vectors
normalized? Three answers, and the third is the useful one.

**The shipped filter has no vectors.** `est_ekf_rise` is 1-D per role. It avoids needing
normalization by making every noise term relative — `R = (sig · max(ref,1))²`, growth
`q_rel · max(mu,1) · dt`, init `P = (0.4·max(z,1))²` — so a 5% error on 120 and on 12 are
treated alike. That is the right call and it is why the question does not arise there.

**The vector arm's default was the trap this file already documents.** `--q-prop`
defaulted to 0, i.e. isotropic. TODO item 10 called proportional Q "a precondition, not a
tuning knob", and Phase 12 records a session mistaking the isotropic default's +1.3379 for
a new catastrophic result. This session reproduced that number for the third time before
noticing. The default is now 0.15 and matches every recorded table; isotropic needs
`--q-prop 0`. **A documented trap that is also the default will keep being paid for.**

**The real question was whether Q should be diagonal at all.** Both options assert state
tolls accrue independently, and the aggregate row `H = [1..1]` is exactly where that bites
because `Var(sum) = Σᵢⱼ Pᵢⱼ`. If the aggregate constraint was rejected on a process model
that cannot represent what the aggregate observes, that is a reason to revisit a closed
question. Tested with `--q-rho`, uniform correlation with marginals preserved:

| ρ | 10% | 35% | 80% |
|---|--:|--:|--:|
| 0.0 | +0.174 (4/40) | +0.057 (10/40) | −0.004 (30/40) |
| 0.6 | +0.286 (5/40) | +0.096 (3/40) | +0.016 (4/40) |
| 0.9 | +0.308 (5/40) | +0.098 (5/40) | +0.024 (2/40) |

**Monotonically worse.** The rejection survives isotropic, proportional-diagonal and
correlated Q. And the *reason* is the finding: correlation degrades parts-only as well, so
these trajectories genuinely are near-independent. One storm drives all six states, yet the
dynamics are dominated by the reporting and revision process — NC's 123 → 102 → 96 is a
reclassification about NC — not by the physical event. Any future joint model of these
streams has to start from that.

Two caveats kept on the record: uniform ρ is the crude first model, where geographic
adjacency or a common-mode factor with per-state loadings would be more physical; and 40
trials over 31 real snapshots with a simulated reporting process is thin. Neither looks
likely to flip a result this monotone, but neither has been ruled out.

**The method note.** The candidate reason came from outside the work — a standard attitude
-estimation formulation, `Q_t = σ²W_tW_tᵀ`, where isotropic source noise becomes
non-diagonal through the Jacobian. A closed question reopened by an analogy from another
field, tested in minutes, and closed again with more support than it had before. Cheaper
than leaving it shut on one process model.
