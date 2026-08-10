# Project Journal — Global Inference on Boundary-Head Candidates

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

## Recurring lessons

1. **Report the baseline every time.** Run B of Turkiye reads as a success at 0.208 without
   `est_last_value` beside it.
2. **A silent filter is worse than a loud failure.** Three separate incidents: `is_file()`
   dropping event test slices, a cached fetch failure making a transient 429 permanent, and
   an FA2 fallback that only warned.
3. **Check the threshold before reading a curve.** It changed a conclusion twice.
4. **Stale state reads as current state.** A frozen tracker page, a cached failure, a
   leftover `FAILED_1` marker firing a waiter early.
5. **A misleadingly-named corpus cost two separate diagnoses.** `text2json` supervises
   entities and holds the longest documents in the mix.
6. **Profile where the phenomenon lives.** CPU profiling cannot find a GPU-kernel problem.

## Open

- Venezuela 2026 — the only genuinely blind test, still unrun.
- Helene needs administrative rollup + `extract_long` before it is a usable instrument.
- The 12 HF models carry `attn_implementation: flash_attention_2`, which silently falls
  back to sdpa: harmless for fp32 inference, a trap for bf16 fine-tuning from them.
- Phase B joint training, still gated on Phase A being positive.
- `RequiredRoles` fill-vs-reject trap, recorded in the registry and deferred.
