# Global Inference over Boundary-Head Candidates: A Research Programme

**William Roe**¹ (whr778@gmail.com) and **Claude**² (noreply@anthropic.com)

¹ Project author and maintainer  ·  ² AI assistant (Anthropic, Claude Opus 5) — design, implementation, and drafting

*Programme map, revision of 2026-09-08. States the unifying thesis, the three papers it
decomposes into, what is established, and what is still open. Numbers quoted here are
summaries; each one's primary record is the working paper cited beside it.*

---

## 1. The thesis

GLiNER2's **boundary head** emits candidate scores per chunk — span boundaries,
mention and pair logits, and contextual candidate states — and a **greedy per-chunk
decode** then selects a record set from them. That decode is adequate for sparse,
single-document extraction. The programme's claim is that it is not adequate for
anything denser:

> **Dense document-level and beyond-document extraction require global structured
> inference over the boundary candidate scores, not the greedy per-chunk decode.**

The claim has two instantiations. They are the *same* top-K hypothesis inference at
different scopes, over one shared `candidate_scores → JointProblem` contract:

| scope | mechanism | what the beam ranges over | target task |
|---|---|---|---|
| **within a document** (combinatorial) | global decode under typed constraints | top-K constraint-consistent assignments | events (RAMS) **and** relations (Re-DocRED) |
| **across documents** (temporal) | EKF/MHT tracker — Kalman bank plus pruning | top-K hypotheses over time | evolving events in a document stream |

Within a document, the two faces are one mechanism: in the beam an event is a *trigger
node plus role edges* and a relation is a *plain edge*, so a single typed-constraint
decode covers both. A win on only one face is a weaker but still reportable result, and
the honest negative — that global decoding helps relations and not events, or the
reverse — localises where greedy per-query decoding actually costs you.

Across documents the row is two questions, not one: **track** an evolving quantity, and
**diarise** observations into the right stream. The distinction matters because the two
halves fail independently, and the second can silently destroy the first.

The boundary head is the shared substrate for both. It removed the span architecture's
19-instances-per-type cap, which had made either mechanism impossible to express.

## 2. What is established

**The span head, not the encoder, was the bottleneck.** Holding fresh heads fixed and
swapping the encoder (mmBERT ↔ DeBERTa-v3) moves RAMS argument F1 barely at all
(0.050 vs 0.042). Holding the encoder fixed and swapping fresh → IE-pretrained heads
moves it roughly elevenfold (0.042 → 0.462). The deficit is head initialisation, not
context window. *(Paper 0, §10.5.)*

**Head warming has a measurable data threshold.** Warming mmBERT's heads on a
structure/argument corpus before RAMS fine-tuning does nothing at 10K, lifts arguments
about 2.3× at 40K, and keeps climbing to 100K with no plateau. The knee is between 10K
and 40K. *(Paper 0, §10.7.)*

**Windowing, not the global decoder, recovers document-level arguments** on a
short-context model: 0.086 → 0.144 argument-strict F1 from matching the eval window to
the trained window, against which the OneIE-style beam is neutral (0.144 → 0.137).
Reported as a negative result for the decoder. *(Paper 0, §10.2.)*

**Replay protects a warm start, and record extraction transfers without record
supervision.** Warm-starting the 137K joint base on real-plus-synthetic data with 30%
exact replay held the original structure capability (0.1119 → 0.1060 on the base's own
test set) while nearly tripling on the new distribution (0.0755 → 0.2179). The new
corpora supplied *no* record-head supervision at all, so the gain is transfer from the
span representations the record head reads its field fillers out of.
*(`JOINT_IE_SCALING.md` §0b–0c.)*

**An uncalibrated operating point cost more than any training intervention.**
*(2026-08-27.)* The stage-0 relevance gate ran its whole life at threshold 0.5. Its softmax
is saturated and it needs 0.998 to sit at the stated recall bar; choosing the threshold on
validation and scoring the blind test once moves overall accuracy 0.719 → 0.847 and the
worst class 0.444 → 0.903, fixing 108 rows against 39 broken (exact McNemar
*p* = 1.1×10⁻⁸). Two GPU fine-tuning runs and a four-way auxiliary label had been spent on
that class; at each model's own validation threshold the rebuilt gate and its predecessor are
indistinguishable (18/18, *p* = 1.0000). This is the §5 rule "quote a curve only with the
operating point it was read at" recurring as a several-hundred-dollar lesson rather than a
misdiagnosis. *(`GATES.md`; `EKF_MHT_DESIGN.md` §5.1.)*

**Single-run variance on these metrics is ±0.02 or worse.** A control re-run of a
published RAMS recipe scored +0.023 above it from the re-run alone. Any curve claim
needs at least two seeds per point before it is quoted, and several earlier readings of
this programme's own curves did not meet that bar.

## 3. What is not established

**The central question is now asked and answered — and the specified design was the wrong
answer.** *(Updated 2026-08-25.)* The design specified association as gate → Hungarian
assignment → top-K hypotheses → track birth/death, and for a long time what shipped was hard
assignment on an observable string key: no hypothesis enumeration, no deferred decision, no
track birth or death. Every real-event failure recorded below is a failure of that
placeholder.

What replaces it is **not** the specified design. A **global Viterbi decode over three
states — own place, aggregate, reject** — improves every event we have at one setting:
Helene 29.3 → 20.7 (−29.4%), Türkiye–Syria 11,581.5 → 10,695.5 (−7.6%), Aegean 2020
74.4 → 15.7 (−78.8%). Two properties carry it, and a hypothesis tree is neither: the
decision must be **global**, because a greedy rule commits per observation and one large
figure admitted early poisons a stream's scale for everything after; and it must be able to
**reject**, because assignment headroom is measured at **zero** and the entire residual is
the null hypothesis. Hungarian assignment optimises the half that is worth nothing.
*(`EKF_MHT_DESIGN.md` §7.7.)*

**On real news the tracker loses to a trivial baseline.** On the Türkiye–Syria 2023
earthquake, pre-registered, `est_last_value` beats the EKF (0.208 vs 0.136), a 1999
death toll quoted in an article's history section is tracked as a 2023 figure, and one
of the two affected countries is never recovered at all. Attribution, not filtering and
not extraction, is the bottleneck. *(`EKF_MHT_DESIGN.md` §4.)*

**The greedy-vs-beam comparison — the actual question of the combinatorial arm — HAS
NOW BEEN RUN, and the thesis is not supported by it.** *(2026-09-07, corrected
2026-09-08.)* One checkpoint (`eb16-rebuild-tr`), 18,786-record blind test,
`boundary_head.decode_mode` as the only variable, strict micro F1:

> **SUPERSEDED 2026-09-15.** These figures are of uncertain provenance and are NOT at threshold 0.5 — the incumbent's model card says `Decision threshold: 0.3`. Measured at the validation-selected 0.2: event_argument strict 0.0991 / relaxed 0.5884, event_trigger 0.5984, event_type 0.8650. See EVENT_ARGUMENT_DIAGNOSIS.md §4c.

| head | greedy | joint | Δ |
|---|--:|--:|--:|
| entity | 0.5695 | 0.5670 | −0.0026 |
| event_type | 0.7545 | 0.7545 | 0.0000 |
| event_argument | 0.1178 | 0.1143 | −0.0034 |
| relation | 0.1037 | 0.0994 | −0.0043 |
| structure | 0.1640 | 0.1582 | −0.0057 |
| event_trigger | 0.6051 | 0.5987 | −0.0064 |
| event | 0.4081 | 0.4004 | −0.0077 |

**Every head is inside the ±0.02 floor, and the joint arm costs ~2.5× the wall clock**
(8.5 min against 22). Beam width is not the lever: a 16× sweep (4 / 16 / 64) moves
structure by 0.0018, and *narrower* is marginally better, the opposite of a
search-capacity story.

**THE STRUCTURE ROW ABOVE IS A CORRECTION, and the size of it is the point.** As first
measured, structure read **0.1208 / 0.0754, a −0.0454 deficit** that was the one result
outside the floor and the whole of the negative verdict. It was an artefact of a
never-connected parameter: `field_dtypes_list` was declared, threaded and consumed
correctly, and **no caller ever passed it**, so every `dtype: str` field compiled
`ZERO_OR_MORE`. Cardinality selects the joint beam's utility (`logit - absent` against
bare `logit`) and its exclusivity slot, so **the beam's entire scalar machinery — decision
B of `JOINT_IE_DESIGN_RECORD` — had never engaged for a structure field.** Greedy looked
almost right anyway because it re-derived the dtype at format time; joint had no such
rescue. Connected, greedy gains +0.0432 and **joint gains +0.0828**, and 87% of the
deficit disappears. The arm that was supposedly losing was the arm running with its
machinery switched off.

**The verdict does not change; its content does.** "Joint decoding costs structure" is
withdrawn. What stands is stronger and duller: **joint decoding matches greedy on all
seven heads and costs 2.5× the wall clock.** A null at parity is a cleaner negative than a
loss, because it cannot be explained away as a defect — and this one nearly was the other
way round.

Three scopes on that negative, all load-bearing:

1. **It tests decoding-*with* the beam, not training-*for* it.** `JOINT_IE_DESIGN_RECORD`
   §7 (Phase B — beam in the loss) remains unrun and is the only version of the thesis
   this does not touch. It is also now more interesting, not less: the beam has only ever
   been measured with its scalar constraints disconnected.
2. **A separate mechanism was tested first and also came back neutral.** `--global-decode`
   is the cross-window event *merge* (`assemble_events_global` operates on results already
   decoded), not inference over candidate scores. Neutral at the trained window, slightly
   negative when fragmented.
3. **Mechanism, not model — measured, on two independent lineages.** The pre-fix deficit
   reproduced on `joint-boundary-mmbert-137k-clean` (−0.0529) as well as on the control,
   so it never was a property of one checkpoint. A third pair
   (`casualty-multilingual-eb16tr`) is **unreadable** and was declared so by a rule written
   before the numbers existed: its greedy structure is 0.0070, a sixth of the floor, and
   its other heads sit at 0.0801 entity / 0.0209 event_type against the base's 0.5695 /
> **SUPERSEDED 2026-09-15.** These figures are of uncertain provenance and are NOT at threshold 0.5 — the incumbent's model card says `Decision threshold: 0.3`. Measured at the validation-selected 0.2: event_argument strict 0.0991 / relaxed 0.5884, event_trigger 0.5984, event_type 0.8650. See EVENT_ARGUMENT_DIAGNOSIS.md §4c.
   0.7545 — catastrophic forgetting, not a decode result.

**Two of this project's own instruments had to be repaired before that table could be
read**, and each hid the next. The scorer dropped list-shaped structure output in silence
(48% of the apparent collapse). Then the first re-baseline measured a fix that never ran,
because the schema carrier was dropped one layer above and **nothing in the log said
whether the fix had executed** — so execution was inferred from the numbers it was meant to
move. Runs now print `[records] compiled N field spec(s): X scalar ...` before scoring, and
the third box was gated on that line rather than on hope.

**Structure supervision now reaches the record head from every corpus on disk — this
entry is a CORRECTION of the one it replaces.** *(2026-09-08.)* A structure whose schema
declares no `record_metadata` cannot be decoded by the record head, and the absence is
valid and silent: no error, no warning, and the rows still count as supervision in every
composition print. 60,948 rows across 13 models were in that state. The corpora were
repaired in place, and the *producer* was fixed separately — `synthetic/validate.py` had
gone on emitting structures without metadata, so every newly generated corpus reproduced
the defect. **Audited across all of `data/`: zero corpora whose structures lack
metadata.** `audit_corpora.py` carries a `STRUCT` check to keep it that way.

What remains is one level deeper and is a *decision*, not a defect: corpora declare only
mode and anchor, never per-field **cardinality**, so a `dtype: str` field is still trained
with list BCE rather than the scalar softmax over candidates plus `ABSENT`. Changing that
moves every future model's structure numbers and needs an A/B first. *(`TODO.md`.)*

## 4. The three papers

**Paper 0 — Foundation.** *Schema-Driven Information Extraction Beyond the Sentence.*
GLiNER2 plus the boundary head plus head-initialisation and multi-corpus training: the
substrate the other two build on. Complete; no new experiments required.

**Paper 1 — Real-time events (temporal).** The EKF/MHT line: censored measurement model,
learned gate, text-to-observation normalisation, held-out synthetic validation, then a
genuinely blind real event. Most mature on the tracking half. Blocked on the diarisation
half, which is the honest statement of where it stands. Design and results:
`EKF_MHT_DESIGN.md`.

**Paper 2 — Traditional events (combinatorial).** Global decode wired to the boundary
head, measured on both RAMS (events) and Re-DocRED (relations), then structured joint
training. The base-volume × architecture curve is complete on repaired data, **and as of
2026-09-08 so is the decode arm** — its result is a null at parity (§3), which is the
honest negative the paper reports rather than the win it was built to find. Phase B (beam
in the loss) remains unrun and is the only untouched version of the thesis. Design and
results: `JOINT_IE_SCALING.md`.

Papers 1 and 2 share the framing in §1 and cross-cite; a later extended version may
merge them into the single global-inference statement.

## 5. Method rules adopted after being learned the hard way

**Verify splits before every run.** A per-row random draw scattered copies of the same
document across train, validation and test in 45 corpora. It is silent — nothing crashes
and no metric looks wrong. Within-split overlap is now gated automatically before
training, and Paper 0 §7.1 reports the audit and its effect on that paper's results
rather than quietly restating the numbers.

**Never compare across test sets.** Two of this programme's retracted findings were
cross-row comparisons on differently-composed blind tests. Support counts must match, or
the comparison is not one.

**Always report the trivial baseline beside the model.** A pre-registered EKF prediction
was scored a success by a range-normalised metric that was blind to contamination landing
mid-range; `est_last_value` alongside it would have caught this immediately.

**Quote a curve only with the operating point it was read at.** A head whose decode
threshold is never calibrated can read exactly zero while working correctly — which is
how one head in this programme was misdiagnosed as broken across five measurements. It
recurred in 2026-08-27 on the stage-0 gate, where the default threshold cost more accuracy
than two fine-tuning runs recovered, and where a model comparison reported as significant
(*p* = 0.0091) vanished entirely (18/18, *p* = 1.0000) once each model was read at its own
threshold. **A comparison between two models at a shared arbitrary threshold measures the
gap between two operating points, not between two models.**

**A stored verdict is valid only for the model it was measured on.** An ablation concluding
that translation does not repair a multilingual gate was correct for the model it ran on —
which reads the language and over-admits — and exactly wrong for a later model that cannot
read the language at all, where translation moved AUC 0.4733 → 0.8359. Re-run a cited
verdict before applying it to a model that was not in it.

**Graduation rule.** A finding lives in its working paper until verified held-out, then
graduates into the target paper. The working papers stay as the design and decision
record; the papers carry only verified, reproducible claims — including the honest
negatives that scope the contributions.

## 6. Source documents

The working papers are the primary record; this map is a summary of them.

| document | role |
|---|---|
| `PAPER_0_FOUNDATION.md` | Paper 0 draft — substrate, head-init finding, data-integrity audit |
| `JOINT_IE_SCALING.md` | Paper 2 — joint decoding, scaling curve, replay, the three silent defects |
| `JOINT_IE_DESIGN_RECORD.md` | Paper 2's build record — decisions, wiring map, cost model |
| `EVENT_LINE.md` | **line 3** — full event support: the named incumbent, what counts as proving out, and the risk register including the EKF bet |
| `EVENT_ARGUMENT_DIAGNOSIS.md` | why `event_argument` reads 0.118 — binding, not extraction; 64.2% of gold instances share a type in-document and the mention path pools them |
| `PHASE_B_PLAN.md` | the beam **in the loss** — the only untested version of the thesis, with pre-registered bars and a throughput probe that gates the spend |
| `EKF_MHT_DESIGN.md` | Paper 1 — the filter, the real-event defeat, and the association half rebuilt as a global decode (§7.7) |
| `EKF_MHT_BUILD_RECORD.md` | Paper 1's build record — attachment points, generator spec, blind-test protocol |
| `GATES.md` | every gate in the pipeline, its type, and whether it is on |
| `THIRD_EVENT_AEGEAN2020.md` | the third event: pre-registration, and the prediction it falsified |
| `BOUNDARY_ARCHITECTURE.md` | how the boundary head works end to end, per task |
| `COUNTING_LAYER.md` | why the span 19-instance cap is a dead end |
| `BOUNDARY_DECODE_AND_EKF.md` | verified decode map and where global inference attaches |
| `RESEARCH_PROGRAM.md sec 1` | the beam ↔ filter origin analysis |
| `PROJECT_HISTORY.md` | chronological record of decisions, including those later overturned |
| `TODO.md` | open defects and next tests, with the evidence for each |
| `PIPELINES.md` | as-built vs as-designed maps, and the divergence table with M1–M5 status |
