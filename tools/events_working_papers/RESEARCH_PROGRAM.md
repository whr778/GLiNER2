# Global Inference over Boundary-Head Candidates: A Research Programme

**William Roe**¹ (whr778@gmail.com) and **Claude**² (noreply@anthropic.com)

¹ Project author and maintainer  ·  ² AI assistant (Anthropic, Claude Opus 5) — design, implementation, and drafting

*Programme map, revision of 2026-08-19. States the unifying thesis, the three papers it
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

**The greedy-vs-beam comparison — the actual question of the combinatorial arm — has
not been run.** Every number produced so far is the greedy arm.

**Structure supervision is not reaching the record head from most corpora.** The
cc_news and synthetic converters emit structures without the metadata the training path
needs, so records that appear to supply structure supervision supply none. This is a
data-design decision rather than a defect to patch, and it gates any future arm claiming
structure capability from those corpora.

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
training. The base-volume × architecture curve is complete on repaired data; the
decode arm is not. Design and results: `JOINT_IE_SCALING.md`.

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
how one head in this programme was misdiagnosed as broken across five measurements.

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
| `JOINT_IE_DESIGN_RECORD.md` | Paper 2's build record — decisions, wiring map, cost model, deferred Phase B |
| `EKF_MHT_DESIGN.md` | Paper 1 — the filter, the real-event defeat, and the association half rebuilt as a global decode (§7.7) |
| `EKF_MHT_BUILD_RECORD.md` | Paper 1's build record — attachment points, generator spec, blind-test protocol |
| `GATES.md` | every gate in the pipeline, its type, and whether it is on |
| `THIRD_EVENT_AEGEAN2020.md` | the third event: pre-registration, and the prediction it falsified |
| `BOUNDARY_ARCHITECTURE.md` | how the boundary head works end to end, per task |
| `COUNTING_LAYER.md` | why the span 19-instance cap is a dead end |
| `BOUNDARY_DECODE_AND_EKF.md` | verified decode map and where global inference attaches |
| `KALMAN_BEAM_SEARCH_EXPLORATION.md` | the beam ↔ filter origin analysis |
| `PROJECT_JOURNAL.md` | chronological record of decisions, including those later overturned |
| `TODO.md` | open defects and next tests, with the evidence for each |
