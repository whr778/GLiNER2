# Research Program: Global Inference on Boundary-Head Candidates

> **⚠ 2026-08-18: every joint_ie / 137k number in this document predates a data repair
> and is superseded.** 45 corpora shipped overlapping train/val/test, and the scaling
> configs additionally paired regenerated train files against frozen val slices
> (252 train-in-val, 22 val-in-test). All were repaired, the slices rebuilt, and the
> four scaling points now gate CLEAN. The curve is being re-run from scratch — see
> `JOINT_IE_SCALING.md` and [[lambda-137k-curve-restart]]. Numbers below are kept as
> the record of what was measured and believed at the time; do not compare new results
> against them.

Status: program map (one page). Date: 2026-08-09 (supersedes 2026-08-07). The unifying
thesis, the three papers under their real names, and the working-paper → paper map.

## The unifying thesis

GLiNER2's **boundary head** emits candidate scores per chunk (spans, mention/pair logits,
contextual `candidate_states`), then a **greedy per-chunk decode** selects a record set. That
greedy decode is fine for sparse, single-document extraction — and **insufficient for dense
or evolving structure**. The program's claim:

> **Global structured inference over the boundary candidate scores — not the greedy per-chunk
> decode — is what dense document-level and beyond-document events require.**

Two instantiations of the *same* top-K hypothesis inference at different scopes, both over the
shared `candidate_scores → JointProblem` contract:

| scope | mechanism | the "beam" is | target task |
|---|---|---|---|
| **within a document** (combinatorial) | joint_ie global decode + typed constraints | top-K constraint-consistent assignments | **events (RAMS) _and_ relations (Re-DocRED)** |
| **across documents / streaming** (temporal) | EKF/MHT tracker (Kalman bank + pruning) | top-K hypotheses over time (MHT) | evolving events (disaster streams, Venezuela) |

The temporal row is two questions, not one: **track** an evolving quantity, and **diarize**
observations into the right stream. The tracker is built and validated on synthetic data;
the diarizer is **not built** — association currently ships as hard assignment on a string
key, with no hypothesis enumeration ([[EKF_MHT_DESIGN]] §1a). Every real-event failure so
far has been in the second half, which means the programme's central question remains
unasked rather than answered.

The within-document scope is **both faces, not relations alone**: in the beam an event is a
*trigger node plus role edges* and a relation is a *plain edge*, so one typed-constraint
decode covers both ([[JOINT_IE_SCALING]] §3b). A win on only one face is a weaker but still
reportable result; the honest negative — global decoding helps relations and not events, or
the reverse — localizes where greedy per-query decoding actually costs you.

MHT's hypothesis beam and joint_ie's constraint beam are the **same machinery**; the boundary
head is the shared substrate — it removed the span 19-instance cap that made either impossible
([[COUNTING_LAYER]]).

### A sharpening, now CONFIRMED (2026-08-10)

Partly. The full 12-arm curve completed 2026-08-10, all points threshold-matched at 0.3.
The *boundary* warm start reaches RAMS argument F1 **0.177 at 10K** and then barely moves:
0.191 / 0.202 / 0.192 through 137K — a valid within-row result (support 2,016 throughout).

> **2026-08-15 — "barely moves" is now measured, not inferred.** A control re-ran the
> published 137K recipe (same base, same 15 epochs) and scored **0.2151** against the
> published **0.192** — **+0.023 from a re-run alone** ([[lambda-rams-warmstart-run]]).
> Single-run variance is therefore ≥±0.02, and the entire boundary spread
> (0.177–0.215) is **one run's noise**. "Barely moves" understates it: the boundary
> curve does not measurably move at all. That *strengthens* the span-head reading
> below, and it means the 0.177-vs-0.158 comparison flagged in the next paragraph is
> marginal on variance grounds too, independent of the support question.

The comparison **against the span curve's 0.158 is not yet verified**: that number comes
from a different experiment and its blind-test support has not been checked against this
one. Given that a support mismatch has already invalidated the cold-base row of this very
curve, the boundary-beats-span claim must be re-derived on a shared test set before it is
used in Paper 0.

So the head-init deficit that motivated the scaling curve is largely an artifact of the
**span head's fixed-width enumeration**, not a data-volume law about mmBERT. "Which head"
is a first-order claim for Paper 0, and Paper 2's curve is *architecture × decode* rather
than *data × decode*.

**A second finding was retracted the same day.** "The cold base is still climbing at 137K
while the warm start is flat" compared rows evaluated on DIFFERENT blind tests: the mmbert
arms straddle the 9 Aug fix that added the missing event `test:` keys, so 10k/40k/100k were
scored on 3,527 argument instances and 137K on 20,845. The cold-base row is not a curve and
the cross-row comparison never shared a test set. Only the RAMS row survives (support 2,016
throughout): it saturates from 100K. Greedy arm only — the beam arm remains unmeasured.

## The three papers

- **Paper 0 — Foundation.** Draft: [[PAPER_0_FOUNDATION]] — *"Schema-Driven Information
  Extraction Beyond the Sentence: Event Extraction, Multilingual Training, and Document-Level
  Global Decoding for GLiNER2"*. GLiNER2 + the boundary head + head-init / multi-corpus
  training — the substrate both other papers build on. Complete draft (Abstract → §13
  References); **finish first, no new experiments required**.
- **Paper 1 — Real-time events (temporal).** Design: [[EKF_MHT_DESIGN]] — *"EKF/MHT Event
  Tracking on the Boundary Head — Design & Decisions"*. Censored measurement model,
  learned/union gate, text→observation normalization, held-out synthetic, then the Venezuela
  2026 double-blind. **Most mature — ships next.** §14-20 are already the results skeleton;
  §20 closes the extraction gap the §19 probe predicted.
- **Paper 2 — Traditional events (combinatorial).** Design: [[JOINT_IE_SCALING]] — *"joint_ie ×
  Head-Init Scaling on the Boundary Head — Design"*. joint_ie global decode wired to the
  boundary head: base-volume curve × decode arm on **both** RAMS (events) and Re-DocRED
  (relations), then structured joint training (Phase B). **In flight** — Phase A curve running.

Papers 1 & 2 open with the same framing paragraph (above) and cross-cite; a later extended /
journal version may merge them into the single "global inference on boundary candidates"
statement.

## Working paper → paper map

**Active — feed a paper:**

| working paper | feeds | role |
|---|---|---|
| [[COUNTING_LAYER]] | 0, 1, 2 | why the span 19-cap is a dead end; boundary removes it |
| [[BOUNDARY_ARCHITECTURE]] | 0, 1, 2 | **the substrate reference** — how the boundary head works end to end, and how NER / structures / relations / events each flow through training, evaluation and extraction |
| [[PROJECT_JOURNAL]] | all | chronological record of decisions, and of the ones later overturned |
| [[TODO]] | all | open defects and next tests, with the evidence for each |
| [[BOUNDARY_DECODE_AND_EKF]] | 1, 2 | verified boundary decode map + where global inference plugs in |
| [[EKF_MHT_DESIGN]] §1-13 | 1 | tracker design + decisions |
| [[EKF_MHT_DESIGN]] §14-20 | 1 | **results** (regime ablation, learned gate, normalization, model arm, missing probe, fine-tuned extractor) |
| [[EKF_MHT_DESIGN]] §21 | 1 | **first real event** (Turkiye–Syria 2023): pre-registered, negative, attribution is the bottleneck |
| [[JOINT_IE_SCALING]] §1-3c | 2 | thesis + boundary wiring (events, relations, structures in one beam) |
| [[JOINT_IE_SCALING]] §4, §4b | 2 | experiment design + **first measured point** |
| [[KALMAN_BEAM_SEARCH_EXPLORATION]] | 1 (+ framing) | the beam↔filter origin analysis |
| [[SCALING_CURVE_EXPERIMENT]] | 0, 2 | the *span* head-init curve spec — the baseline Paper 2 is measured against |
| [[HEAD_INIT_DATA_SCALE]] | 0, 2 | the prior estimate the curve tested (reasoned bracket) |
| [[DOCUMENT_EXTRACTION_PLAN]] | 0 §9, 2 | OneIE-style global decode over windowed candidates |
| `mmbert-head-init-finding` (memory) | 0, 2 | head-init scaling evidence (span arm: arg 0.050/0.115/0.158) |

**Reference / implementation record — not paper inputs:**

| working paper | role |
|---|---|
| [[EVENT_LOSS_PLAN]] | event-loss separation from `structure_loss` (implemented) |
| [[EVALUATION_PLAN]] | per-language blind-test methodology (feeds Paper 0 §8 indirectly) |
| [[FASTINO_GLINER2_TRAINING]] | how `fastino/gliner2-base-v1` was trained; why it warms our event heads |
| [[RE_DIFFERENCES]] | relation extraction: GLiNER vs GLiNER2 |
| [[CORE_CHANGES]], [[MAIN_MERGE_CONFLICT_MAP]] | branch/merge archaeology — historical, not cited |
| [[RECOMMENDATIONS]], [[DEFINITIONS]], [[EVALUATION]], [[INSTRUCTIONS]] | early scratch notes; superseded, kept for provenance |

## Sequencing (by maturity; billing/effort aware)

1. **Paper 0** — finish the existing draft (it's the substrate; no new runs). If §4b survives,
   add the head-architecture claim to §10.
2. **Paper 1** — EKF line: fine-tuned extractor **done** (§20); first real event **run and
   lost** ([[EKF_MHT_DESIGN]] §21, Turkiye–Syria 2023). The tracker is beaten by
   `est_last_value` on real news (0.208 vs 0.136) and a 1999 death toll is tracked as a
   2023 one. **Attribution, not filtering, is the bottleneck** — so the next step is an
   attribution mechanism (the §10/MHT crux), not more extractor fine-tuning. Venezuela
   remains the blind test; it is no longer the *only* thing standing between here and a
   paper.
3. **Paper 2** — joint_ie line: Phase A curve **COMPLETE** (12/12 arms, ~23h on one H100,
   all on HF). The base-volume × architecture result is in hand; the **greedy-vs-beam
   comparison is still unmeasured**, and that is the actual Phase A question — every number
   so far is the greedy arm. Then (Phase B) joint training → write.

## Graduation rule

A finding lives in its working paper (lab notebook) until **verified held-out**, then
graduates into the target paper's draft. Working papers stay as the design/decision record;
the papers carry only the verified, reproducible claims — including the honest negatives
(e.g. symmetric 3σ gate, confidence-as-soft-R) that scope the contributions. §4b above is
explicitly *pre*-graduation: one point, one arm, not metric-selected.
