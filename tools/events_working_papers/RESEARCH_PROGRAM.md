# Research Program: Global Inference on Boundary-Head Candidates

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

The within-document scope is **both faces, not relations alone**: in the beam an event is a
*trigger node plus role edges* and a relation is a *plain edge*, so one typed-constraint
decode covers both ([[JOINT_IE_SCALING]] §3b). A win on only one face is a weaker but still
reportable result; the honest negative — global decoding helps relations and not events, or
the reverse — localizes where greedy per-query decoding actually costs you.

MHT's hypothesis beam and joint_ie's constraint beam are the **same machinery**; the boundary
head is the shared substrate — it removed the span 19-instance cap that made either impossible
([[COUNTING_LAYER]]).

### A sharpening under test (provisional, 2026-08-09)

The first measured boundary point ([[JOINT_IE_SCALING]] §4b) reaches RAMS argument F1 **0.182**
at a 10K base — above the *span* curve's ~100K point (0.158), on ~27% fewer event records. If
that holds at 40K/100K, the head-init deficit that motivated the scaling curve is largely an
artifact of the **span head's fixed-width enumeration**, not a data-volume law. That would make
"which head" a first-order claim for Paper 0 and reframe Paper 2's curve as
*architecture × decode* rather than *data × decode*. One point, greedy arm only, not yet
metric-selected — it does not change the thesis yet, and is recorded here so it is not
quietly forgotten if it survives.

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
| [[BOUNDARY_DECODE_AND_EKF]] | 1, 2 | verified boundary decode map + where global inference plugs in |
| [[EKF_MHT_DESIGN]] §1-13 | 1 | tracker design + decisions |
| [[EKF_MHT_DESIGN]] §14-20 | 1 | **results** (regime ablation, learned gate, normalization, model arm, missing probe, fine-tuned extractor) |
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
2. **Paper 1** — EKF line: fine-tuned extractor **done** (§20) → Venezuela double-blind → write.
3. **Paper 2** — joint_ie line: Phase A curve **running** (4 base sizes × RAMS/Re-DocRED warm
   starts, greedy vs beam) → greedy-vs-beam comparison → (Phase B) joint training → write.

## Graduation rule

A finding lives in its working paper (lab notebook) until **verified held-out**, then
graduates into the target paper's draft. Working papers stay as the design/decision record;
the papers carry only the verified, reproducible claims — including the honest negatives
(e.g. symmetric 3σ gate, confidence-as-soft-R) that scope the contributions. §4b above is
explicitly *pre*-graduation: one point, one arm, not metric-selected.
