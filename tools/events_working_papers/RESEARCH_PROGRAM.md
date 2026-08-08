# Research Program: Global Inference on Boundary-Head Candidates

Status: program map (one page). Date: 2026-08-07. The unifying thesis + the working-paper →
paper mapping. Design docs: [[EKF_MHT_DESIGN]], [[JOINT_IE_SCALING]],
[[BOUNDARY_DECODE_AND_EKF]], [[COUNTING_LAYER]], [[KALMAN_BEAM_SEARCH_EXPLORATION]].

## The unifying thesis

GLiNER2's **boundary head** emits candidate scores per chunk (spans, mention/pair logits,
contextual `candidate_states`), then a **greedy per-chunk decode** selects a record set. That
greedy decode is fine for sparse, single-document extraction — and **insufficient for dense
or evolving events**. The program's claim:

> **Global structured inference over the boundary candidate scores — not the greedy per-chunk
> decode — is what dense document-level and beyond-document events require.**

Two instantiations of the *same* top-K hypothesis inference at different scopes, both over the
shared `candidate_scores → JointProblem` contract:

| scope | mechanism | the "beam" is | target task |
|---|---|---|---|
| **within a document** (combinatorial) | joint_ie global decode + typed constraints | top-K constraint-consistent assignments | dense relations (Re-DocRED) |
| **across documents / streaming** (temporal) | EKF/MHT tracker (Kalman bank + pruning) | top-K hypotheses over time (MHT) | evolving events (disaster streams, Venezuela) |

MHT's hypothesis beam and joint_ie's constraint beam are the **same machinery**; the boundary
head is the shared substrate — it removed the span 19-instance cap that made either impossible
([[COUNTING_LAYER]]).

## The three papers

- **Paper 0 — Foundation.** GLiNER2 + the boundary head + head-init / multi-corpus training —
  the substrate both build on. Draft: [[PAPER_0_FOUNDATION]] — "Schema-Driven Information
  Extraction Beyond the Sentence". **Finish first** (no new experiments needed).
- **Paper 1 — Real-time events (temporal).** EKF/MHT streaming tracker on the boundary head:
  censored measurement model, learned/union gate, text→obs normalization, held-out synthetic +
  the Venezuela 2026 double-blind. **Most mature — ships next.**
- **Paper 2 — Traditional events (combinatorial).** joint_ie global decode wired to the
  boundary head: head-init scaling curve × decode arm on Re-DocRED, then structured joint
  training. **Nascent — follows.**

Papers 1 & 2 open with the same framing paragraph (above) and cross-cite; a later extended /
journal version may merge them into the single "global inference on boundary candidates"
statement.

## Working paper → paper map

| working paper | feeds | role |
|---|---|---|
| [[COUNTING_LAYER]] | 0, 1, 2 | why the span 19-cap is a dead end; boundary removes it |
| [[BOUNDARY_DECODE_AND_EKF]] | 1, 2 | verified boundary decode map + where global inference plugs in |
| [[EKF_MHT_DESIGN]] §1-13 | 1 | tracker design + decisions |
| [[EKF_MHT_DESIGN]] §14-19 | 1 | **results** (regime ablation, learned gate, normalization, model arm, missing probe) — graduate to Paper 1's results |
| [[JOINT_IE_SCALING]] | 2 | boundary wiring + scaling-curve experiment design |
| [[KALMAN_BEAM_SEARCH_EXPLORATION]] | 1 (+ framing) | the beam↔filter origin analysis |
| `mmbert-head-init-finding` (memory) | 0, 2 | head-init scaling evidence |

## Sequencing (by maturity; billing/effort aware)

1. **Paper 0** — finish the existing draft (it's the substrate; no new runs).
2. **Paper 1** — EKF line: casualty fine-tune (in flight) → Venezuela double-blind → write.
   `EKF_MHT_DESIGN` §14-19 are already the results skeleton.
3. **Paper 2** — joint_ie line: wire (Phase A, decode-only) → scaling runs → greedy-vs-beam
   curve → (Phase B) joint training → write.

## Graduation rule

A finding lives in its working paper (lab notebook) until **verified held-out**, then
graduates into the target paper's draft. Working papers stay as the design/decision record;
the papers carry only the verified, reproducible claims — including the honest negatives
(e.g. symmetric 3σ gate, confidence-as-soft-R) that scope the contributions.
