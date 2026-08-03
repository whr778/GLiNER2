# Exploration: Extended Kalman Filter + Beam Search for Document-Level Event Assembly

Status: exploratory research note (no code). Date: 2026-08-03.
Scope: whether an Extended Kalman Filter (EKF) could work *in conjunction with*
the beam search in `gliner2/inference/global_decode.py`. Includes a literature
scan and two related decoding papers read alongside (OneEE; PosEKE-GPT2).

---

## 1. The idea being explored

A vague but interesting instinct: pair an **Extended Kalman Filter** with our
**beam search**. No papers appear to explore this in information extraction (IE).
This note tests whether the pairing is (a) sound, (b) novel, and (c) useful for
GLiNER2 — and where it is instead elegant over-machinery.

Bottom line up front: **the pairing is not novel in general — it is canonical in
target tracking, where it is called Multiple Hypothesis Tracking (MHT). It *is*
essentially unexplored in IE.** Whether it helps GLiNER2 depends entirely on
whether the problem has *real continuous dynamics*. Intra-document event assembly
(what our beam does today) mostly does not, so an EKF there collapses to plain
recursive evidence fusion. The framing where it genuinely earns its keep is
**streaming / temporal event tracking**, where an event's state actually evolves.

---

## 2. What our "beam search" actually is

`gliner2/inference/global_decode.py` is **not** autoregressive token decoding. It
is **model-free post-processing** that assembles one document-level event set from
the per-window results of the long-doc path (`batch_extract_long` ->
`merge_chunk_results`). Overlapping windows re-detect the same event and split its
arguments; the decoder reconnects them. Two layers:

- `_greedy_assemble` — cluster mentions across windows by **trigger IoU**
  (`GlobalDecodeConfig.trigger_iou`), union their arguments.
- `beam_decode` — refine under global constraints: drop events below
  `min_trigger_conf`, then choose which argument **edges** `(event, role, entity)`
  to keep to maximize `sum(confidence) - conflict_penalty` for every span reused
  across kept edges, subject to `single_filler_roles` cardinality. Beam state is
  `(score, kept-edge-index tuple, span-use counts, filled single-roles)`;
  `beam_width` defaults to 8. Triggers pass through unchanged.

Key properties:
- It is a **weighted set-selection / constraint-satisfaction** beam over a *static*
  candidate graph, not a sequence decoder.
- Its trade-offs are **hand-set heuristics** (`trigger_iou`, `conflict_penalty`,
  `min_trigger_conf`, `single_filler_roles`) — not learned, not calibrated.
- It runs **only in the windowed path** (DeBERTa-v3 `large-v1`, 512-token cap).
  The mmBERT-8192 path fits whole documents in one pass and **skips global decode
  entirely** (see `tools/train/config/mmbert-base-wikievents.yaml` header). This
  matters for the whole idea (Section 6).

---

## 3. Why EKF + beam is canonical: it is Multiple Hypothesis Tracking

In target tracking (radar/sonar/robotics), when there is **data-association
ambiguity** — which measurement belongs to which track — the hypothesis count
grows exponentially over time. The standard remedy:

1. maintain a **Kalman/EKF per hypothesis** (continuous state + covariance),
2. keep the **top-K hypotheses by likelihood and prune the rest.**

That top-K pruning over a bank of Kalman filters **is a beam search.** This is
**Multiple Hypothesis Tracking** (Reid, 1979) and its relatives: Gaussian Sum
Filter, JPDA, IMM, Rao-Blackwellized particle filter. The literature scan
(Section 5) confirms MHT is "a Kalman filter-based algorithm incorporated with
data association ... maintaining tracking trees, gating, and pruning," and that
its pruning shares the structure of beam search.

There is also a clean **theoretical bridge**:

- Beam search generalizes the **Viterbi** algorithm to state spaces too large to
  enumerate (dynamic programming without full Markov tractability).
- The **Kalman filter is the continuous-state analogue of the HMM forward pass**
  (linear-Gaussian SSM <-> HMM).
- Therefore "beam + EKF" = approximate inference in a **switching (hybrid
  discrete-continuous) state-space model**: the *discrete* part (which event a
  detection belongs to, how many events) is searched by beam; the *continuous*
  part (an evidence/confidence state per event) is tracked by the EKF. This is
  exactly a **switching Kalman filter / Gaussian-sum filter**.

So the instinct is structurally correct. The open question is purely whether IE
supplies the ingredient that makes a Kalman filter more than recursive averaging:
**genuine dynamics.**

---

## 4. The mapping onto event assembly

Scan windows left-to-right along a long document, treating window index as the
"time" axis:

| Tracking (MHT) | Our event assembly |
|---|---|
| Targets | Document-level events |
| Time steps (scans) | Overlapping windows |
| Measurements | Per-window detected triggers/arguments + confidences |
| Data association | "Is this detection the same event already assembled?" (today: trigger-IoU clustering in `_greedy_assemble`) |
| Per-track state estimate | The event's accumulated argument set + a continuous evidence/confidence state |
| Kalman gain | The accept/fuse trade-off currently hard-coded as `conflict_penalty` / `min_trigger_conf` |
| Hypothesis pruning (top-K) | `beam_decode` (`beam_width = 8`) |

The appeal is concrete: the EKF would replace **hand-tuned `GlobalDecodeConfig`
weights** with a probabilistic model where those trade-offs fall out of estimated
covariances, and it would give **principled uncertainty propagation across
windows** — which our beam lacks entirely (it sums point confidences with no
notion of *compounding* uncertainty as evidence accrues).

---

## 5. Literature scan (2026-08-03)

**(a) EKF + beam already exists — in tracking, under other names.**
- MHT is Kalman-based data association with tree pruning; hypothesis count grows
  exponentially and is controlled by pruning (HOMHT vs TOMHT).
  [MHT + Ensemble Kalman, MDPI Sensors 2019](https://www.mdpi.com/1424-8220/19/14/3118) ·
  [PMC mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679329/) ·
  [Target Tracking Using Kalman-Filter-Based Algorithms (IOP 2021)](https://iopscience.iop.org/article/10.1088/1742-6596/2078/1/012020/pdf)
- Modern neural MOT still fuses Kalman filtering with probabilistic data
  association: [PKF: Probabilistic Data Association Kalman Filter for Multi-Object
  Tracking (2024)](https://arxiv.org/html/2411.06378).
- Seminal reference: D. B. Reid, "An Algorithm for Tracking Multiple Targets,"
  IEEE T-AC, 1979 (MHT). Classic families: JPDA, IMM, Gaussian Sum Filter.

**(b) Kalman × event/information extraction in NLP — essentially absent.**
The IE-focused search surfaced only standard EE surveys and explicitly concluded
the results "do not specifically discuss the use of Kalman filters in these
contexts." This supports the claim that the IE crossover is genuinely unexplored.
- [Event Extraction in LLMs: A Holistic Survey (2025)](https://arxiv.org/html/2512.19537v1) ·
  [Overview of Event Extraction and Its Applications (2021)](https://arxiv.org/pdf/2111.03212)

**(c) Beam <-> Viterbi <-> Kalman/HMM theoretical link.**
Beam search extends Viterbi when Markov/independence assumptions fail; the Kalman
filter corresponds to the HMM forward pass for linear-Gaussian systems. (General
NLP-decoding background:
[Beam search in NLP decoding](https://www.analyticsvidhya.com/blog/2025/01/beam-search-in-nlp-decoding/).)

**(d) Neural-Kalman / deep state-space models exist — but not for IE decoding.**
Active line, mostly time-series forecasting and LM efficiency, not event-structure
assembly:
- Structured Inference Networks for Nonlinear State Space Models (Krishnan et al.):
  [code](https://github.com/clinicalml/structuredinference) (Deep Kalman Filters lineage).
- [Latent-KalmanNet: Learned Kalman Filtering from High-Dimensional Signals (2023)](https://arxiv.org/pdf/2304.07827)
- [Recurrent Neural Filters: Independent Bayesian Filtering Steps (2019)](https://arxiv.org/pdf/1901.08096)
- [Incorporating Transformer/LSTM into Kalman Filter with EM (2021)](https://arxiv.org/pdf/2105.00250)
- [Kalman Linear Attention: Parallel Bayesian Filtering for Language Modelling & State Tracking (2026)](https://arxiv.org/pdf/2602.10743) ·
  [KOSS: Kalman-Optimal Selective State Spaces (2025)](https://arxiv.org/html/2512.16723v1)

Takeaway: the *machinery* (MHT, GSF, neural Kalman) is mature; the *application to
IE event decoding* is the white space.

---

## 6. Critical assessment — where it earns its keep, where it does not

A Kalman filter is worth its cost when there is a **continuous dynamical state with
real dynamics** (position/velocity that evolves). Applied honestly to our beam:

**Against, for the intra-document assembly beam:**
1. **No natural dynamics.** Events do not "move" across windows. The state would
   be an artificial evidence-embedding; the process model `F` is ~identity+noise.
   Strip the dynamics and the EKF collapses to **recursive Bayesian fusion** — a
   running, uncertainty-weighted average. Useful, but a fraction of the EKF, and
   it trades `GlobalDecodeConfig` heuristics for `Q`/`R` covariance heuristics.
2. **Gaussian belief is a stretch.** Confidences are bounded [0,1]; embeddings are
   high-dim. EKF linearization can be poor or diverge. A particle filter or
   logit-space Gaussians may be needed — more machinery.
3. **The hard part is discrete.** Association + constraint satisfaction is what the
   beam already handles; the EKF only upgrades the *continuous fusion* sub-problem,
   which is not our measured bottleneck.
4. **We are retiring the substrate.** The mmBERT-8192 single-pass path skips global
   decode. If single-pass wins the current A/B (Section 7), there is no
   cross-window sequence for a filter to run over — the EKF-over-windows idea
   applies mainly to the *legacy* short-context path we are trying to eliminate.

**For — where it becomes compelling (real dynamics):**
1. **Streaming / temporal event tracking.** A stream of documents about an
   *evolving* situation (developing news, an entity's state over quarters). Now the
   event's argument set genuinely changes over real time; an EKF tracking that
   evolving state does real work (uncertainty propagation), and beam/MHT resolves
   "same event vs update vs new event." This is a legitimately novel IE framing,
   not an overfit — and the closest thing to the "missing papers."
2. **Lightweight calibration/smoothing.** Even in the static case, treat per-window
   logits as noisy measurements of a latent true score and Kalman-**smooth** them
   before the beam consumes them. Low-risk; a pre-processing layer, not a
   re-architecture.

---

## 7. Relation to current work and two decoding papers read alongside

**Current experiment (memory `lambda-combined-experiment`).** We are validating
whether a long-context, single-pass mmBERT event model matches/beats the windowed
`large-v1` + beam path. If single-pass wins, that is direct evidence the beam is
redundant — and the EKF-over-windows idea is moot for that path. If single-pass
*underperforms*, the beam's cross-window constraints were doing real work, and a
principled MHT/EKF replacement becomes worth exploring.

**OneEE (arXiv 2209.02693, "One-Stage Framework for Fast Overlapping and Nested
Event Extraction").** Discriminative, single-pass, **search-free** joint decoding
via a word-word relation grid (S-*/R-* tags), adaptive event fusion, distance-aware
role scoring; parallel decode of all event types; SOTA on FewFC / Genia11 / Genia13
with large speedups. Removes the beam *within a pass* by table-filling — but is
closed-ontology and needs long context to cover a document. Fits our mmBERT
direction; its event-fusion + distance-aware role scoring are targeted borrows for
our weak **argument head**.

**PosEKE-GPT2 (Nature Sci. Rep. s41598-025-23093-w, 2025).** Generative EE: GPT-2
emits "structured canonical text" via **task-based autoregressive decoding** (event
type -> trigger -> arguments) with position expansion (long financial docs) and a
knowledge-augmentation module (prompt-injected external knowledge, attention
weighted). DuEE-Fin F1 90.61, FewFC 88.85. It "replaces" the assembly beam by
*generating* assembled structure directly — but swaps GLiNER2's discriminative,
zero-shot, calibrated, fast core for a slower, closed-schema, hallucination-prone
autoregressive decoder that typically **reintroduces beam search at the token
level**. Notably it cites both OneEE and **ONEIE** (Lin et al. 2020, "global graph
optimization ... via beam search decoding") — the direct ancestor of our
`global_decode.py`. Worth borrowing: **knowledge-augmented conditioning** for the
argument head (GLiNER2 already has the `entity_descriptions` / schema-prompt hook).

**Three decoding paradigms on one axis** (how each treats the beam):
- ONEIE: discriminative + **global beam assembly** (our status quo).
- OneEE: discriminative, **single-pass, search-free** (removes beam within a pass).
- PosEKE-GPT2: **generative** (removes assembly beam, reintroduces token beam).

The EKF idea is orthogonal to all three: it is not a *within-pass* decoder but a
*cross-observation* estimator. It only has a home where observations arrive in a
sequence with association ambiguity — windows (being retired) or, better, a
temporal document stream.

---

## 8. Open questions (exploration only)

1. What is the *state vector*? Candidates: per-event evidence embedding, running
   argument-slot confidences, trigger-salience score. Does any have real dynamics,
   or only accumulation?
2. Where do `Q` / `R` come from — learned from data, or hand-set (in which case,
   is it simpler than the current heuristics)?
3. Is the right object a full EKF, a **Gaussian Sum Filter** (bank of KFs = beam of
   hypotheses), or a **Rao-Blackwellized particle filter** (discrete association
   sampled, continuous state Kalman-marginalized)?
4. Does the **streaming/temporal** framing (Section 6) justify a standalone
   research thread independent of GLiNER2's intra-document decoder?
5. Cheapest falsifiable probe: a Kalman **smoothing/calibration** layer on
   per-window logits feeding the existing beam — measure whether calibrated inputs
   improve assembly F1 at all before touching the beam's structure.

Recommendation: park EKF-over-windows until the current single-pass A/B reports.
If single-pass wins, redirect the idea to **streaming/temporal event tracking**,
where the dynamics are real and the MHT/EKF machinery earns its cost.

---

## 9. References

- Reid, D. B. "An Algorithm for Tracking Multiple Targets." IEEE T-AC, 1979. (MHT.)
- Lin, Y. et al. "A Joint Neural Model for Information Extraction with Global
  Features" (ONEIE). ACL 2020. (Beam-search global graph decoding; our beam's ancestor.)
- Cao, H. et al. "OneEE: A One-Stage Framework for Fast Overlapping and Nested
  Event Extraction." COLING 2022. arXiv:2209.02693.
- An, T. et al. "An improved GPT2-based joint event extraction method with position
  expansion and knowledge augmentation" (PosEKE-GPT2). Sci. Rep. 2025,
  s41598-025-23093-w.
- Krishnan, R. et al. "Deep Kalman Filters" / "Structured Inference Networks for
  Nonlinear State Space Models." https://github.com/clinicalml/structuredinference
- MHT + Ensemble Kalman: MDPI Sensors 19(14):3118, 2019.
- PKF: Probabilistic Data Association Kalman Filter for MOT, arXiv:2411.06378, 2024.
- Latent-KalmanNet, arXiv:2304.07827, 2023.
- Kalman Linear Attention, arXiv:2602.10743, 2026.
</content>
</invoke>
