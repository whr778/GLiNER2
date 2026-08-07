# EKF/MHT Event Tracking on the Boundary Head — Design & Decisions

Status: design consolidation (no code yet). Date: 2026-08-07.
Companion to [[BOUNDARY_DECODE_AND_EKF.md]] (the verified code map),
[[COUNTING_LAYER.md]] (why the span head is a dead end), and
[[KALMAN_BEAM_SEARCH_EXPLORATION.md]] (the original EKF-vs-beam analysis).

## 1. Thesis / target

Research target: **Document-Level and Beyond-Document-Level Events.** The span
architecture caps at 19 instances/type/doc (dead end for dense doc-level, impossible
for beyond-doc). The **boundary** head removes the cap (DETR-style set prediction).
On top of it, an **EKF/MHT tracker** handles *beyond-document* events — a real-world
situation reported across a document stream, whose state evolves.

Scoped claim (must stay scoped): *for streaming, quantitatively-evolving events,
EKF/MHT tracking beats recursive fusion / the heuristic merge* — NOT "EKF helps event
extraction" in general.

## 2. Where it lives (from the verified map)

- Base decode = the boundary model's own **joint IE** decode (records + relations),
  live via `BoundaryExtractor`. NOT the `joint_ie` module (dormant, span-bound).
- The seam = the **cross-chunk / cross-document association** step
  (`merge_chunk_results`, currently a shallow dedupe + the span-era heuristic beam).
- Tracking state material = the boundary **`candidate_states [B,Q,C,H]`**. Plumbing
  gap: they die at per-chunk decode and must be carried forward to the seam.

## 3. The tracker design

**Granularity:** situation/process level (real dynamics), not just event-instance
(static → weak dynamics → EKF collapses to fusion).

**State vector `x` per track:** embedding `e` (pooled `candidate_states`), continuous
quantitative arguments `{v_r}` (normalized: counts/amounts/dates/magnitudes), salience
`s`; covariance `P`.

**Dynamics `f`, process noise `Q`:** slow-drift embedding; monotonic / random-walk /
parametric-process for `{v_r}`; decay for `s` (→ track death). **`Q → 0` degenerates to
optimal static fusion** — so single-doc is safe (a mild win over naive merge), never a
loss. Rule: **dynamics inert by default, engaged only by evidence of change.**

**Measurement `h`, noise `R`:** each document/chunk's boundary extraction is a noisy,
often **nonlinear/censored** observation ("at least 12", "roughly $3M") → hence the
*Extended* KF (linearize `h`).

**Association (MHT):** gate (embedding + type + temporal window) → Hungarian assignment
→ keep top-K hypotheses (the beam) → track birth/death.

## 4. The MoE gate (Reading B — DECIDED)

Two experts, per event/attribute, blended by a learned **router**:
- Expert 1 = **local read** (the per-doc boundary extraction).
- Expert 2 = **tracked state** (the EKF estimate).
- Router weights by regime: static single-doc → local read; evolving multi-doc → tracked.

Idiomatic — the count head already ships `CountLSTMoE` (experts + softmax router,
`layers.py:219`). This IS the graceful-degradation / regime gate, formalized: it learns
to withhold the EKF exactly where it would degrade a static read. Cheap (runs on top of
the base decode, not as a second full decoder). Trainable (supervise: "did the tracked
estimate beat the local read?").

## 5. Modes / CLI

Mirrors the existing `--global-decode` switch (`tools/infer.py:62` →
`batch_extract_long(global_decode=...)` → `merge_chunk_results`):
- *(no flag)* → base boundary joint-IE decode + naive merge.
- `--ekf-mode` → EKF/MHT tracker replaces the cross-chunk merge.
- `--moe-mode` → the gated {local read, tracked state} mixer (EKF runs + router blends).

## 6. Training (DECIDED: train the model to be tracking-friendly)

- **Model, tracking-friendly:** add an auxiliary **cross-document coreference /
  contrastive loss** on `candidate_states` so same-event mentions across docs are close
  and different ones far. Net-new (no coref/contrastive objective in the model today).
- **Tracker:** learn dynamics `f/Q`, measurement `R`, association affinity — on stream
  data with ground-truth trajectories.
- **Gate:** train the MoE router (regime weighting).
- **End-to-end (feasible):** the boundary model already uses the *non-differentiable
  association + differentiable recomputed scores* idiom (`records.py:1211`,
  `proposal.py:582/605`); the EKF fits it (differentiable filter, detached association),
  so backprop of a trajectory loss into the encoder is tractable.

## 7. Data

Single-doc corpora (RAMS/ACE/WikiEvents) cannot test beyond-document tracking. Need
multi-document event **streams with ground-truth evolving state**. Strongest for
controlled eval: **synthetic streams** — generate a process with a known state
trajectory, emit documents as noisy measurements, score the tracker against the true
trajectory (a clean baseline vs. recursive fusion). Connects to the large-multi-task
synthetic-corpus direction.

## 8. Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Direction: boundary head + EKF/MHT for doc-level + beyond-doc events | **DECIDED** |
| 2 | Granularity = process/situation, continuous state (embedding + quantitative + salience) | PROPOSED (finalize state/dynamics) |
| 3 | MoE gate = Reading B, experts {local read, tracked state}, learned router | **DECIDED** |
| 4 | Train the boundary model tracking-friendly (coref/contrastive aux loss on `candidate_states`) | **DECIDED** |
| 5 | CLI: default = base boundary joint-IE; `--ekf-mode`; `--moe-mode` | **DECIDED** |
| 6 | EKF must degenerate gracefully (Q→0 = fusion); dynamics inert by default | **DECIDED** |
| 7 | Revive `joint_ie` beam as a **3rd expert** | DEFERRED — big ask (needs the unbuilt boundary→`JointProblem` adapter) |
| 8 | Data via synthetic evolving-event streams (+ ground-truth trajectories) | PROPOSED |
| 9 | Quantitative-argument **normalization** layer (span → value + uncertainty) | PROPOSED (required new component) |

## 9. Phasing

1. **Inference-only `--ekf-mode`** — frozen boundary model, hand-set dynamics, `Q→0`
   graceful. Cheap validation that tracking helps on real/synthetic streams.
2. **Synthetic streaming generator** — ground-truth trajectories for train + eval.
3. **Train the tracker + MoE gate** on synthetic streams.
4. **Train the boundary model tracking-friendly** (coref/contrastive aux loss).
5. *(end-game)* end-to-end differentiable filtering.
6. *(deferred stretch)* revive `joint_ie` beam as a 3rd expert (build the adapter first).

## 10. Still open (the crux)

- **Finalize the state vector + dynamics models** per target domain (§3, decision #2) —
  the "does the EKF earn its keep" crux. **Answered for mass-casualty streams in §14**
  (yes; edge widens under unreliability/censoring, shrinks under sparsity).
- **Domain/dataset anchor** — which evolving-quantitative events; synthetic vs a real
  corpus (e.g. cross-doc event coref, ECB+).
- **Normalization layer** design (numbers/dates/amounts → continuous + noise model).
- **Router supervision** signal.

## 11. Synthetic stream generator — specification (the $0 backbone)

Anchored on mass-casualty tracking; parameters seeded from the real Venezuela 2026
trajectory (920 → 1,719 → 6,000+ dead) so synthetic streams look like real disasters.

**Per stream:**
1. **Sample the truth.** Draw asymptotes `d*, i*` (dead/injured; log-normal, magnitude-
   scaled), initial missing `m0`, dynamics rates `β` (approach), `γ` (missing decay),
   coupling `κ` (resolved missing → dead), attention decay `λ`, duration `T`. Integrate
   the linear ODEs (§3) on a fine grid → the **ground-truth trajectory** `x(t)`.
2. **Sample report times** `t_k` ~ inhomogeneous Poisson with intensity ∝ salience
   `s(t)` (dense early, sparse late).
3. **Emit observations.** At each `t_k` pick reported roles ⊆ {dead, injured, missing,
   displaced}, a **qualifier** ∈ {point ("rises to"), lower_bound ("at least"),
   interval ("dozens/hundreds"), coarse}, and a **source** ∈ {official, major_outlet,
   preliminary} → `value` = censored/noisy read of `x(t_k)` (noise & bias scaled by
   source; lower_bound ⇒ value ≤ truth). Record `(stream_id, t_k, role, value,
   qualifier, source)`.

**Outputs (committable, small):**
- `observations.jsonl` — one row per observation (the tracker's input).
- `trajectory.jsonl` — ground-truth `x(t)` on the grid (the eval target).
- `config.json` + seed (reproducible).

**LLM-realistic layer (sonnet-5, ~$9, only for the end-to-end val/test subset):**
render each observation as a short realistic news snippet conditioned on
`(role, value, qualifier, source)`; reuse `tools/data/synthetic/` infra. The boundary
model extracts figures from this text; the tracker consumes the figures. Parametric
streams (no text) cover train + controlled ablations for free.

**Eval (vs `trajectory.jsonl`):** trajectory RMSE over time, final-toll error, CI
calibration; baselines = last-value, weighted-average, heuristic merge.

## 12. Venezuela 2026 — the double-blind real test (copyright-aware)

Validated real event (7.5+7.2, 24 Jun 2026; official toll 920→1,719→6,000+→~6,125).
Sources: CNN hourly liveblogs (map onto hourly buckets), UN News, USGS, Al Jazeera,
Wikipedia timeline.

- **Held out entirely** (blind) — never in train/val.
- **Ground truth** = the *official-source* figure trajectory (govt/UN/USGS over time) +
  settled toll, annotated separately from the reporting stream.
- **Copyright:** commit only **extracted observations + timestamps + source URLs +
  the GT trajectory** — NOT full article text. Text is fetched at eval time from the
  stored URLs.

## 13. Data layout + git

`/data/` is git-ignored; committable tracking data lives under a new top-level
**`datasets/`**:

```
datasets/disaster_streams/{train,val,test}/{observations,trajectory}.jsonl   # synthetic
datasets/disaster_streams/config.json
datasets/venezuela_2026/observations.jsonl   # extracted (role,value,qualifier,source,ts,url)
datasets/venezuela_2026/trajectory.jsonl      # official GT toll over time
datasets/venezuela_2026/sources.jsonl         # url, outlet, timestamp (no full text)
```

Generator: `datasets/disaster_streams/generate.py` (parametric = free; `--realize`
adds the sonnet-5 text layer for the val/test subset).

## 14. Harder-regime ablation — the EKF earns its keep (+ a measurement-model fix)

Date: 2026-08-07. Free/CPU, validated on val + held-out test. Resolves the §10 crux.

`generate.py --regime` stresses the *observation* regime by flipping three knobs from
`normal`: **sparse** reporting (rate0 8-30 → 2-6), **unreliable** sources (official/major/
prelim 50/35/15 → 20/30/50), **heavy censoring** (more at_least/interval/feared).
Trajectory params draw before rate0, so **every regime shares byte-identical ground
truth** — the same 40 disasters, only the reporting differs (paired eval).

**The naive prediction (edge widens under stress) was falsified — and the failure was a
bug, not a regime effect.** Single-knob decomposition isolated it (rise-role EKF penalty
over `last_value`): sparse **+0.064**, unreliable **+0.110 (dominant)**, censored **+0.010
(negligible)**. Root cause: measurement noise `R = (sig·value)²` scaled by the *observed*
value → a low report got a small R (over-trusted) → systematic **downward drag on a rising
toll**, amplified by noisy sources. The one-sided `at_least` handling for censoring was fine.

**Fix:** scale R by the *estimate*, not the raw reading (`_R_at(o, ref)` — linearize the
measurement noise around the state). Tracker then beats `last_value` in every regime:

| regime | last_value | EKF | MoE_gate |
|---|---|---|---|
| normal | 0.194 | **0.121** | 0.123 |
| hard | 0.265 | 0.244 | **0.240** |
| hard-test (held out) | 0.285 | 0.236 | **0.234** |

(overall normalized RMSE, lower better; ~25-40% over the baseline, decay unregressed).

**Answer to §10:** the EKF earns its keep — and the edge **widens under unreliability +
censoring** (noise-weighting + one-sided updates pay off) and **shrinks under sparsity**
(less to fuse → converges to `last_value`). Two follow-ons: (a) the fixed EKF now *ties*
the hand-set MoE gate → motivates a **learned router** (decision #3, next build);
(b) the win rides on a correct measurement model → **Venezuela** (real, model-mismatched)
is the true test.

## 15. Learned gate — a trained router beats the hand-set tables

Date: 2026-08-07. Free/CPU. `evaluate.py --learn-gate`. Realizes decision #3 (Reading B).

Replaces §4's hand tables (`SRC_TRUST`/`QUAL_TRUST`/`GATE_TAU`) with a logistic router
`alpha = sigmoid(w·x)`, `x = [staleness, source σ, qualifier factor, at_least flag,
decay-role flag, ekf−lastvalue gap, #reports]`. Trained on the train split by GD to
minimize the peak-normalized blend MSE (the eval metric). ~8 params, ~70k points, <1s.

Best method in every regime/split (overall normalized RMSE):

| | last_value | EKF | MoE_hand | **MoE_learned** |
|---|---|---|---|---|
| normal/val | 0.194 | 0.121 | 0.123 | **0.120** |
| normal/test | 0.186 | 0.114 | 0.115 | **0.113** |
| hard/val | 0.265 | 0.244 | 0.241 | **0.228** |
| hard/test | 0.285 | 0.236 | 0.234 | **0.231** |

Margin over the fixed EKF is tiny in normal (the corrected measurement model is already
near-optimal there) but real in the **hard** regime (0.228 vs 0.244) — where a static
hand-table can't adapt. The learned weights recover the ablation's lesson (unreliable
source / coarse qualifier / `at_least` → trust the tracker) and add role-dependence + a
disagreement regularizer the hand table lacked. This is the piece meant to transfer to
the real boundary model / **Venezuela**, where hand-set trust constants won't.
(Honest caveat: it optimizes trajectory RMSE and trades a hair of final-value accuracy on
hard-test; and it's fit + eval'd in-regime — cross-regime transfer is the next check.)
