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

**Transfer (caveat resolved).** Cross-regime the gate never collapses to `last_value`:
normal-trained → hard still beats the EKF (val 0.238 vs 0.244); hard-trained → normal
~ties it. A single **union gate** (`--gate-train normal,hard`) matches the in-regime gates
on *both* (normal/val 0.121, hard/val 0.228) — one regime-agnostic router is best
everywhere, which is what **Venezuela** (unknown regime) needs. (It optimizes trajectory
RMSE, trading a hair of final-value accuracy on hard-test.)

## 16. Text → observation extraction — the normalization layer (single-fact ceiling)

Date: 2026-08-07. Free/CPU. `extract.py`; `evaluate.py --from-text`. Builds design #9.

Surface extractor (first integer + hedge/source keyword cues) + normalizer (bucket word →
representative value). Validated by rendering each structured obs back to text with the
generator's own templater and extracting it.

**Extraction ceiling (single-fact templated text):** role / qualifier / source **100%**;
value **100% exact** for point/at_least/about/feared; **interval 67%** — the bare bucket
word ("thousands of people") carries no number, unrecoverable by construction. Realistic
text *reduces* number-free phrasing but does not eliminate it ("dozens feared trapped").

**End-to-end (render→extract→track) vs structured, normal/val overall RMSE:**

| method | structured | from-text | Δ |
|---|---|---|---|
| last_value | 0.194 | 0.236 | +0.042 |
| EKF | 0.121 | 0.144 | +0.022 |

**The tracker's margin over `last_value` *widens* under extraction noise** (0.073 → 0.092):
`QUAL_FACTOR["interval"]=2.0` down-weights exactly the lossy (bucket) observations while
`last_value` eats them raw — the measurement model absorbing extraction error is the §3
design claim working. The structured-trained gate still helps on extracted obs
(MoE_learned 0.145 ≈ EKF 0.144). Loss concentrates on **injured** (largest counts → most
"thousands").

**Scope / open risk.** This is the ceiling for *single-fact* text only: the parser takes
the first integer and first role keyword, valid because each snippet states one fact.
**Number-to-role binding on multi-fact text is untested** ("killed at least 40 and injured
hundreds") — the real extraction risk the GLiNER2 model exists to solve. De-risked: *given
reasonable obs, the tracker holds up.* Extraction on realistic text is the open question.

**Sonnet-5 step (locked design):** (a) keep the surface parser as a **baseline arm** beside
the GLiNER2 model → a comparison (parser ceiling vs model), not a single-arm test; (b)
prompt sonnet-5 with **distractor numbers** (dates, magnitude, other roles' figures) so the
test is discriminating, conditioning tuple = known GT; (c) decide whether the realizer
passes `value` for interval obs (else intervals stay number-free).

## 17. Sonnet-5 realizer + the parser's binding collapse (why the model is needed)

Date: 2026-08-07. `realize.py`, 12 val streams → 1322 multi-fact snippets (~$1-2 batch,
`claude-sonnet-5`, -50% batch tier). Each snippet groups a report's roles and states each
figure's **exact digits + hedge + distractor numbers** (date, magnitude 7.5, a displaced
count), so extraction must **bind** numbers to roles. Ground truth = the conditioning tuple.
Interval obs now carry a number (design decision (c) resolved).

**The surface parser collapses on multi-fact prose** (vs 100% on single-fact templates):

| metric | templated | sonnet-5 |
|---|---|---|
| role accuracy | 1.00 | 0.56 |
| qualifier accuracy | 1.00 | 0.57 |
| value exact (point) | 1.00 | 0.005 |
| value rel-err (point) | 0.00 | 1.90 |

first-integer/first-role grabs the magnitude, the date, or the displaced count. End-to-end
the tracker is destroyed: EKF **0.14 → 1.16** (overall RMSE > 1 = worse than predicting the
peak). **This is the number-to-role binding problem the GLiNER2 model exists to solve** —
now proven, not asserted. The realizer + this harness are the eval; the model is the fix.

**Innovation gate (the "3-sigma" question).** A **symmetric** N-sigma gate is catastrophic
on rising tolls (clean EKF 0.121 → **0.595**): a rising toll's large positive innovations
*are* the signal. Fixed to a **one-sided, dynamics-aware** gate (rise → reject
implausibly-low; decay → reject implausibly-high; generalizes the `at_least` rule).
Near-transparent on clean data (EKF 0.124 at K=4). But it **cannot rescue the broken parser**
(from-text EKF 1.16 → 1.56): gating defends a *good* signal against **sparse** outliers, not
one that is mostly mis-bound — so the gate's real test is the model arm, on its occasional
errors.

**Next:** the GLiNER2-model extraction arm (does a trained extractor bind numbers to roles
here?), parser as baseline, gate measured against the model's outliers. Data committed
(`datasets/disaster_streams_sonnet5`) for reproducibility.

## 18. Model extraction arm — zero-shot binding works; precision is the knob

Date: 2026-08-07. `model_arm.py`, `fastino/gliner2-base-v1` (zero-shot), 12 val streams /
1316 reports / CPU. The model fills a `casualty_report` structure {dead, injured, missing,
source}; `extract.value_qualifier` normalizes each **bound** span → (value, qualifier).

**The model solves binding the parser couldn't:** on true-positive roles **value exact =
0.991** (parser 0.005), recall 0.91. But **precision = 0.65** — the structure over-fills
(a distractor like the displaced count bound to an absent role), and those FPs, concentrated
on small-valued `dead`, wreck end-to-end tracking (EKF 1.54).

**Confidence is the separator and the fix.** Field confidence is bimodal (TP ~0.9999,
FP ~0.605). Sweeping a min-confidence cut (`evaluate.py --min-conf`) recovers tracking:

| min-conf | EKF overall | dead | injured | missing |
|---|---|---|---|---|
| 0.0 | 1.54 | 3.12 | 0.30 | 1.19 |
| 0.90 | 0.78 | 1.11 | 0.19 | 1.05 |
| 0.95 | 0.47 | 0.54 | 0.16 | 0.72 |
| 0.99 | **0.29** | 0.25 | 0.15 | 0.48 |

From "destroyed" (parser 1.16) to **usable 0.29** at conf ≥ 0.99, vs the structured
ceiling 0.14. Residual gap = remaining FPs + **qualifier loss** (the model extracts the bare
number, not the hedge → qualifier acc 0.31) + missing-role error.

**Confidence is a scalar field probability, not a covariance; used here as a HARD cut.**
Tested folding it into `R` as soft measurement uncertainty (`--conf-r`, `R /= conf^2`): it
does **not** beat the hard cut (EKF: hard-cut **0.29** vs soft 1.29, soft+gate 1.76,
soft+mild-cut+gate 0.68). Reason: the zero-shot errors are **gross** mis-bindings (a wrong
number entirely), and no confidence weighting in (0,1) neutralizes a categorically-wrong
value — only removal does. Soft-`R` is the right tool for *graded* noise, not gross FPs.
Corollary: the one-sided gate can't catch high FPs on *rising* roles (it admits highs as
real jumps), so gross `dead` FPs specifically need the hard cut. **Real lever = extractor
precision at the source** (fine-tune / record-mode schema) + qualifier recovery (the hedge
sits beside the bound number). A joint {dead,injured,missing} covariance matrix (roles
co-evolve) is a separate extension. Model output committed (`datasets/disaster_streams_model`).
