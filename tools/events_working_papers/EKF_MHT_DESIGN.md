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

### 1a. The question, stated the way it was actually asked

**Can a Kalman filter track real-world events reported in text, and DIARIZE them into
separate streams?**

Two halves, co-equal, and the second is not downstream of the first:

| half | mechanism | status |
|---|---|---|
| **track** — recover an evolving quantity from noisy, censored, lagged reports | EKF (§3) | built, and validated on synthetic streams (§14, §20) |
| **diarize** — decide WHICH event stream each observation belongs to | MHT (§3 association) | **NOT BUILT** — see below |

"Diarize" is the right word and it is borrowed deliberately from speaker diarization: not
"who spoke when" but *which event is this figure about, over time*. Naming it that way is
clarifying, because it makes obvious that the two halves fail independently and that the
second can silently destroy the first — which is exactly what the real-event validations
found (§21, and the Helene run).

**The MHT half is unbuilt, and this should be stated plainly.** §3 specifies association as
*gate → Hungarian assignment → top-K hypotheses (the beam) → track birth/death*. What
actually ships is **hard assignment on an observable string key** (`association_key` /
`record_key` / `merge_prefix_keys` in the showcase) feeding **one single-stream EKF per
key** (`est_ekf`). There is no hypothesis enumeration, no deferred decision, no track
birth/death. The only Hungarian solver in the repository is inside the boundary model's
record LOSS, matching predictions to gold during training — unrelated to tracking.

That reframes every diarization result so far. Turkiye-Syria losing Syria entirely, and
Helene fragmenting one event into 18 keys while needing the opposite (pooling to a national
aggregate), are failures of a **placeholder**, not of MHT. The mechanism the design names
for this exact problem has never been tested, so the programme's central question is not
yet answered — it is, so far, unasked.

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

**$0 extractor wins + held-out validation.** Two normalization/schema wins, no training:
(1) **qualifier recovery** — read the hedge from the bound number's local context
(`extract.qualifier_near`): qual-acc 0.31 → 0.74; (2) **distractor-excluding field
descriptions** ("injured, *not* displaced"). Val end-to-end EKF: 0.29 → 0.20 → **0.172**
(min-conf 0.99). Confirmed on a **held-out test** split (12 fresh sonnet-5 streams, never
seen): extraction stable (P 0.63, value 0.991, qual 0.72); end-to-end **EKF 0.291** — beats
parser (~1+) and last_value (1.09), but ~2.5× the test structured ceiling (0.115) and worse
than val's 0.172 (the val figure was optimistic — threshold/descriptions were val-informed).
Residual weak role: **missing** (test 0.46). Record-mode precision needs a *boundary* model
(span decoder ignores `mode`). The model pass is cached (`raw.jsonl`) so normalization
re-runs free (`--from-raw`); a paid batch whose poll dies is recoverable by id
(`providers.fetch_batch` / `realize --batch-id`) — no re-spend.

## 19. `missing`-role probe ($0) — it's confidence-cut selection, not extraction

Date: 2026-08-07. `missing` is the residual weak role (test EKF 0.46, val 0.23 vs structured
ceiling ~0.12). Diagnosed on the cached raw spans + GT:

- **Extraction quality is fine.** At conf 0.99: missing **value-exact 0.996, precision 0.96**
  (as good as dead/injured). What differs: **recall 0.55** (lowest) and **qualifier acc 0.48**
  ("feared" follows the number, so a small right-window labels it "point").
- **The qualifier fix does NOT help.** Recovering "feared" (wide right window: qual 0.48→0.83)
  and treating it as a one-sided upper bound made missing *worse* (0.23→0.29): a sparse decay
  filter needs "feared" readings as usable anchors, not down-weighted/skipped. Reverted.
- **Sparsity alone doesn't explain it.** Random 55%-recall subsample of the *structured*
  missing obs → 0.138 (vs model 0.233).
- **It's selection bias.** Structured missing (perfect values + quals) restricted to the exact
  time-points the model retained → **0.252** — worse than the model itself. So *which*
  report-times survive the confidence cut is the whole story: the cut drops the (hedged/vague)
  late reports that anchor the decay tail, leaving an early/high-skewed set that over-estimates
  the tail.

**Conclusion:** not a $0 normalization fix. The lever is recall/**calibration** of the retained
set (extractor fine-tune, or a role-aware threshold) or a stronger decay-tail prior — all
bigger than $0. Pipeline stays at the validated val 0.172 / test 0.291.

## 20. Fine-tuned extractor closes the gap (the §19 prediction, delivered)

Date: 2026-08-07. Fine-tuned `fastino/gliner2-base-v1` on **29,198** casualty-structure
examples (250 sonnet-5 train streams), A100 · 8 epochs · ~$2 GPU + ~$19 data. Same held-out
test harness (`model_arm.py` → `evaluate.py`). §19 said the bottleneck was extractor
precision + confidence calibration (the `missing` selection bias) — extractor-side. It was:

| held-out test | zero-shot | fine-tuned |
|---|---|---|
| role precision | 0.627 | **0.914** |
| role recall | 0.906 | **0.965** |
| value exact (binding) | 0.991 | **1.000** |
| qualifier acc | 0.724 | 0.691 |

End-to-end (EKF, normalized RMSE; structured ceiling **0.115**):

| | zero-shot | fine-tuned |
|---|---|---|
| best (min-conf 0.99) | 0.291 | **0.165** |
| **no cut** (min-conf 0) | 1.16 | **0.193** |
| **`missing`** role | 0.458 | **0.122** |

The fine-tune closed **~75% of the gap** to the structured ceiling and **fixed the `missing`
selection bias** (0.458 → 0.122 ≈ the structured `missing` ceiling) — exactly §19's
prediction. High precision (0.91) means it barely needs the confidence cut (0.19 *uncut*);
number binding is perfect (1.000). **Caveat:** fine-tuned + tested on sonnet-5 text (same
distribution) — **Venezuela (real news) remains the true generalization test**. This is the
end of the synthetic arc: text → extraction → tracking, held-out, at ~1.4× the clean-obs
ceiling.

## 21. Turkiye–Syria 2023 — the first real event, and the first real defeat

Everything above §20 was measured on text we generated. This is the arc's first run
against a real event with an externally sourced trajectory, and it **reverses the
standing conclusion**: on real news the EKF *loses to the trivial baseline*, and the
binding component that §17–§20 spent the most effort on is not the thing that breaks.

Full write-up and provenance: `datasets/turkey2023/{PREREGISTRATION,RESULTS}.md`. The
configuration and four predictions were committed **before** the run.

**Ground truth.** 16 daily points, 6–21 Feb 2023, one Al Jazeera live-tracker page
sampled through the Wayback Machine, one archive URL per point. Turkiye 1,014 → 41,000,
Syria 783 → 5,800, both monotonic. Truncated at 21 Feb, where the page froze at 41,000
and reported it into April while the real toll reached 53,537 — the *source* going stale,
not the event plateauing. Search-result summaries were rejected as a source outright after
they disagreed with each other on the same dates.

**Results** (`dead`, nRMSE vs the Turkiye trajectory, 6-hour grid):

| Run | Streams | EKF | `last_value` | 1999 toll bound |
|---|---|---|---|---|
| Pre-registered config | 1 | 0.288 | 0.343 | 12 / 20 obs |
| `+ --event-model` | 1 | 0.208 | **0.136** | 16 / 91 obs |
| `+ --associate envelope` | 5 | 0.228 / 0.196 | — | 16 / 91 obs |

Three findings, in order of how much they should change what we do next:

1. **Attribution is the bottleneck — not extraction, not the filter.** Extraction reads
   the real trajectory almost point-for-point (1,014 / 2,316 / 5,434 / 9,057 / 17,674 /
   … / 41,020). The filter is sound on clean observations (§14). What fails is deciding
   *which event a number belongs to*.
2. **The tracker loses to "repeat the last number you read"** (0.208 vs 0.136), ending at
   26,972 against a truth of 41,000 while the baseline ends at 40,642. Not because the
   filter is wrong, but because its noise model assumes error roughly zero-mean about the
   truth, and the observation stream is *contaminated*, not noisy.
3. **A death toll from a different earthquake is tracked as this one.** The article's
   historical round-up mentions the 1999 Izmit quake's 17,500 dead; that value is bound
   as a 2023 observation in every configuration, and is the single most frequent value in
   the set. The gate answers "is this article about a mass-casualty event" (correctly,
   16/16). Nothing answers "does this number belong to *that* event."

**Syria is never recovered.** `association_key` was computed once per *document*, outside
the envelope loop, while `casualty_windows` directly above builds one envelope per
incident. Every article names Turkey before Syria, so all 16 documents keyed to
`Earthquakes|turkey`. Syria *was* detected — 123 location spans — it never reached the key.
Keying per envelope by nearest location (`--associate envelope`, opt-in) splits the feed
into 5 streams and still does not fix it: the resulting `syria` stream carries 17,674,
20,213, 35,418 and 41,000. Character distance is not syntactic attachment.

**This is the §10 crux arriving in the wild.** The file already reserved genuine
attachment ambiguity for MHT proper. Turkiye–Syria says that case is not exotic — it is
what one ordinary news sentence looks like: *"At least 41,000 deaths have been reported
in Turkey, while 5,800 people have died in Syria."*

**Method note that outlived the experiment.** A pre-registered prediction *failed*: pooling
was predicted to drive nRMSE past 1.0 and it read 0.288. Range-normalized RMSE never
noticed that 12 of 20 readings came from a 1999 earthquake, because 17,500 happens to sit
mid-range of a 1,014 → 41,000 trajectory. **A badly wrong observation scored well.** Only
carrying `est_last_value` alongside exposed it. Report the baseline, always.

**Caveat.** Known-answer, not blind: Feb 2023 precedes the assistant's cutoff, which is
why every figure is sourced and cited rather than written from memory. §12's Venezuela
2026 remains the genuinely blind test.

## 22. References

- **Kozak, M. C. "Multiple Model Methods for Cost Function Based Multiple Hypothesis
  Trackers." 2012.**
  [Semantic Scholar](https://www.semanticscholar.org/paper/75221aef94167cb5797428d55cb01b826283e840)
  Incorporates **multiple-model Kalman filters into an Integral Square Error (ISE)
  cost-function-based MHT** to raise state-estimation fidelity. Finds the multiple-model
  structure correctly identifies a target's maneuver mode **in dense clutter**, so an
  appropriately tuned filter is used — large position/velocity RMS reductions vs. a
  single-filter MHT during benign flight.

  **Why it is the closest prior work to this design.** It is the same two choices we made,
  in the classical setting: a *cost-function* hypothesis score (what §3's tracker
  optimizes) plus a *bank of models arbitrated per hypothesis* — the classical form of the
  §4 MoE gate. Our learned router over {local read, tracked state} experts (§15) is a
  learned instantiation of what the classical treatment does with fixed IMM mixing; the
  dense-clutter result is the analogue of the §14 harder-regime ablation, where the EKF
  earns its keep precisely as conditions degrade.

  **The honest negative to design against.** During *deferred decision periods* — when the
  mixture mean drifts far from true target position — the multiple-model structures
  **accumulate greater RMS error** than the single filter. That is a direct prediction
  about our gate: a router is a liability exactly when the mixture is bimodal and no expert
  is yet right. Worth probing against §15's learned gate before trusting it on the
  Venezuela stream, where sparse contradictory reporting is the real-world deferred-decision
  case. **Not yet tested.**
- Williams, J. L. and Maybeck, P. S. "Cost-Function-Based Hypothesis Control Techniques for
  Multiple Hypothesis Tracking." 2006. The cost-function MHT line Kozak builds on.
- Reid, D. B. "An Algorithm for Tracking Multiple Targets." IEEE T-AC, 1979. (Seminal MHT;
  the hypothesis-beam ancestor — see [[KALMAN_BEAM_SEARCH_EXPLORATION]] §3.)

Fuller literature scan lives in [[KALMAN_BEAM_SEARCH_EXPLORATION]] §9; this section carries
only what this design leans on directly.

### Expository / practitioner references

Not peer-reviewed and not evidence for any claim here; listed because they are the
accessible on-ramps to the machinery §3 uses, and because a reader arriving from the
extraction side of this program will not have the tracking background the sections above
assume. Titles and authors verified by fetching each article, not inferred from the URL —
the second one's slug says "oise-injection" and its actual title says *Noise Injection*.

- **Weaver, J. "Exposing the Power of the Kalman Filter." Towards Data Science,
  7 Nov 2023.** — **the origin of this research programme.** Read September 2024; the idea
  was turned over for roughly two years before the first commit of this line of work
  (2026-08-07).
  https://medium.com/data-science/exposing-the-power-of-the-kalman-filter-1b78621c3f56

  Walks the predict/update cycle from first principles in Python, moves to 4D object
  tracking, then motivates the **Extended** Kalman filter for nonlinear systems — which is
  the shape this programme took: §3's tracker is an EKF precisely because a casualty toll
  is nonlinear, saturating and censored rather than linear.

  Worth stating plainly for anyone reading the design cold, because the papers do not
  otherwise show it: the direction of travel was **filter first, extraction second**. The
  question was whether a Kalman filter could track a real-world quantity reported across a
  stream of documents; document-level event extraction, the boundary head and the whole
  joint_ie line are what that question turned out to require. That ordering explains why
  §1 scopes the claim so narrowly to *streaming, quantitatively-evolving events* rather
  than to event extraction in general.

- **Maxwell's Demon. "Kalman Filters Demystified — The Algorithm Behind Moon Landings."
  Towards AI, 5 Nov 2025.**
  https://pub.towardsai.net/kalman-filters-demystified-the-algorithm-behind-moon-landings-6fcf46433a50
  Conceptual introduction aimed at readers without the control-theory background, starting
  from noisy GPS. Useful for framing the filter as *fusing a prediction with a
  measurement, weighted by relative confidence* — which is exactly the intuition our
  ``QUAL_FACTOR``/``SRC_REL_SIGMA`` measurement model encodes, where a hedged or
  preliminary report is admitted with wider variance rather than rejected.

- **Weaver, J. "Unravelling Complexity: A Novel Approach to Manifold Learning Using Noise
  Injection." Towards Data Science, 17 Nov 2023.**
  https://medium.com/data-science/unravelling-complexity-a-novel-approach-to-manifold-learning-using-oise-injection-41251565fded
  Not about Kalman filtering — it compares PCA, LLE, spectral embedding and Isomap, and
  evaluates them by **injecting synthetic noise and measuring structural stability via
  Procrustes analysis**. The relevance is methodological rather than topical: it is the
  same evaluation stance as §14's harder-regime ablation, where a method is judged by what
  survives as conditions degrade rather than by its score on the clean case.

## 23. Aggregate-vs-parts: the measurement-model reframe is right and does NOT pay off here

Helene reporting carries a national total and its state components together -- "227 across
six states, including 120 in North Carolina and 17 in Tennessee". Filing 227 under whichever
state is nearest (what ships) is plainly wrong: it is not a rival claim about North Carolina.

**The reframe.** This is a measurement-model question, not a clustering one. Make the state a
vector and every report is a linear observation differing only in its `H` row:

    "120 in North Carolina"     H = e_NC          z = 120
    "227 across six states"     H = [1,1,1,1,1,1] z = 227

A Kalman filter fuses both natively. A Gaussian mixture or mean shift cannot express the sum
at all and would cluster in VALUE space, where 12 (Florida) and 17 (Tennessee) look like one
cluster and 120 and 227 like two -- backwards.

**Tested** (`vector_state_test.py`): real Wikipedia per-state trajectories including the
genuine North Carolina revision 123 -> 102 -> 96; only the reporting PROCESS simulated
(frequent totals, sparse per-state, 10% noise), because the real feed's 25 observations are
too thin to test a filter. *(Stale as of 2026-08-11: that count is from the superseded
`tracked_lead` run. `extract_long` brings the same feed to 106 dead observations across 44 of
70 articles, which is 4.2x and is no longer starvation — the simulation is still needed for
per-state DENSITY sweeps, but "too thin" is no longer the reason.)* Both arms use the same filter and the same per-state
observations; only the aggregates differ. Mean per-state nRMSE, 40 trials:

| per-state report rate | parts-only | vector | delta | vector wins |
|---|--:|--:|--:|--:|
| 10% | 0.4348 | 0.6085 | +0.174 | 4/40 |
| 20% | 0.3135 | 0.3940 | +0.081 | 10/40 |
| 35% | 0.2292 | 0.2860 | +0.057 | 10/40 |
| 50% | 0.2030 | 0.2234 | +0.020 | 22/40 |
| **80%** | 0.1556 | **0.1520** | **-0.004** | 30/40 |

**Two predictions, both wrong, and the second is the finding.**

1. *Isotropic process noise was a modelling error I made, not a property of the approach.*
   With `H = [1,1,...]` and isotropic `P`, an aggregate's correction spreads EQUALLY across
   components, shoving Virginia (range 1->2) as hard as North Carolina (6->123). Scaling `Q`
   with the estimate cut the harm 7.7x at the sparse end (+1.338 -> +0.174). Components at
   different orders of magnitude need proportional process noise; that is a precondition,
   not a tuning knob.

2. *I predicted the aggregate would help MOST where per-state reports are sparse -- adding
   information where there is little. The opposite is true.* An aggregate constrains the SUM
   but says nothing about the SPLIT, and the split has to come from the parts. When parts
   are sparse the filter must guess the division, so a total injects error into individual
   states; when parts are dense the division is already pinned and the total refines the sum
   cleanly. **Aggregates are a refinement, not a substitute -- they cannot bootstrap a
   split.**

**Consequence for this programme, and it is a negative.** AP wire copy reports mostly
national totals and rarely breaks them down -- measured on the Helene feed. That is exactly
the sparse-parts regime where the vector formulation HURTS. So the reframe is structurally
correct and buys nothing on the data we actually have; the best case measured is a 0.004
improvement at a report density the source does not supply.

The blocker is therefore not the measurement model. It is that per-state reporting is too
sparse to pin the split, which is a RECALL and source-coverage problem -- the same
conclusion the Helene run reached from the other direction. Fixing extraction recall and
adding sources should come before any further work on the aggregate machinery, and
certainly before MHT, whose hypothesis space would inherit the same starvation.

Caveat: the reporting process is simulated and `Q`, `R` and the noise model are chosen. The
qualitative reversal is mechanistically explicable and holds across the whole sweep, but the
exact crossover density is not a measured property of the world.

## 24. Revision test: the EKF wins reliably and marginally, and §14's regime claim does not reproduce

The only real revision data this project has: Wikipedia's Helene table records North Carolina
falling **123 -> 102 -> 96** as deaths were reclassified direct-to-indirect. Turkiye-Syria was
purely monotone and never exercised this. Both estimators are the ones that ship
(`evaluate.est_ekf` / `est_last_value`); truth is real, only the reporting process is
simulated, because per-state recall is still too thin to drive a filter from the feed.

A revision is the case where `last_value` looks BEST -- it adopts the new figure at once
while a filter lags -- so the EKF can only win by smoothing noise the baseline chases.

| regime | ekf | last_value | gain | ekf wins |
|---|--:|--:|--:|--:|
| noise 0.05, report 80% | 0.094 | 0.095 | +1.3% | 191/240 |
| noise 0.10, report 60% | 0.189 | 0.192 | +1.5% | 205/240 |
| noise 0.20, report 40% | 0.322 | 0.327 | **+1.8%** | 217/240 |
| noise 0.30, report 30% | 0.499 | 0.504 | +1.0% | 222/240 |
| noise 0.40, report 20% | 0.628 | 0.634 | +0.8% | 217/240 |

**It wins, reliably.** 80-93% of trials, and 6/6 states at the default setting. On the first
benchmark in this project whose baseline is not an oracle, the filter is ahead.

**By 1-2%.** That is the whole margin. It is a real effect and a small one, and it should be
reported as such rather than as vindication.

**§14 does not reproduce.** That ablation concluded the EKF's edge *widens* under
unreliability and censoring -- it is quoted in §10 as the answer to the "does the EKF earn
its keep" crux. On real trajectories the gain is roughly flat and *shrinks* at the hardest
setting (+1.8% at noise 0.20 falling to +0.8% at 0.40). The most likely reason is that §14
measured synthetic streams generated by the same rise/decay dynamics `est_ekf` models, so
the filter was matched to the process by construction; Helene's trajectories -- plateaus,
reclassifications, a genuine downward step -- are not that process. **A dynamics model
validated on data generated from it is not validated.**

Consequence: the §10 crux should be reopened, not treated as answered. The honest position
is that the EKF is *slightly* better than repeating the last reading on real revision data,
and that the case for it has to rest on something other than the harder-regime argument --
most plausibly on multi-source disagreement, which no benchmark here has yet tested because
every feed so far has one source.

## 25. Scope gate: the attachment blocker, mostly solved -- and what held-out testing cost it

§23 asked whether an aggregate can *constrain* the parts and answered no. This section asks
the opposite question -- whether the parts can *classify* the aggregate -- and the answer is
largely yes. Same two quantities, opposite direction of information flow.

### 25.1 The failure is one-directional, which is what makes it tractable

Every per-place stream in the Helene run carries larger-scope numbers, and the leak is
**always upward**. Not once does a smaller scope contaminate a larger one:

| stream | truth (final) | contaminants received |
|---|--:|---|
| Florida | 26 | 64, 150, 150, 160, 180, 230, 230, 300 |
| North Carolina | 96 | 200, 215, 215, 227, 230x3, 250, **1400** |
| South Carolina | 51 | 72, 200, 227 |
| Georgia | 34 | 178 |

A one-directional error with a 2-10x separation is gateable. A symmetric one with a few
percent separation would not be.

Note also what the same audit killed: routing **unlocated** figures to `__aggregate__`,
proposed first because it had no bootstrap dependency, touches **4 of 106** observations.
It is a no-op, and only counting said so.

### 25.2 Judge against a larger scope, and classify three ways

Gating a stream against its **own** running scale fails on its early history, where a toll
legitimately jumps 6 -> 25 faster than any ratio tolerates. It has to be judged against
something larger. Three outcomes, and the third is load-bearing:

    keep     below ratio of the larger-scope value -- plausibly the place's own
    reroute  within [1/ratio, ratio] of it -- it IS the larger-scope figure
    drop     above it -- no scope in this event can exceed the whole

An earlier **two-way** version rerouted every reject and destroyed the national stream
(0.402 -> 2.110), because North Carolina's **1400** is not a national total -- it is not a
casualty count at all -- and moving it into `__aggregate__` poisoned the one measurement in
this project whose scope is known correct.

### 25.3 On Helene it is a 9x win, with a control

| ratio | Total | per-state mean |
|---|--:|--:|
| off | 0.402 | 5.247 |
| 2.5 | **0.316** | 0.592 |
| 2.0 | **0.316** | **0.591** |
| 1.5 | 0.317 | 0.591 |

Per-state **5.247 -> 0.591**, and the national stream *improves* rather than paying for it.
Flat across ratio 1.5-2.5, so not a knife-edge setting. **Control:** removing the same 25
observations at random over 40 trials gives 4.427, so the gate is *selecting* and not merely
thinning a filter into looking better.

This is also the first concrete job for the MHT/gating layer of §3, which had been deferred
as inheriting upstream defects. It is gating in its simplest useful form and needed no new
model.

### 25.4 Held out on Turkiye-Syria: the mechanism transfers, the reference does not

Ratio fixed at 2.0 from Helene, not retuned.

**As specified it cannot run.** The gate judges against `__aggregate__` and Turkiye-Syria has
none: turkey and syria are siblings with no declared parent, and the combined toll never got
its own stream. The gate is a no-op at every ratio.

Generalizing the reference to the running max across all streams makes it run, and splits:

| | turkey | syria | mean |
|---|--:|--:|--:|
| off | **0.228** | 3.401 | 1.815 |
| gate @2.0 | 0.522 | **0.923** | 0.723 |

Syria was 65% contaminated -- 11 of 17 values are Turkey's tolls, *not* an aggregate -- and
improves 3.7x. But **Turkey, already clean, degrades 2.3x**: the global maximum is dominated
by Turkey's own values, so Turkey is judged against a reference it defines itself. It
rerouted 1,014 at t=12.5h, which is Turkey's *true* value at that time. Circular by
construction. The mean still improves 2.5x and the control still shows selection (1.440 vs
0.723), so the mechanism is real.

**The finding: the gate needs a declared scope HIERARCHY, not just a magnitude.** Helene has
one -- `rollup.json`'s `__aggregate__` declares states subset national. Without it, a
magnitude test cannot distinguish "a larger scope" from "the largest part", and those are
genuinely identical from the numbers alone: Turkey's 41,000 filed under Syria and Turkey's
41,000 filed under Turkey look the same to any ratio. Declaring the hierarchy per event is
the next step and is the same information `rollup.json` already carries.

### 25.5 Declaring the hierarchy: done, and the reference it enables does NOT pay off

Both events now declare containment explicitly rather than leaving a magnitude test to infer
it. `rollup.json` gains:

    "hierarchy": {"aggregate": "__aggregate__", "parts": ["florida", ...]}

and `datasets/turkey2023/rollup.json` is created for the first time, declaring
`turkey`/`syria` as parts and mapping the joint-scope phrasing this feed actually uses --
verified present as *"Syria and Turkey"*, with the reverse order absent.

The declaration enables an **implied-maximum** reference: judge a part against
`aggregate - sum(other parts)` rather than against the aggregate directly. It is the right
shape for the dominant-part problem -- Turkey against an implied max of 46,800 - 5,800 =
41,000 is exactly at its ceiling and correctly kept, while the same 41,000 filed under Syria
faces an implied max of 5,800 and is correctly rerouted.

**Measured on Helene, it is much worse than the plain aggregate reference: 2.590 vs 0.591.**

A one-pass version does nothing at all (5.228, and the control says it is no better than
thinning), because the other parts are *themselves* contaminated -- North Carolina's stream
holds 250 and Florida's holds 300, so the raw sum of parts exceeds the whole and every
implied maximum clamps to zero. A two-pass version (gate against the raw aggregate first,
then compute implied maxima from what survives) recovers most of that but still trails
badly: Florida stays at 9.437 because early in the event the parts' running sum already
meets the aggregate, the implied maximum clamps to zero anyway, and Florida's 300 walks
straight through.

**So the refinement that Turkiye motivated makes Helene worse, and Turkiye still cannot test
it** -- the implied maximum needs *independent* observations of the whole, and Turkiye-Syria
has none. Declaring the hierarchy does not manufacture the data. Getting those observations
needs a re-extraction that labels joint-scope figures `__aggregate__`; the aliases for that
now exist, but the original `--associate envelope` run's `--event-model` is not recorded in
the pre-registration, so reproducing its 91 observations is a separate task.

Net: the declaration is correct and worth keeping -- it is what tells the gate which streams
are parts at all, previously hardcoded. The reference it was built to enable is a measured
negative. **Plain aggregate remains the recommended reference wherever an aggregate stream
exists, and no gate is possible where one does not.**

### 25.6 Honest scorecard

On the event it was designed against, a 9x improvement with a clean control and a stable
plateau. On held-out data, a 3.7x win on the contaminated stream and a **2.3x loss on the
clean one**. Both belong in any writeup; the Helene number alone is the misleading half.

Remaining caveat: the ratio was chosen after seeing Helene's contaminated values. The
plateau mitigates that and does not remove it. And 0.591 is 9x better than catastrophic,
not good in absolute terms.

What the gate does **not** address, visible in the same audit: North Carolina's 1400 is
**Hurricane Katrina's** toll quoted inside a Helene article, its 250 is a Typhoon and its 230
is Hurricane Milton -- *cross-event* contamination, not scope. Turkiye's fifteen 17,500s are
the 1999 Izmit toll, a *temporal* error the date filter handles. A magnitude gate removes all
of these for the wrong reason: it drops them because they are large, not because they belong
to another event, so it would keep any cross-event figure that happened to be small.

*(Corrected 2026-08-11: this section originally called the 1400 a non-casualty number "in the
same family as the 140-mile distance". Context audit shows it is Katrina's death toll. Same
symptom, different cause, and a different fix -- see §27.)*

## 26. Summarizer-as-segmenter: tested before building, and it is not the answer

Proposal: a purpose-built model emitting self-contained bullets so number-to-place binding
becomes local, guarded by a verbatim-number check. Tested with **hand-written** bullets, so a
negative costs nothing and a positive specifies exactly what the model must produce.

**Finding the real cases killed the framing first.** The motivating example -- "120 in NC, 17
in TN, 227 total" -- **does not occur in this corpus**. The Helene feed has 5 multi-number
casualty sentences and none has that shape. What is actually there: a 140-mile distance, a
30-year career, a town's 6,000 population, 30.5cm of rain, the year 2004, and four deaths
belonging to **Hurricane Ivan**. The failure modes are non-casualty numbers and cross-event
leakage, not aggregate splitting.

| arm | correct | false positives | fabrications |
|---|--:|--:|--:|
| raw text | **3/5** | 1 | 0 |
| free bullets | 2/5 | 3 | **2** |
| extractive bullets | **3/5** | 1 | 0 |

Restructuring is neutral at best; the free variant actively harms.

**The durable finding is that the guard and the summarizer are in direct tension.** The
summarizer's highest-value act is normalizing implicit prose into digits -- "they died
together" -> "2 people died" -- and that is precisely what a verbatim-number guard must
reject. Constraining it to extractive-only resolves the conflict and costs real recall: a
death that is never quantified becomes unreachable. A second hard constraint: it must emit
**digits, not words**, since "One firefighter died" extracts nothing.

The result is what redirected the work to §25 -- raw extraction's one error on that sample is
a *scope* error, not an attachment error.

## 27. Type vs event energies, and the negative-sampling axis they expose

§25 removed the scope errors. The context audit of what remained (§26 and the 2026-08-11
recount) left two error classes, and the question was whether a single mechanism -- ranking
by the model's own type/association energies -- handles both. It does not, and the way it
fails is more useful than the way it succeeds.

### 27.1 Unit errors: solved, and the competitor set is the whole design

Score each casualty span under `death toll` against competing type queries and reject on a
negative margin. Against **physically incompatible** types only -- measurement, duration,
money:

    real unit errors   -0.996  -0.993  -0.993  -0.972     (9 mph, 25 mph x2, "two-day")
    everything else     0.000   0.001   0.003   0.007 ...

**4/4 caught, 0/83 false positives**, stable for any threshold in `[0.05, 0.9]`. There is no
tuning: the gap is the width of the scale.

**`quantity` must not be a competitor.** Described as "a count of things that are not
people", it is semantically *adjacent* to a death toll rather than incompatible with it.
Including it takes false positives from 0% to **21.7%**, rejecting genuine tolls -- "Dozens"
(0.96 quantity), "300" (0.61 quantity, 0.0 death toll), "three". So: **compete against what
a death toll physically cannot be, not against what it resembles.**

That rule is an inference-time patch. §27.3 argues it is really a training-time gap.

### 27.2 Cross-event: not solved, and not by any of three signals

The type energy catches **0/11** here, exactly as predicted -- the type is *right*: Hurricane
Katrina's 1,400 scores `death toll` 0.95. What is wrong is which event owns it. Three ways to
ask, with generic words ("hurricane", "storm") excluded so only NAMED competitors count:

| signal | catches | false positives |
|---|--:|--:|
| A. nearest named event is a competitor | 3/11 | 32.5% |
| B. only competitors named in the window | 3/11 | 31.3% |
| C. record head binds a competitor as the event | 2/11 | 26.5% |

Not shippable at any threshold: Helene articles routinely name Milton, Katrina and historical
storms for comparison, so ~30% of *genuine* observations are flagged.

Per distinct case, the detail matters more than the totals: Mexico's Hurricane John is caught
(`bound='John'`), Katrina is caught in one of two instances, the Typhoon binds to the generic
"hurricane" and is missed, and **Bosnia's 16 is structurally invisible** -- Bosnia is a place,
not a named storm, so an event probe cannot see it at all.

One incidental result worth keeping: the probe **corrected its own labels**. Six "230"s were
labelled cross-event by an audit that keyed on `(span, value)`; all six bound to Helene, and
Helene's toll genuinely reached 230. The measurement was right and the label was wrong.

**Why this half fails where the other succeeded.** A type is a property *of the span*. Event
membership is a *relation* between the span and something else in the document -- and there
is no supervision for it: `RECORD_TASK_TYPES` is `("json_structures",)`, so events never
reach the anchor machinery and no trained trigger->argument binding exists to exploit. The
probe was asking an untrained capability to discriminate.

### 27.3 The real finding: hard negatives are mined on ONE axis

The codebase already mines hard negatives -- `select_hard_negative_candidates`
(`losses.py:180`), wired at `model.py:660`, `hard_negatives_per_positive: int = 5`. It ranks
the highest-scoring **negative spans per query**.

Both failures above need a different axis:

| axis | question | status |
|---|---|---|
| **span** | for query *q*, which SPANS are hard negatives? | ✅ built |
| **query** | for span *s*, which TYPE QUERIES are hard negatives? | ⛔ missing |
| **instance** | for a casualty, which EVENT INSTANCE owns it? | ⛔ missing |

`quantity` vs `death toll` is a **query-axis** gap: the model has never been taught that
boundary, because negatives are drawn over spans, never over sibling type queries. Dropping
`quantity` at inference is therefore a workaround for a training-time omission -- and an
expensive one, since with it the rule caught 6 further cases.

Cross-event is an **instance-axis** gap, and harder: it needs negatives of the form "this
casualty belongs to Helene, not to the Katrina mentioned two sentences away".

### 27.4 Candidate losses, and why GISTEmbed first

Practitioner experience recorded because it is not derivable from this repo:

- **MultipleNegativesRankingLoss** — in-batch negatives; **normalizes scores cross-lingually**,
  which is directly relevant here, since the calibration complaint in §27.1 is exactly that
  GLiNER2 scores each `(query, span)` through an independent sigmoid with no softmax across
  types, so margins are not commensurable. MNRL would fix that by construction. **But it is
  hard to train** — recorded from prior use, and a reason not to lead with it.
- **BatchHardTripletLoss / OnlineContrastiveLoss** — mine the hardest negative per anchor.
  Cheap, but they assume the hardest negative is a true negative.
- **GISTEmbedLoss** — uses a guide model to detect in-batch negatives that are *actually
  positives*, and drops them from the loss instead of penalizing them.

**GISTEmbed is the right starting point for this specific failure.** `quantity` is not a
false competitor in general — it is genuinely correct for "1.2 million homes" and genuinely
wrong for "300 died". A loss that penalizes it unconditionally would teach the model that
`quantity` never applies; GIST's guide-filtering is precisely the mechanism for "this
negative is valid *here* but not *there*".

Implementation note: this is not an import. sentence-transformers losses operate on
bi-encoder `(anchor, positive[, negative])` tuples; GLiNER2 is a query-span scorer with BCE
over `(query, candidate)` pairs. What transfers is the *mechanism* — guide-filtered negative
selection — and its natural insertion point is `select_hard_negative_candidates`, which
already ranks and selects negatives. A query-axis variant with a guide-model veto is a
surgical change to an existing function rather than a new training path.

### 27.5 The check ran: the query-axis negative exists, and it is abundant

Before wiring a loss, confirm the boundary is actually present in the *training* data rather
than only in 83 Helene observations. `tools/train/probe_query_negatives.py` scores 250 GOLD
death figures from `casualty_multi_loc.train.jsonl` under the same competitor set:

    a competing type OUTSCORES `death toll` on   71/250 = 28.4%
        quantity      64   90.1%
        measurement    7    9.9%
    margin (death toll - best rival)   p10 -0.310   median +0.283   p90 +0.809

**28.4% of genuine training positives are outscored by a sibling type**, and **90% of it is
`quantity`**. So there is a real, abundant, minable boundary — and the inference-time finding
reproduces independently on 3x the sample and a different corpus. GIST has plenty to bite on.

It also confirms *why* GIST rather than a plain contrastive loss: `quantity` wins on 64 of
250 genuine death tolls, and it is simultaneously the CORRECT answer for "1.2 million homes".
A loss that penalizes it unconditionally would teach the model that `quantity` never applies.
Guide-filtering is the mechanism for a negative that is valid in one context and not another.

**And it corrects the §27.1 claim.** The shipped rule — physically incompatible competitors
only — falsely rejects **9/250 = 3.6%** of genuine death tolls on this corpus, not the 0/83
measured on Helene. `measurement` beats `death toll` on 7 real casualty figures. The Helene
zero was small-sample luck; **3.6% is the number to quote**, and it is the standing cost of
the inference-time workaround until the training-time fix lands.


### 27.6 What can serve as the GIST guide — gold cannot, and the corpus says why

GIST needs a guide that answers, per instance, "is this sampled negative actually a positive
here?". The cheapest candidate was the gold labels themselves. Measured on
`mix_natural.train.jsonl` (84,280 records, 42,609 carrying entity annotations):

    distinct entity types                                    17,128
    entity mentions                                         357,849
    number-ish types                       1,649 distinct / 25,101 mentions
      Casualties and Losses 1180 | Quantity 129 | Measurement 104 | Currency 98 | dates ...
    records declaring BOTH a casualty-ish AND a quantity-ish type    99 / 42,609 = 0.23%

**Gold cannot be the guide.** The corpus is zero-shot with a per-record type vocabulary --
each record declares its own handful out of 17,128 -- so it essentially never annotates the
correct type and the competing type on the same span. At 0.23% coverage there is no signal
about *when* `quantity` is genuinely right.

Three consequences, and the second is the useful one:

1. **A real guide model is required** — the actual GIST design, not a shortcut.
2. **Those 99 records are a gold validation set for the guide itself**, and they are exactly
   on point: the pairs are things like `Number of Vehicles Involved in the Crash` +
   `Casualties and Losses`. Both are counts; one is of people. Any candidate guide can be
   scored on whether it says *Quantity is right for the vehicle count and wrong for the death
   count*. That is a cheap, direct test of the guide before it is trusted to filter anything.
3. **The negatives themselves are free and abundant.** Because every record declares its own
   type set, a type query from another record in the batch is already a query-axis negative.
   That is precisely MNRL's in-batch mechanism -- which is why MNRL is the natural second step
   once GIST establishes which of those in-batch negatives are false.

`entity_descriptions` is present on 30,064 records (often with anonymized `e_0..e_n` names and
real descriptions), so a description-level guide is available. Caveat to settle by
measurement: a description-level guide is TYPE-level, while GIST's veto is INSTANCE-level --
"quantity is close to death toll in general" is a weaker claim than "quantity is correct for
*this* span". The 99 records are how to find out whether the weaker signal suffices.


### 27.7 Guides, tested on the 99 -- the self-guide clears the bar

Of the 99 records naming both a casualty type and a competing count type, **40** carry a
digit-bearing span of each kind, and they test a HARDER boundary than the inference probe
did: the corpus's real rivals are largely counts OF people -- `Number of Evacuated People`,
`Number of Rescued People` -- so the distinction is the verb, not the noun.

Asking the model itself, with the record's own type names:

    casualty span scored higher under its own type   37/40 = 92.5%
    rival span scored higher under its own type      36/40 = 90.0%
    BOTH directions right on the same document       33/40 = 82.5%   (chance = 25%)

**The self-guide works**, against my prediction that a model cannot referee a boundary it
does not know. Consequence: GIST needs no external guide model or separate embedder, which
removes a dependency and a calibration problem.

The failures are mostly not model failures. `'destroyed over 50 residential houses'` is gold-
labelled `Casualties and Losses` and the model says `Number of Damaged Houses` 0.96 -- the
model looks right and the gold looks odd. `'13 of the injured'` is annotated under *both*
types. `'more than 200 have been injured'` competes with `Number of Aftershocks` over the
same numeral 200, which is a span-identity problem rather than a type problem.

### 27.8 The free win was mostly one bad description, and it was already excluded

If the model separates specific count types at 82.5%, the vague catch-all becomes the
suspect rather than the model. Re-running the 250-positive probe with concrete rivals in
place of `quantity` = "a count of things that are not people":

| competitor set | rival outscores `death toll` | median margin | p10 |
|---|--:|--:|--:|
| generic (with `quantity`) | 71/250 = **28.4%** | +0.283 | -0.310 |
| specific rivals | 35/250 = **14.0%** | **+0.709** | -0.125 |

Confusion halves and the median margin multiplies by 2.5. **But that is entirely the
`quantity` effect.** Holding the rival set to physically-incompatible types and changing only
the wording -- `measurement`/`duration`/`money` to `wind speed`/`rainfall`/`distance`/
`elapsed time`/`cost` -- moves false rejects **9/250 -> 8/250, 3.6% -> 3.2%**. Negligible.

Since the shipped rule already excluded `quantity`, the free win mostly **confirms the
shipped rule rather than improving it**: still 4/4 unit errors and 0/83 false positives on
Helene, now at 3.2% false-reject on training positives. The descriptions are kept anyway
because "avoid a negatively-defined catch-all" is now measured rather than asserted.

**What remains is the real target for GIST.** With specific rivals the residual confusion is
`people evacuated` 28 of 35 wins -- **11.2% of genuine death tolls**. That is "N people
killed" against "N people evacuated": both counts of people, separated only by the verb. No
description fixes it, and it is exactly the hard negative a guide-filtered loss exists for.
