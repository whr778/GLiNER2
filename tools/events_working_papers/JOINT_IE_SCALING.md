# joint_ie × Head-Init Scaling on the Boundary Head — Design

Status: design + build. Date: 2026-08-07. Companion to [[BOUNDARY_DECODE_AND_EKF.md]]
(the boundary decode map) and the mmBERT head-init scaling finding
([[mmbert-head-init-finding]]). Sibling line to [[EKF_MHT_DESIGN]] — a *different* route to
dense document-level extraction: a global typed-constraint decode instead of a tracker.

## 1. Thesis

Does the dormant **joint_ie global beam** (typed constraints + `Calibrator`), wired to the
**boundary** head (no span 20-cap), improve dense **document-level relation** extraction
(Re-DocRED), and how does that interact with base-training **data volume** (head-init)?

## 2. Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Architecture = **boundary** (no span, no 20-instance cap) | DECIDED |
| 2 | Downstream target = **Re-DocRED** (dense relations, where the cap bites) | DECIDED |
| 3 | Bases **retrained `from_encoder`** (mmBERT), NOT span | DECIDED |
| 4 | joint_ie **wired to boundary** (new adapter; the contract already exists) | building |
| 5 | **Phase A = decode-only** (paired greedy vs beam); **Phase B = joint training** only if A is positive | DECIDED |
| 6 | Base mix = event corpora **+ relation-rich corpora** (warms the relation head; also pushes past 100K free) | DECIDED |
| 7 | Sizes {10K,40K,100K} from the existing 100,080 pool; **>100K via a new config** once corpora are added; **NO LLM generation** (100K synthetic ≈ $400-860 batch) | DECIDED |

## 3. The wiring (joint_ie → boundary) — the net-new work

`candidate_scores.py` already defines the **architecture-agnostic contract**:
`CandidateScoreSet` → `candidate_score_set_to_problem` → `JointProblem` → `BeamOptimizer`
+ `TypedEndpoints`/constraints + `Calibrator`. The **span** direction was built
(`score_lattice_to_candidate_score_set`; dense width lattice → sparse). Progress:

- ✅ **`boundary_candidates_to_candidate_score_set`** — one sample's `CandidateTensorBatch`
  (`indices`/`pair_logits`/`valid_mask`/`query_mask`) → `MentionScore`s, typed by the query's
  schema `role_name`. Duck-typed; unit-tested. (commit `508880d`)
- ✅ **`boundary_relation_pairs_to_edges`** — a `RelationPairBatch` + per-pair relation logits
  → `ScoredRelationEdge`s. The pair batch's `head_keys`/`tail_keys` are `(role_name, start,
  end)` = the mention keys, so edges reference the nodes directly; unit-tested. (commit
  `332a86a`)
- ⬜ **Decode wiring** — `BoundaryExtractor` + a `--joint-decode` flag: build `query_types`
  from the layout, run both adapters + the relation scorer, `candidate_score_set_to_problem`
  → `BeamOptimizer`, format results.
- ⬜ **Integration test** on a shipped boundary checkpoint (greedy vs beam parity + a
  constraint case).

Reuses the entire optimizer/constraint/calibration stack — this is the contribution, not a
rebuild. The two adapters (the tensor→contract mapping) are the crux, and they're in.

## 4. Experiment (Phase A — decode-only)

- **Bases:** boundary `from_encoder` mmBERT-base, sizes {10,40,100}K on the event+relation
  mix (Re-DocRED / any DocRED-derived set **excluded** — leakage). Config takes an arbitrary
  size list.
- **Warm-start Re-DocRED** from each base (identical recipe; only `pretrained` differs).
- **Decode arms** per model: (a) boundary greedy set-prediction; (b) boundary + joint_ie beam.
- **Metric:** Re-DocRED relation-strict micro-F1 (+ F1-ign).
- **Curves:** F1 vs base volume × decode arm → elbow + whether the beam lifts it, and where
  (low-data = compensating weak head-init, vs high-data).

## 5. Data (surveyed 2026-08-07)

- Event pool = **100,080** records (10 corpora). 10/40/100K = nested subsamples
  (`build_scaling_mix.py`, seed 42). These corpora carry **no relations** → they don't warm
  the relation head on their own (hence the relation add).
- **Non-leaking relation corpora on disk**: `sentence_rex` **34,314**, `bio_ner_relations`
  **2,085**, `biored` 308, `scierc` 265 ≈ **~37K**. → base mix ≈ **137K**, which **reaches a
  >100K point for free** (no generation) *and* warms the relation head. To verify:
  `sentence_rex` provenance (must not be DocRED-derived).
- **EXCLUDE `docred` (83,951)** — Re-DocRED re-annotates the *same documents*; including it
  leaks the downstream. This is the one large relation corpus, so a true 200K without it
  would still need generation.
- **Re-DocRED is ready**: `data/redocred.{train}.jsonl` (3,053 train) already in the GLiNER2
  `input/output` training format — no build needed.
- **No generation**: 100K synthetic multi-task docs ≈ $430 (haiku) / $860 (sonnet) batch —
  not worth one curve point when relation corpora get us to ~137K free.
- **Scaling sizes**: {10K, 40K, 100K} + a **~137K** point (event+relation mix); config takes
  an arbitrary size list.

## 6. Cost / time

- Data + wiring: **$0**.
- A100: 3 bases + 3 Re-DocRED fine-tunes + evals ≈ **6-10 hr ≈ $12-20** (reuse the casualty
  instance after that job frees it).

## 7. Phase B (deferred — only if A is positive)

Joint training: put the joint_ie beam **in the loss** via the boundary model's existing
**detached-association + differentiably-recomputed-scores** idiom (`records.py`,
`proposal.py`) — a structured-prediction objective. Feasible because that idiom already
exists; a training-loop change, not a bolt-on. Answers: does training-*for* the beam beat
decoding-*with* it?
