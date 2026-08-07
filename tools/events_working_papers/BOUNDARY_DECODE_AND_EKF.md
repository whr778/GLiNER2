# Boundary Decode Path + Where the EKF/MHT Plugs In

Status: architecture map from a read-only audit (no code changes, no GPU). Date: 2026-08-07.
File:line references are against the code as of this date and may drift.
Supersedes the boundary-relevant parts of [[KALMAN_BEAM_SEARCH_EXPLORATION.md]]
(which was written against the span `global_decode.py` beam).

## TL;DR

- **`joint_ie` is NOT the boundary decoder.** It is a dormant, test-only, *span-bound*
  subsystem (its `RawScorer` calls span-only methods). The beam you remembered lives
  there, but it can't decode a boundary checkpoint without an unbuilt adapter.
- **The boundary model has its own live decode:** `BoundaryExtractor` → `_decode_records`
  / `_decode_relations`. Record decode is **DETR-style set prediction** (fixed instance
  queries), which is what removes the span 19-instance cap — not the count log-rate.
- **Decode is greedy set selection, not Hungarian.** `linear_sum_assignment` appears
  only in the *training* bipartite-matching loss.
- **The cross-chunk merge is shallow and result-level** (`merge_chunk_results`): remap
  offsets + naive dedupe, with an optional span-era heuristic event beam
  (`global_decode.py`). This is the seam the EKF/MHT replaces.
- **The deep EKF needs one plumbing addition:** the boundary candidate *states*
  (`candidate_states [B,Q,C,H]`) die at the per-chunk decode; a stateful Kalman tracker
  needs them carried forward to the cross-chunk association step.

## 1. Two decode subsystems (only one is live)

`AutoExtractor` dispatches by `config.architecture` to `SpanExtractor` / `BoundaryExtractor`
(`gliner2/inference/engine.py:50,57`). Neither touches `joint_ie`.

- **`joint_ie/` — dormant, test-only, span-bound.** No production code instantiates
  `JointIEEngine` / `BeamOptimizer` / `GreedyOptimizer` / `RawScorer`; the only
  non-test mentions in `gliner2/` are comments in `classification/*`. `RawScorer.batch_score`
  calls `model.compute_span_rep_batched` / `count_pred` / `count_embed`
  (`joint_ie/scoring.py:248,308,322`) — methods that exist **only** on the span model
  (`models/span/model.py`); the boundary model has none. So its beam + typed constraints
  (`TypedEndpoints`, `MaxRelationsPerHead`, …, `compiler.py:22-46`) + `Calibrator` are
  real but **cannot run on a boundary checkpoint** without the unbuilt
  `CandidateTensorBatch → JointProblem` adapter (`joint_ie/candidate_scores.py` sketches
  only the span direction).
- **boundary live decode.** `BoundaryExtractor._extract_from_batch`
  (`models/boundary/engine.py:91`) → `_decode_records` (`:279`) + `_decode_relations`
  (`:197`).

## 2. Boundary head outputs (the EKF's raw material)

`ExtractorOutput` (`models/outputs.py:213`), built at `models/boundary/model.py:553-565`:

- `candidates: CandidateTensorBatch` (`outputs.py:23-41`), shapes `[B,Q,C]`:
  `indices [B,Q,C,2]` (half-open token spans), `proposal_logits`, `pair_logits`
  (mention score), `valid_mask`, `query_mask`, and **`candidate_states [B,Q,C,H]`**
  (contextual embeddings — the natural continuous state for a Kalman filter; populated
  when the record head is enabled).
- `start_logits`/`end_logits`/`inside_logits` — boundary marginals (`heads.py:21-28`).
- `null_logits [B,Q]` — abstention (`model.py:404-406`).
- `count_log_rates [B,Q]` — per-query Poisson log-rates (`count_head = nn.Linear(dim,1)`,
  `model.py:198-203,408-410`); auxiliary, feeds adaptive thresholding
  (`predicted_count = torch.exp(count_log_rates).round()`, `model.py:829`).

## 3. Per-chunk record decode: DETR set prediction + greedy selection

`RecordSetDecoder` (`models/boundary/records.py:114`): a fixed bank of learned
**`instance_queries`** (`instance_embed [I,H]`) cross-attends the field-candidate states
(`_condition`, `:134`) → per-query `object_logits [B,I]` (is-a-record) + field-pointer
logits `[B,I,F,C]` (`forward`, `:156`; `RecordSetOutput`, `:101`).

`decode_group` (`records.py:675`) — **greedy, not assignment**:
1. Sort instances by object prob; keep those above `object_threshold` (`:695`).
2. Per field: scalar → softmax with a null column, pick best non-excluded (`:715-735`);
   list → sigmoid, take all above `field_threshold` (`:736-752`); `used_exclusive`
   enforces cross-instance exclusivity.
3. Latent/anchorless modes dedupe by `_dedup_key`, max score wins (`:756-762`).
4. `derive_count` (`:766`): **"count = number of selected instances — never predicted."**

`DecodedRecord` (`records.py:660`). `RecordHead` wraps the above:
`forward_groups_dense` (`:307`), `forward_group_dense` (`:463`), `forward_group` (`:548`).

**Hungarian is training-only.** `linear_sum_assignment` (`records.py:1001,1123,1220`)
lives in the "Record training losses" section (`:771+`) — the standard DETR
pred↔gold bipartite matching for the set-prediction loss, not decode.

## 4. Why the 19-cap is gone here

The span cap was the 20-way `count_pred` classifier (see [[COUNTING_LAYER.md]]). The
boundary record decoder removes it by construction: `RecordSetDecoder`'s docstring
(`records.py:119`) — count is input-dependent via query cross-attention,
*"well beyond the legacy 19-instance cap."* The ceiling is now the (configurable,
large) `instance_queries` budget `I`, and the realized count is whatever passes
`object_threshold`.

## 5. Cross-chunk merge — the seam (shared, architecture-agnostic)

Long docs use the shared runtime: `batch_extract_long` (`inference/runtime.py:1246`)
→ per-chunk `batch_extract` (`:1294`) → `merge_chunk_results` (`:1310`).

`merge_chunk_results` (`inference/chunking.py`) is **result-level, post-decode**:
1. `remap_result_spans` — chunk-local → document char offsets.
2. `_merge_result_dicts` — naive concatenate/dedupe per key (scalars collapse to best).
3. if `global_decode`: rebuild `event_extraction` via `assemble_events_global` — the
   **span-era OneIE heuristic beam** (`inference/global_decode.py`: trigger-IoU
   clustering + hand-set conflict penalty). Architecture-agnostic (operates on formatted
   dicts), so this is the only "beam" the boundary long-doc path uses today.

## 6. Where the EKF/MHT plugs in (and the plumbing gap)

The seam is steps 2/3 of `merge_chunk_results` — cross-chunk / cross-document
association. Two levels:

- **Shallow (works today):** replace `assemble_events_global` with an MHT tracker over
  the **formatted results** (spans, doc offsets, confidences). No model changes, but the
  Kalman "state" is only surface features — thin.
- **Deep (the real EKF/MHT — needs one addition):** track the boundary
  `candidate_states [B,Q,C,H]` as continuous state. **Gap:** those states are consumed at
  the per-chunk decode and are *not* carried into `merge_chunk_results` (which sees only
  formatted dicts). The deep path requires plumbing the candidate states forward to the
  cross-chunk association step, then: MHT = Hungarian **data association across
  chunks/time** + a per-track **EKF** over those states, with top-K hypothesis pruning
  (= the beam). This is exactly the "beyond-document, state evolves" regime of
  [[KALMAN_BEAM_SEARCH_EXPLORATION.md]] §1.

The EKF is therefore **not** a drop-in optimizer on an existing beam. It is a new
cross-chunk/streaming association layer that (a) supersedes the result-level merge + the
heuristic `global_decode`, and (b) needs the boundary candidate states surfaced to that
seam.

## 7. Corrections to earlier working assumptions

- ~~"Moving to boundary routes decode through `joint_ie`'s beam."~~ No — `joint_ie` is
  dormant + span-bound; boundary has its own live decode.
- ~~"The boundary decode is Hungarian assignment."~~ No — decode is DETR set-prediction +
  greedy selection; Hungarian is the training loss.
- ~~"count_log_rate drives the record count."~~ It feeds adaptive thresholding; the record
  count is derived from set-prediction selection (`derive_count`).

## 8. Open design questions (before building the EKF layer)

1. Shallow (surface-feature MHT on formatted results) vs deep (state-space EKF on
   `candidate_states`) — the deep path is the research contribution but needs the state
   plumbing.
2. What is the per-track **state vector** and its **dynamics**? (Embedding-space EKF? A
   task-specific state — trigger position, argument set, salience?) [[KALMAN_BEAM_SEARCH_EXPLORATION.md]]
   §1 warns the EKF only earns its keep with *real* continuous dynamics — beyond-document
   temporal tracking is where that holds.
3. Reuse `joint_ie`'s typed constraints / `Calibrator` *ideas* at the association step
   (without reviving its span-bound beam)?
4. Where does the association run — inside `merge_chunk_results`, or a new streaming
   engine that consumes per-chunk `BoundaryExtractor` outputs directly?
