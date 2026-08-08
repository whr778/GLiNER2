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
| 2 | Downstream target = **Re-DocRED + RAMS** (dense relations *and* document-level event arguments) | DECIDED (2026-08-08) |
| 3 | Bases **retrained `from_encoder`** (mmBERT), NOT span | DECIDED |
| 4 | joint_ie **wired to boundary** — mentions + relation edges | **DONE** |
| 4b | joint_ie must support **structures, relations, AND events** in the beam | **DECIDED (2026-08-08) — BLOCKS Phase A** |
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
- ✅ **`joint_decode`** — end-to-end composition: candidates + relation pairs/logits +
  constraints → both adapters → `candidate_score_set_to_problem` → `BeamOptimizer` → the
  selected node/edge solution. Unit-tested from synthetic boundary outputs. (commit `e8f2bad`)
  **The joint_ie side is now complete.**
- ✅ **Engine plumbing** — `_decode_joint` on `BoundaryExtractor`, gated by
  `boundary_head.decode_mode` (`"greedy"` default | `"joint"`) + `joint_beam_width`.
  A `_layout_from_ext_specs` helper builds the one real `QueryLayout` from
  `core["ext_specs"]` — **not** `batch.query_layouts`, which only the `fast_routing`
  path populates — and both the mention `query_types` and the pair `head_keys`/
  `tail_keys` are typed from it. `_relation_pairs_and_logits` is now shared by the
  greedy and joint paths (returns *raw* logits; each caller applies its own
  temperature). The empty-`QueryLayout` call is gone, so both paths are un-forked.
  **Key consistency proven, not assumed** — same fixture, real vs empty layout:

  | layout | `head_keys` | edges kept |
  |---|---|---|
  | real | `('person', 0, 1)` | **1** |
  | empty (old) | `('0', 0, 1)` | **0** |

  Mention keys are `('person',0,1)`/`('org',3,4)`; the empty layout silently pruned
  *every* edge. This is the failure the wiring had to avoid, and it is now guarded.
- ✅ **Integration test** — `tests/models/boundary/test_joint_decode.py` (5 tests, tiny
  encoder, no download): the layout builder types by `role_name`; the joint decode links
  the relation to its mention nodes (the crux regression guard); joint == greedy on an
  unambiguous case; `decode_mode` defaults to `"greedy"`; the flag runs through the
  public `extract_relations` path.
- ⛔ **BLOCKER for Phase A — events are not in the beam.** `JointProblem` models only
  `NodeCandidate` (mentions) + `EdgeCandidate` (relations); there is no record/instance
  concept. In joint mode today, `_decode_records` still runs *before* the joint branch, so
  event/structure output is produced — but **greedily**, bypassing the beam entirely, and
  record-field mentions get pulled into the beam as nodes only to have their selections
  discarded. With **RAMS** now in the warm-start (decision 2), events are on the evaluated
  path, so this blocks. Design in §3b. *(An earlier note here judged this non-blocking on
  the assumption of a Re-DocRED-only downstream — that assumption is void.)*
- ⚠ **Arm-comparability caveat (must settle before Phase A):** the joint path threads the
  engine `threshold` through as `mention_threshold`, but four greedy-side threshold
  behaviours are **not** mirrored:
  1. **adaptive thresholding** (`boundary_settings.adaptive_threshold`) — greedy-only;
  2. **null-abstention** (`abstention_threshold` on `null_logits`) — greedy-only;
  3. **per-relation-type thresholds** from `metadata["relation_metadata"]` — greedy looks
     them up per relation; `candidate_score_set_to_problem` centers edges at a fixed
     `decision_threshold=0.5`. Moot for a single-threshold Re-DocRED eval, but it is the
     same family of decisions.
  4. **`decode_group`'s three record thresholds** (`anchor_threshold`, `field_threshold`,
     `object_threshold`) — greedy-only, and they become live the moment records enter the
     beam (§3b).

  The arms are therefore not yet threshold-identical. Decide whether to port these to the
  joint path or disable them in both arms **before** reading the greedy-vs-beam curve —
  otherwise the curve confounds decode strategy with thresholding.

<details><summary>Original engine-read mapping (superseded by the above)</summary>
  - **Relations ARE wired** (not a blocker): `enable_relations` in config
    (`model.py:1006`) builds `relation_pair_generator` + `relation_scorer`; `_decode_relations`
    runs. The public-api e2e test just never enabled relations, so a relation-enabled boundary
    model (config + a relation schema) is the fixture.
  - **Hooks**: `query_types` from `core["ext_specs"][qid]["field_name"]`; reuse the pairs +
    logits `_decode_relations` already computes; format token spans → char offsets via
    `token_boundaries_to_character_offsets` / `_format_spans`; gate behind a `decode_mode`
    setting (default off).
  - **QueryLayout gotcha**: `_decode_relations` passes an *empty* `QueryLayout`, so
    `head_keys`/`tail_keys` types are `str(query_id)`. Build a real layout
    (`QuerySpec.role_name`; `processing/layouts.py` builds one from the schema).
  - **Key consistency (resolvable, mechanical)**: boundary types relation endpoints by the
    head/tail query `role_name` and entity mentions by the entity query `role_name`. Build ONE
    real `QueryLayout` and type BOTH the mention adapter's `query_types` and the pair
    `head_keys`/`tail_keys` from it (same source) → a head candidate from query *q* keys as
    `(role_name_q, start, end)` on both sides, so edges reference the mention nodes by
    construction. (An earlier note here over-called this a design blocker; it is not — one
    layout used consistently resolves it.)
  - So the whole hook is mechanical: build the real layout, type mentions + edges from it,
    reuse pairs/logits, `joint_decode`, char-offset format, gate the flag.

</details>

Reuses the entire optimizer/constraint/calibration stack — this is the contribution, not a
rebuild. The two adapters (the tensor→contract mapping) are the crux, and they're in.

### Decode-wiring integration notes — SUPERSEDED (all implemented)

> **Historical.** Every item below is done and shipped; the ⚠ GOTCHA in particular is
> **fixed** (no empty `QueryLayout` remains in the engine). Kept only as the record of how
> the hook was scoped. For current behaviour read the code and the ✅ entries above.

<details><summary>Original scoping notes</summary>

Hooks in `BoundaryExtractor._extract_from_batch` (`models/boundary/engine.py`):
- **query_types** from `core["ext_specs"][i]` (per-query `field_name`/`roles`) → `query_id →
  role_name`, passed to `boundary_candidates_to_candidate_score_set`.
- **edges** reuse the pairs + logits already computed in `_decode_relations`
  (`relation_pair_generator.generate` + `relation_scorer`) → `boundary_relation_pairs_to_edges`.
- **⚠ GOTCHA**: `_decode_relations` calls `generate(..., [QueryLayout(queries=())], ...)` — an
  **empty** layout — so `head_keys`/`tail_keys` types are `str(query_id)`, **not** role names.
  Fix: build a real `QueryLayout` from `ext_specs` and pass it to `generate` so the endpoint
  keys carry `role_name` (matching the mention keys); else `TypedEndpoints` constraints won't
  bind. (Type mentions by the same source.)
- **constraints** = `TypedEndpoints(rel, head_types, tail_types)` from the relation schema
  (`rel_specs`).
- **format**: `BeamOptimizer(...).optimize(problem)` → solution nodes/edges (typed token
  spans) → char offsets via `start_map`/`end_map` (as greedy does) → `{entities:{type:[…]},
  <rel>:[(head,tail)]}`.
- **flag**: a `decode_mode`/`--joint-decode` setting gates a new `_decode_joint` beside the
  greedy entity+relation decode (default OFF → zero risk to the shipped path).
- **test**: build a `BoundaryExtractor` per `tests/models/boundary/test_end_to_end_real_
  deberta.py`, run greedy vs joint on a simple + a constraint case.
  *(Shipped as `tests/models/boundary/test_joint_decode.py` on the **tiny-encoder** fixture
  instead — the check is structural/constraint-level on an untrained model, so the offline
  deterministic fixture serves it better than a real-deberta download.)*

</details>

## 3b. Structures, relations and events in the beam (the Phase A blocker)

Status: designed 2026-08-08, building. Unblocks decision 4b.

### The finding that shapes it: no new candidate class is needed

The obvious design — add a `RecordCandidate` to `JointProblem` and teach the beam a third
candidate class — is **not** required. Reading `optimizers/beam.py` + `base.py` shows the
contract already expresses instances:

- The beam **expands over edges**, pulling nodes in as endpoints; `_finish_nodes` then adds
  any leftover positive-score nodes.
- `EdgeCandidate` already carries **`slot`**, **`hypothesis`** and `count_alternative`, with
  `exclusion_keys = ("slot", hypothesis, count_alternative, slot)`. The compiler already
  emits `UniqueRelationSlot(name, "slot")`.
- `base.py:90-100` runs whole-result **`validate(relations, nodes)`** hooks after
  construction, in *both* optimizers — so constraints are **not** rejection-only. "At least"
  semantics (required roles, required anchor) have a home.

So an **event is a trigger node plus role edges**:

| record concept | joint_ie realization |
|---|---|
| event instance | the **trigger node** (a real mention candidate) |
| role assignment | `EdgeCandidate(head=trigger, tail=argument, slot=role)` |
| instance identity | `hypothesis` = **trigger node id** |
| scalar role cardinality | falls out of `edge_conflicts` via `exclusion_keys` — free |
| required roles / anchor | a `validate`-hook constraint |
| relation | unchanged — plain edges, as shipped |

Zero contract change for events (`natural` mode). This is the paper's thesis applied to
itself: reuse the optimizer/constraint stack rather than rebuild it.

### Decisions

| # | Decision | Why |
|---|---|---|
| A | `hypothesis` = **trigger node id**, not a synthetic instance index | `EdgeCandidate.key` is `(relation_type, head, tail, slot, count_alternative)` — `hypothesis` is **not** in it, so two instances sharing a trigger would collapse their edges. Trigger-as-identity makes instance identity and `head` coincide, so nothing collapses. |
| B | **scalar** role → `slot=role`; **list** role → `slot=None`, role in `relation_type` (`f"{task}::{role}"` uniformly) | `slot=role` gives scalar cardinality free, but would wrongly block the 2nd filler of a `ZERO_OR_MORE` role. Uniform `relation_type` means the output formatter never parses slots. |
| C | Edge utility: **scalar** = `logit_c - logit_ABSENT`; **list** = `center_logit(logit_c)` | `_scalar_field_nll` is `log_softmax` over {ABSENT, candidates} (competitive), `_list_field_bce` is BCE over candidates only. Centering a softmax logit would be wrong; this silently determines beam behaviour. |
| D | v1 **ignores `object_logits`** — the trigger mention score is the existence evidence | Keeps one score per decision. Adding the centered per-instance logit uniformly to that instance's role edges is the alternative; deferred as a **Phase-B calibration item**, not silently dropped. |
| E | Source is **`RecordGroupOutput`** (raw `object_logits`/`assign_logits`/`field_spans`), never `decode_group` output | Adapting `DecodedRecord` would re-rank already-greedy decisions — the exact trap the relation path avoided. |

### Known v1 behaviour, recorded not hidden

- **A sub-threshold trigger kills its whole event.** `candidate_score_set_to_problem` drops
  nodes below `mention_threshold`, then drops edges referencing them — so an event whose
  trigger mention scores 0.4 vanishes even if the record head is confident. Consistent with
  how relations already behave, but a live failure mode for RAMS. Belongs to the
  arm-comparability family above.
- **`latent` mode is deferred, not broken** — it keeps working through the greedy record
  path and is simply not exercised in the beam. Asserted by a test.

### Increments — status

1. ✅ **Adapter** `boundary_record_groups_to_role_edges` — role edges from the raw
   `RecordGroupOutput`, endpoints keyed `(role_name, start, end)` from the **same layout**
   as the mention adapter. **Key probe in place**: `test_miskeyed_query_types_drop_every_
   role_edge` types endpoints from a different source and asserts all edges are pruned —
   the §3 relation crux, guarded for records.
2. ✅ **Engine** — `_record_groups` factors the raw lattice out of `_decode_records`;
   `_decode_joint` feeds role edges + instance nodes into `joint_decode`;
   `_format_joint_records` rebuilds instances by grouping on `hypothesis` and emits
   greedy's exact shape. Joint mode **skips** the greedy record pass (else double-emit),
   and role edges are excluded from relation output (matched by `::`).
3. ✅ **Anchorless structures** — `boundary_record_instance_nodes` synthesizes one node per
   instance, scored by `object_logits` (the one place decision D's per-instance signal is
   genuinely needed — there is no trigger mention to carry it), injected via
   `candidate_score_set_to_problem(extra_nodes=...)`. Proven load-bearing:
   `test_anchorless_roles_are_pruned_without_the_instance_node`.
   **⚠ Caveat:** a synthetic span participates in `EntityOverlapPolicy` checks under a
   non-`allow` policy. The engine never sets that policy on the joint path, so this is
   recorded, not engineered around.
4. ✅ **Tests** — 10 across `tests/joint_ie/test_record_role_edges.py` +
   `tests/models/boundary/test_joint_records.py`, incl. multi-valued roles keeping every
   filler (guards decision B), two triggers not merging (guards decision A), and no
   double-emission through `_extract_from_batch`.
5. ⬜ **`RequiredRoles`** constraint via the `validate` hook + registration in
   `_CONSTRAINT_TYPES` (`constraints.py:304`), with a compiler hook beside the existing
   `UniqueRelationSlot` emission. **Not yet built** — without it an instance can be emitted
   with a required role unfilled, which the greedy path also permits, so this is a quality
   improvement rather than a parity gap.

**Decision 4b is met**: structures, relations and events all decode through the beam.

## 4. Experiment (Phase A — decode-only)

- **Bases:** boundary `from_encoder` mmBERT-base, sizes {10,40,100}K on the event+relation
  mix (Re-DocRED / any DocRED-derived set **excluded** — leakage). Config takes an arbitrary
  size list.
- **Warm-start = Re-DocRED *and* RAMS** from each base (identical recipe; only `pretrained`
  differs). Two downstreams, not one: Re-DocRED exercises dense **relations**, RAMS exercises
  document-level **event arguments** (roles dispersed across sentences). Decision 2.
- **Decode arms** per model: (a) boundary greedy set-prediction; (b) boundary + joint_ie beam.
- **Metrics:** Re-DocRED relation-strict micro-F1 (+ F1-ign); RAMS **argument F1** (the
  head-init-sensitive number from [[mmbert-head-init-finding]] — the existing 10k/40k/100k
  arg curve 0.050/0.115/0.158 is the greedy-arm baseline to beat).
- **Curves:** F1 vs base volume × decode arm, per downstream → elbow + whether the beam lifts
  it, and where (low-data = compensating weak head-init, vs high-data).
- **Consequence:** the RAMS arm only means anything once events are in the beam (§3b);
  until then arm (b) on RAMS is identical to arm (a) plus noise, because record decoding
  bypasses the beam entirely.

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
