# joint_ie × Head-Init Scaling on the Boundary Head — Design

Status: design + build. Date: 2026-08-07 (revised 2026-08-08). Companion to
[[BOUNDARY_DECODE_AND_EKF.md]] (the boundary decode map) and the mmBERT head-init scaling
finding ([[mmbert-head-init-finding]]). Sibling line to [[EKF_MHT_DESIGN]] — a *different*
route to dense document-level extraction: a global typed-constraint decode instead of a
tracker.

## 1. Thesis

Does the dormant **joint_ie global beam** (typed constraints + `Calibrator`), wired to the
**boundary** head (no span 20-cap), improve dense **document-level events *and* relations**,
and how does that interact with base-training **data volume** (head-init)?

Both, not either. The claim is about **structured output that does not fit a per-query
greedy decode**, and events and relations are its two faces:

| face | what is jointly decided | downstream |
|---|---|---|
| **relations** | which typed `(head, tail)` edges survive together | **Re-DocRED** (dense; the span 20-cap bites) |
| **events** | which arguments bind to which trigger, under role typing and cardinality | **RAMS** (roles dispersed across sentences) |

They share one mechanism, which is the point: in the beam an event is a **trigger node plus
role edges** and a relation is a **plain edge** (§3b), so a single typed-constraint decode
covers both. A win on only one face is a weaker but still reportable result; the honest
negative — that global decoding helps relations and not events, or the reverse — is itself
the finding, because it localizes where greedy per-query decoding actually costs you.

> **It held (§4b).** The *boundary* head beats the span curve on arguments at all three
> points (10K 0.177 vs 0.050, 40K 0.191 vs 0.115, 100K 0.202 vs 0.158) and the curves have
> opposite shapes: span **+216%** across the range and still climbing, boundary **+14%**
> and nearly flat. The head-init deficit is largely a property of the **span head**, not a
> data-volume law about mmBERT. Greedy arm only; the beam arm is still unmeasured.

This is also what makes the line the *document-level* half of the program
([[RESEARCH_PROGRAM]]): [[EKF_MHT_DESIGN]] carries events **beyond** the document via a
tracker, and this paper carries events **within** it via a global decode. Framing this half
as relations-only would break that symmetry and understate the shared claim.

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
| 7 | Sizes {10K,40K,100K,137K} as **total mix records**, every point carrying events + relations at the pool's 73/27 ratio; **NO LLM generation** (100K synthetic ≈ $400-860 batch) | DECIDED (revised 2026-08-08) |

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
  4. **`decode_group`'s record thresholds** — *mostly resolved*. Audited 2026-08-08 with
     both defaults at 0.5:

     | behaviour | greedy | joint | parity |
     |---|---|---|---|
     | list-field selection | `sigmoid(logit) ≥ 0.5` | centered utility > 0 | **same** |
     | scalar-field selection | argmax must beat ABSENT | `logit_c − logit_ABSENT > 0` | **same test** |
     | required role unfilled | allowed (`decode_group` leaves the field empty when ABSENT wins the argmax, *regardless of cardinality*) | allowed | **same** |
     | instance existence | `obj_prob ≥ threshold` | **now gated identically** (decision D revised) | **same** |

     Residual, recorded not fixed: joint gates on the trigger **mention** score *and*
     `obj_prob`, greedy on `obj_prob` alone. Two decoders cannot be made identical; the
     goal is no *unaccounted* asymmetry on the headline path.

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
| D | ~~v1 ignores `object_logits`~~ → **REVISED 2026-08-08**: `object_logits` gates instance existence, mirroring `decode_group` exactly | The original call treated this as Phase-B calibration. It is not — it is a **live arm confound**. `decode_group:22` drops an instance whose object probability is below threshold; without the same gate the joint arm had *no* existence gate at all and would emit events surviving on a single positive role edge, biasing the RAMS number **against the beam** and inviting the wrong conclusion from a plumbing gap. Verified aligned: joint now emits an instance **iff** greedy does, across the threshold boundary. |
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
5. ⏸ **`RequiredRoles`** constraint via the `validate` hook + registration in
   `_CONSTRAINT_TYPES` (`constraints.py:304`), with a compiler hook beside the existing
   `UniqueRelationSlot` emission. **DEFERRED past Phase A** (decided 2026-08-08), and the
   premise is now **verified in code**: `decode_group`'s scalar path does
   `if chosen is None or chosen == 0: continue` *before* the cardinality check, so when
   ABSENT wins the argmax a `REQUIRED_ONE` field is left empty exactly as an optional one
   is. Greedy permits unfilled required roles, so the arms stay comparable. Carried to
   Phase B (§7).

   Worth stating because it is counter-intuitive: implementing `RequiredRoles` as a
   *rejection* constraint would move the arms **further apart**, not closer — greedy never
   rejects such an instance. Parity, if ever wanted, means matching greedy's fill
   semantics, not adding a rejection rule.

**Decision 4b is met**: structures, relations and events all decode through the beam.

## 3c. ✅ RESOLVED: the training harness can now build a boundary model

Found *and fixed* 2026-08-08 while scoping the scaling configs. Decision 1 says the
architecture is **boundary**; decision 3 says the bases are retrained `from_encoder`.
Neither was reachable from a training config. Evidence as found:

| check | result |
|---|---|
| `AutoExtractor` / `architecture` / `boundary` in `tools/train/train.py` | **0 occurrences** |
| `_build_model` (`train.py:397`) | calls `GLiNER2.from_encoder(...)` |
| `GLiNER2` | `class GLiNER2(SpanExtractor)` — `from_encoder` hardcodes `architecture="span"` |
| `max_width: 20`, set by every `scaling-mmbert-*.yaml` | a **`span_head`** field (`configuration.py:33`; the validator message reads `span_head.max_width`) |
| anything under `tools/` referencing `BoundaryExtractor` | none — only these working papers |

So **`tools/train/config/scaling-mmbert-{10k,40k,100k}.yaml` train the SPAN architecture.**
The published head-init scaling curve (arg F1 0.050/0.115/0.158) is a *span* curve — which
is consistent with it being the head-init finding's baseline, but it is **not** a boundary
base and cannot warm-start the joint_ie experiment.

Consequences, in order:
1. **Writing new scaling YAMLs alone does not work.** `train.py` needs architecture
   dispatch (`AutoExtractor` / a boundary `from_encoder`) before any boundary config can
   train. That is the actual first task, ahead of the configs.
2. `max_width` must **not** be carried into boundary configs — the boundary head has no
   span-width cap ([[COUNTING_LAYER]]); that cap is the thing the architecture removes.
3. `decode_mode: joint` is a *decode* setting. It belongs on the eval/inference side, not
   in the training recipe — the two decode arms are an eval-time switch over one trained
   model, so the arms must not be baked into separate training runs.

### The fix (shipped)

- **`AutoExtractor.from_encoder`** — the architecture-dispatching counterpart to the
  per-class `from_encoder`. Dispatch lives in `auto.py` rather than duplicating hub
  loading per model class.
- **`_build_model` reads `model.architecture`** — default `"span"` keeps every existing
  config byte-identical. On the `pretrained` path the declared architecture is passed
  through, so a warm start against the wrong architecture raises
  `ArchitectureMismatchError` instead of silently training the wrong thing.
- The three remaining `GLiNER2.from_pretrained` sites (blind-test reload, threshold
  sweep, `push_to_hub`) now use `AutoExtractor`, so boundary checkpoints can be
  evaluated and pushed.
- **Audited, not assumed:** `GLiNER2Trainer` already derives `architecture` from the
  model (`trainer.py:1499`) and passes it to `ExtractorCollator` — trainer and collator
  needed no change.

### The configs (shipped)

| config | role |
|---|---|
| `joint-boundary-mmbert-{10k,40k,100k,137k}.yaml` | boundary cold-start bases, **total** mix sizes, all carrying events + relations (§4) |
| `joint-boundary-rams.yaml` | **event** warm-start arm (`eval_event_argument_strict_micro_f1`) |
| `joint-boundary-redocred.yaml` | **relation** warm-start arm (`eval_relation_strict_micro_f1`) |

New names and `./out/joint-boundary-*` paths throughout — the span `scaling-mmbert-*`
runs and anything on HF are untouched. No `max_width` (span-only field) and no
`decode_mode` (eval-time arm switch) in any of them; `test_train_configs` now *enforces*
both.

**§5's open provenance question is closed, empirically.** `sentence_rex` is
`knowledgator/sentence_rex` — sentence-level Wikidata-property RE, a different dataset
and granularity from `thunlp/docred`. Measured, not argued: **3,000 sampled sentences,
zero verbatim occurrences** anywhere in Re-DocRED's 3.2M chars of train text. The ~137K
point is safe to run. DocRED itself stays excluded.

## 4. Experiment (Phase A — decode-only)

- **Bases:** boundary `from_encoder` mmBERT-base at **{10K, 40K, 100K, 137K} TOTAL mix
  records** (Re-DocRED / any DocRED-derived set **excluded** — leakage). Configs shipped:
  see §3c.
- **Every point carries both event and relation data** at the pool's own ratio (events
  73.02% / relations 26.98%), so the only factor varying across the curve is **volume**:

  | total | events | relations |
  |---|---|---|
  | 10K | 7,302 | 2,697 |
  | 40K | 29,209 | 10,791 |
  | 100K | 73,022 | 26,977 |
  | 137,052 | 100,080 | 36,972 (whole pool) |

  Decided 2026-08-08, and it **removes a confound**: an earlier draft put relation
  corpora only at the top point, which would have left the relation head cold at the low
  end and made the Re-DocRED arm's slope partly an artifact of *whether* relations were
  seen rather than *how much* data was. Slices are built by
  `tools/train/build_joint_scaling_mix.py` (seed 42) and are **nested** — each corpus's
  10K slice is a prefix of its 40K, and 40K of 100K — so the sizes are strictly
  cumulative. Verified: nesting holds for all 14 corpora.
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

## 4b. First measured point (PROVISIONAL — 2026-08-09)

The 10K boundary base, warm-started on RAMS, on the **greedy arm** (arm (a); the beam
arm is not yet measured). RAMS blind test, 871 docs, strict micro-F1 — the same test
set and metric as the span curve in [[mmbert-head-init-finding]], so the columns are
directly comparable:

| Stage-A size | architecture | event_argument (S) | event_trigger (S) | event_type (S) |
|--:|:--|--:|--:|--:|
| 10K | span (prior curve) | 0.050 | 0.598 | 0.952 |
| 40K | span (prior curve) | 0.115 | 0.706 | 0.931 |
| ~100K | span (prior curve) | 0.158 | 0.732 | 0.949 |
| **10K** | **boundary (this work)** | **0.177** | **0.764** | 0.913 |
| **40K** | **boundary (this work)** | **0.191** | **0.812** | 0.925 |
| **100K** | **boundary (this work)** | **0.202** | **0.829** | 0.936 |

All boundary points calibrated to threshold **0.3**, so they are mutually comparable.

**The boundary head wins on arguments and triggers at EVERY point**, and the margin is
largest where data is scarcest:

| Stage-A | span arg | boundary arg | ratio |
|--:|--:|--:|--:|
| 10K | 0.050 | 0.177 | **3.54×** |
| 40K | 0.115 | 0.191 | 1.66× |
| 100K | 0.158 | 0.202 | 1.28× |

Two readings, and together they answer §1's question about how the beam interacts with
head-init:

1. **The boundary 10K point (0.177) still beats the span ~100K point (0.158)** — a tenth
   of the Stage-A volume, on ~27% fewer event records (the boundary mix is 73/27
   events/relations; the span curve's slice is events-only).
2. **The two curves have opposite shapes.** Span climbs **+216%** across the range
   (0.050 → 0.158) and is still climbing; boundary moves **+14%** (0.177 → 0.202) and is
   nearly flat. The span head spends the whole curve recovering from a low floor; the
   boundary head starts near its ceiling.

That is the finding: **the head-init data-scaling curve is largely a property of the SPAN
head, not of mmBERT.** What looked like "a cold multilingual encoder needs ≥40K of
structure/argument warming" is better read as "fixed-width span enumeration needs a great
deal of data to compensate for it, and a start/end factorization mostly does not." The
practical consequence flips the earlier recommendation: reach for the boundary head
before reaching for more warm-up data.

Event *type* is the exception — span is marginally ahead at 10K (0.952 vs 0.913) and 100K
(0.949 vs 0.936). Type is a whole-document classification where the span head's width cap
never bites, so no advantage is expected, and none appears.

**Why provisional — do not cite yet:**
- **Now metric-selected.** The point was retrained once the decode defect was fixed, so
  `eval_event_argument_strict_micro_f1` is live during training and `best/` is a real
  selection rather than epoch 1. The earlier provisional figure (0.182) came from
  `final/`; the metric-selected `best/` gives **0.177**, and event_type moves 0.946 →
  0.913. Normal val/test variance — the comparison against 0.158 survives either way,
  and 0.177 is the number to cite because it is the checkpoint one would ship.
- Decision threshold 0.3 (calibrated on val), not the per-model threshold the span
  curve used.
- One point. The claim above is a *conditional* — 40K and 100K are still running.
- Greedy arm only; the beam arm is not yet measured.

### The base curve (2026-08-09) — everything scales except relations

The cold-start bases are evaluated on a **fixed** blind test (identical across curve
points by construction, so support never moves and a metric change is attributable to
training volume alone). Strict micro-F1:

| | 10K | 40K | 100K |
|---|--:|--:|--:|
| entity | 0.311 | 0.402 | **0.459** |
| classification | 0.108 | 0.545 | **0.586** |
| event_type | 0.497 | **0.830** | 0.797 |
| event_trigger | 0.061 | 0.277 | **0.327** |
| event_argument | 0.010 | 0.039 | **0.079** |
| event (overall) | 0.070 | 0.186 | **0.201** |
| **relation** | 0.007 | 0.012 | **0.073** |

All three calibrated to threshold **0.3**, so these columns are directly comparable
(see the matched-threshold rule below).

Two different knees, and that is the finding:

- **Events and classification warm early.** Trigger 4.5× and classification +0.437 by
  40K, then continue more slowly. The knee is between 10K and 40K.
- **Relations warm LATE.** Flat at ~0.01 through 40K, then **6× at 100K** (0.012 →
  0.073, precision 0.071 → 0.456). The knee is between 40K and 100K — an order of
  magnitude later than events, on the same mixture.

The earlier reading here ("relations may not be warming from this curriculum at all")
was premature: they warm, just much later. Waiting for 100K rather than concluding at
40K was the right call, and the practical consequence is concrete — a mixture that
suffices to warm event heads can leave the relation head still cold, so the two faces
of §1's thesis have genuinely different data requirements.

Recall remains the relation bottleneck at every point (100K: P 0.456 / R 0.040), so the
head is learning to be *right* well before it learns to be *complete*.

**These numbers did not exist until the blind test was fixed.** No event corpus in the
four base configs declared a `test:` key, so `_event_split` returned nothing and the
bases — 73% event data — were scored only on relation corpora. Worse, adding the keys
was not enough: `_event_split` filters on `Path(p).is_file()`, and the event test slices
had never been copied to the training box, so the first regeneration silently reproduced
the event-free metrics with no error. A missing-path filter that drops silently is
indistinguishable from "this corpus has no test data".

### The relation head works — and curve points must be read at a MATCHED threshold

The bases score ~0.01 on relations (above), which invited the reading that the relation
head is unwired or dead. It is neither. Verified directly: `relation_scorer` is built
(2,952,193 params), relation queries appear in the layout, `edge_targets` carries real
positive labels, `relation_loss` is finite and `requires_grad`, and it back-propagates
into the scorer (4 parameter tensors, total |grad| 5030). The head is fine — the *base
curriculum* simply does not teach relations, while a Re-DocRED warm start does:
relation F1 **0.176** (10K-based) versus ~0.01 for the bases.

The warm starts then appeared to REGRESS with more base data — 0.176 (10K) → 0.136
(40K) — which would have been a striking negative. It is an artifact. Each model
calibrates its own decision threshold on its own val set, and the two landed at
**different operating points**: 10K at 0.1, 40K at 0.3. A 3× threshold fully accounts
for the recall collapse (0.193 → 0.086) and the precision jump (0.161 → 0.327).

At **matched** thresholds the ordering reverses and more base data helps everywhere it
matters:

| threshold | 10K base | 40K base | Δ |
|--:|--:|--:|--:|
| 0.1 | 0.1637 | **0.1777** | +0.0141 |
| 0.3 | 0.0890 | **0.1288** | +0.0398 |
| 0.5 | 0.0301 | **0.0422** | +0.0122 |
| 0.7 | 0.0066 | 0.0063 | −0.0003 |

**Methodological rule for this paper: a per-model calibrated threshold is correct for
shipping a checkpoint and wrong for reading a curve.** Card numbers are each model's own
best operating point; curve comparisons must fix the threshold across points, or the
operating point moves with the variable under study and the comparison means nothing.
Every scaling claim here should be quoted at a stated, matched threshold.

### The defect that hid this, and why it matters methodologically

Every event metric in this experiment read **0.0000 at every threshold, down to 0.01**,
until 2026-08-09. Cause: the boundary engine skipped every non-entity query on the
assumption the record head would decode it, but the record head is **inert for events** —
an events schema never produces `record_metadata`, so `compile_record_specs` returns `{}`.
Confirmed in *training* collation, not just inference, so `enable_records: true` never did
anything for events at any stage.

Events are in fact supervised as **mentions**: one extractive `[V]` query per trigger and
per role. The models had learned trigger and role spans the whole time; nothing assembled
them into output. Once assembled, the same checkpoint went from 0.0000 to the table above.

Two lessons worth carrying: (i) a metric that cannot leave 0.0 is indistinguishable from a
model that has learned nothing — `metric_for_best` pointed at exactly such a metric and
silently pinned checkpoint selection to epoch 1; (ii) the failure was invisible to the test
suite because it needs a trained boundary checkpoint plus a real event schema, which no
unit test constructs.

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

Re-estimated 2026-08-08. *(The previous figure — "3 bases + 3 Re-DocRED fine-tunes ≈ 6-10 hr
≈ $12-20" — predated the fourth base, the second downstream, and 15-epoch warm-starts. It is
superseded, not merely refined: the run is now **12 jobs, not 6**.)*

- **Data + wiring: $0.** Slices are on disk (`build_joint_scaling_mix.py`), no generation.

**Throughput anchor — measured, not assumed.** [[SCALING_CURVE_EXPERIMENT]] §"Memory"
records the combined base (~96K records × 2 epochs) at **~5 h on an A100** under this exact
recipe (mmBERT-base, 2048 window, `batch_size 4` × `grad_accum 8`, gradient checkpointing,
bf16) → **10.7 samples/s**. Everything below scales from that one number.

**Workload** — 12 runs: 4 bases + 4 RAMS + 4 Re-DocRED warm-starts.

| | record-epochs |
|---|---|
| cold-start, 5 epochs × 4 bases | 1,435K |
| RAMS warm-start, 15 epochs × 4 | 440K |
| Re-DocRED warm-start, 15 epochs × 4 (**×2.5**, it runs at `max_len 4096` not 2048) | 458K equiv |
| + 15% for per-epoch eval and the threshold/metric sweeps | **2,683K** |

→ **~70 GPU-hours on 1× A100.**

**Lambda on-demand** (rates verified 2026-08; ±30% on the anchor gives **$80-282 / 14-91 h**):

| config | wall-clock | cost | note |
|---|---|---|---|
| 1× A100 40GB | ~70 h | $139 | **OOM risk** — see below |
| 2× A100 40GB | ~39 h | $154 | DDP ~1.8×, not 2× |
| 1× A100 80GB | ~70 h | $195 | no OOM risk |
| 2× A100 80GB | ~39 h | $217 | safest A100 option |
| 1× H100 PCIe | ~35 h | $115 | ~2× A100 on bf16 |
| **2× H100 PCIe** | **~19 h** | **$128** | **recommended** — best time per dollar |

**Recommendation: 2× H100 PCIe.** The whole curve lands inside a day and it is *cheaper*
than 2× A100 80GB. 1× H100 saves $13 and costs 16 extra hours — not worth it. The DDP path
is already validated on 2× A10G, so multi-GPU is not new ground.

Three constraints that matter more than the arithmetic:

1. **Lambda sells H100/A100 *SXM* only as 8-GPU nodes** — requesting "2 GPUs" bills for
   eight. The table prices **H100 PCIe @ $3.29/h**, which is available in small counts.
   Re-price at 8× if SXM is actually wanted.
2. **1× A100 40GB will likely OOM on the Re-DocRED arm.** [[SCALING_CURVE_EXPERIMENT]]
   already records mmBERT-base OOM-ing 40GB at batch 8 / 2048; `joint-boundary-redocred.yaml`
   runs at **4096**. Take 80GB, or drop that arm to `batch_size 2` × `grad_accum 16`. The
   $56 gap is cheaper than discovering it three hours in.
3. **The estimate is deliberately conservative.** 27% of every base mix is now `sentence_rex`
   — single sentences, far shorter than event documents — so real throughput on the
   relation-carrying mixes should *beat* the event-only anchor. No discount was applied.

## 7. Phase B (deferred — only if A is positive)

Joint training: put the joint_ie beam **in the loss** via the boundary model's existing
**detached-association + differentiably-recomputed-scores** idiom (`records.py`,
`proposal.py`) — a structured-prediction objective. Feasible because that idiom already
exists; a training-loop change, not a bolt-on. Answers: does training-*for* the beam beat
decoding-*with* it?

Applies to **both faces** (§1): the role edges of an event and the plain edges of a relation
are the same `EdgeCandidate` in the same problem, so one structured objective covers events
and relations without a second mechanism.

### Carried here from Phase A

- ~~**decision D** (`object_logits` unused)~~ — **no longer deferred.** It was a live arm
  confound, not calibration, and is fixed: the joint path now gates instance existence on
  `object_logits` exactly as `decode_group` does (§3b decision D, revised 2026-08-08).
- **`RequiredRoles`** — deferred (§3b increment 5). Greedy permits unfilled required roles
  too, so it is not an arm-parity gap.

  > **Worth recording because it is counter-intuitive:** building `RequiredRoles` as a
  > **rejection** constraint would move the arms **further apart**, not closer — greedy
  > never rejects such an instance. If parity is ever wanted there, it means matching
  > greedy's **fill** semantics, not adding a rejection rule.

  The reflex when reading "required role" is to add a constraint that throws the instance
  away. That reflex is wrong here, and the code says why: `decode_group`'s scalar path runs
  `if chosen is None or chosen == 0: continue` **before** the cardinality check, so when
  ABSENT wins the argmax a `REQUIRED_ONE` field is simply left empty — exactly as an
  optional one is. A rejection constraint would make the beam stricter than the decoder it
  is being compared against, and the resulting precision/recall gap would be read as a
  property of global decoding rather than of the constraint. Anyone implementing this in
  Phase B should decide *fill vs reject* first, and know that only **fill** preserves
  comparability with the greedy arm.
