# joint_ie x Boundary Head — Design and Engineering Record

Companion to the paper `JOINT_IE_SCALING.md`. This file holds the material that belongs
to the build rather than to the result: the decision log, the wiring map, the increment
status, the data survey, the cost model, and the deferred Phase B plan. It is kept
verbatim so the reasoning behind each choice — including the choices later overturned —
stays available. The paper cites it for anything a reader needs to reproduce the system.

Extracted from the working paper on 2026-08-19, when that document was rewritten as a
paper. Nothing here was edited in the move.

---

## 2. Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Architecture = **boundary** (no span, no 20-instance cap) | DECIDED |
| 2 | Downstream target = **Re-DocRED + RAMS** (dense relations *and* document-level event arguments) | DECIDED (2026-08-08) |
| 3 | Bases **retrained `from_encoder`** (mmBERT), NOT span | DECIDED |
| 4 | joint_ie **wired to boundary** — mentions + relation edges | **DONE** |
| 4b | joint_ie must support **structures, relations, AND events** in the beam | DECIDED (2026-08-08) — **MET 2026-08-12** (`bf2c9b4`), no longer blocking; see §3b |
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
- ✅ **RESOLVED 2026-08-12 (`bf2c9b4`) — events ARE in the beam.** See "TIER 1 SHIPPED"
  below for the fix and its verification. The diagnosis that follows is kept because it
  is the record of *what* was wrong and how it was found; read it as history, not as a
  live blocker. (This bullet read "Still true" for two days after the fix landed under
  it — the header was never updated when Tier 1 shipped the same day.)

  **What was wrong.** Diagnosed 2026-08-12, narrower and worse than the older note said.

  **What is now right.** `_decode_records` no longer runs before the joint branch — it is
  gated behind `if not joint` (`engine.py:203`) precisely to avoid double emission, and
  `_decode_joint` builds `boundary_record_groups_to_role_edges` **and**
  `boundary_record_instance_nodes`, passing them to `joint_decode` as `extra_edges` /
  `extra_nodes` with an existence gate mirroring `decode_group`'s
  `record_anchor_threshold`. Records *are* in the beam, as instance nodes plus role edges.

  **What is actually broken.** `compile_record_specs` compiles a group only when its task
  type is in `RECORD_TASK_TYPES`, and that tuple is `("json_structures",)`
  (`processing/records.py:31`). Event groups therefore never become record specs.
  Measured directly — same layout, same `record_metadata`, only `task_type` changed:

  | task_type | record specs compiled |
  |---|--:|
  | `json_structures` | 1 |
  | `events` | **0** |

  So in joint mode an event group yields no record spec → no role edges and no instance
  nodes → nothing in the beam; and `_decode_events` is unreachable there, sitting after
  the `if joint: … continue`. **Events are not decoded greedily in joint mode — they are
  not decoded at all.** The old wording ("produced, but greedily") describes a safer
  failure than the one present.

  **Why the tests do not catch it.** `test_joint_records.py` and `test_record_role_edges.py`
  both construct `RecordSpec(task_type="events", …)` **by hand** and never call
  `compile_record_specs`. `test_role_edges_rebuild_an_event_instance` passes. So the
  role-edge machinery is *already proven to work for events*; only the gate excludes them.

  **The fix is not a one-line tuple edit.** `compile_record_specs` feeds the greedy path
  too, so adding `"events"` to `RECORD_TASK_TYPES` would route event groups through the
  record decoder *and* leave `_decode_events` running in greedy mode — double emission
  (`engine.py:234` currently splits them: "events are assembled below; records by the
  record head"). Whoever takes this must settle greedy-side ownership of events in the
  same change. Design in §3b. *(An earlier note judged this non-blocking on the assumption
  of a Re-DocRED-only downstream — that assumption is void; RAMS is on the evaluated path
  via decision 2.)*

  **TIER 1 SHIPPED 2026-08-12** (`bf2c9b4`). Events are now emitted from the beam:
  `_decode_joint` collects event-typed nodes out of `solution.nodes` and
  `_format_joint_events` assembles them in `_decode_events`'s exact shape. Measured on
  `gliner2-joint-boundary-rams-137k`, first RAMS train record, threshold 0.3 — pre-fix
  greedy emitted the event and joint emitted **nothing**; post-fix joint is byte-identical
  to greedy. Selection is now the beam's, which is the actual Phase A claim.

  **TIER 2 — multi-instance events. Sized before building, and it is the largest headroom
  in these papers.** Both decode paths emit **one instance per event type**, because the
  mention axis carries no instance dimension. What that costs, counting gold instances
  that are unreachable under that cap:

  | corpus | docs with >1 event of the same type | gold instances | unreachable |
  |---|--:|--:|--:|
  | RAMS | **0.0%** | 7,329 | **0 (0.0%)** |
  | MAVEN | 96.1% | 77,993 | 29,885 (**38.3%**) |
  | WikiEvents | 93.8% | 3,241 | 2,027 (**62.5%**) |
  | CASIE | 94.5% | 6,708 | 5,285 (**78.8%**) |

  Two consequences. **RAMS cannot see this at all** — it is 100% single-event documents,
  so the base-word arms A/B/C are unaffected either way and Tier 2 must be evaluated on
  CASIE or WikiEvents, not RAMS. And every event metric already published on
  CASIE/WikiEvents/MAVEN was measured under a hard recall cap, so a post-Tier-2 model is
  **not comparable** to them.

  **TIER 2 SHIPPED behind `event_records`, and its first measurement is NEGATIVE — for
  head-init reasons, not mechanism ones.** Ran on CASIE 2026-08-12, two arms differing in
  that one key, base `gliner2-joint-boundary-rams-137k`, 375 steps on 798 documents:

  | metric | control | event_records |
  |---|--:|--:|
  | event argument fair | **0.2998** | 0.0036 |
  | event argument relaxed | **0.3355** | 0.0366 |
  | event argument strict | 0.0158 | 0.0000 |
  | event strict | 0.0906 | **0.0999** |
  | event type strict | 0.6972 | **0.7372** |

  **The mechanism works and that is separable from the score.** Probing 39 multi-instance
  CASIE test documents with the trained treatment, 17 emitted 2–9 instances of a single
  event type — structurally impossible before this change. The instances are simply mostly
  wrong. The control's mention path arrived pre-trained on RAMS events and transfers; the
  treatment's record head had never been supervised on events and got 375 steps to learn
  instance formation cold. Head-init, evidenced by contrast rather than assumed: the
  control is functional on softer matching (0.2998) while the treatment sits at the floor
  (0.0036).

  **What would actually test it:** warm-start the record head on events with MAVEN (2,913
  documents, 38.3% unreachable gold), THEN fine-tune on CASIE, with a real step budget.
  MAVEN ran on its own (below) and was flat; **the CASIE stage-2 leg — warm-start THEN
  fine-tune — is still unrun**, and it is the one that tests the head-init explanation.
  Two pre-existing CASIE/boundary blockers are already fixed in the configs and will bite
  anyone else: `error_policy: raise` aborts on unlocatable entity surfaces, and a single
  CASIE query carries up to **188** gold spans against the default `max_gold_per_query` of
  32. Both verified pre-existing with the flag off.

  Metrics and trimmed logs are local under `out/casie-tier2-{control,eventrecords}/`; the
  checkpoints went with the terminated box.

  **MAVEN RAN 2026-08-12, and Tier 2 bought nothing there either.** Half of MAVEN's
  official *valid* split held out document-wise (355 docs / 13,637 gold instances), two
  arms differing only in `event_records`, base `gliner2-joint-boundary-rams-137k`,
  threshold swept on val over (0.1, 0.3, 0.5, 0.7, 0.9):

  | metric | control | event_records | delta |
  |---|--:|--:|--:|
  | event_trigger strict | **0.7407** | 0.7327 | **−0.0080** |
  | event_type strict | 0.8893 | **0.8943** | +0.0050 |
  | event strict | **0.8011** | 0.7980 | −0.0031 |

  At **5.4x the training cost** (2h47m at 7.35 s/it against 35m at 1.37 s/it). Two things
  make this weaker evidence than it looks, and both should be stated before anyone cites
  it: the treatment arm is **flat at 0.7295 across every threshold**, because the
  record-head decode does not consume the mention threshold, so threshold tuning is inert
  on that arm; and trigger strict F1 **aggregates trigger spans per event type**, so the
  control's single instance already carries multiple triggers. That metric therefore does
  *not* isolate multi-instance separation, which is the actual Tier 2 claim on a corpus
  where 40.8% of test gold is unreachable one-per-type. **A metric that counts instances
  is still needed to test this properly.**

  ⚠ **The run's apparent headline was an artifact, and it is the reason two metric
  defects got fixed.** `train_results.json` reported `best_metric` 0.8886 treatment vs
  0.8392 control — a "+0.049 win" that is **not F1 at all**, but epoch-1 eval *loss*.
  Chain, each link verified: `_schema_from_gold` dropped every role-less event type, and
  MAVEN is trigger-only (168 types, zero arguments), so all 355 schemas went empty and
  `compute_metrics` returned `{}`; with the key absent, `metric_for_best` silently fell
  back to `eval_loss`, and `greater_is_better: true` then selected the *highest* loss —
  epoch 1, step 91 of 1365, for both arms. The treatment scoring higher is expected under
  a loss reading, since it carries an extra supervised head. All three defects are now
  fixed (role-less types kept; `c0ab89c` raises on an absent `metric_for_best`; `3d21eba`
  raises in the threshold sweep, which had been scoring every grid point 0.0). The table
  above is scored on `final/`, not `best/`.

  Metrics and `final/` checkpoints are local under `out/maven-tier2-{control,eventrecords}/`;
  both arms are on the Hub as `whr778/gliner2-maven-tier2-{control,eventrecords}`. Any
  *fair* number from this run predates `d6debaf` and must be recomputed before citing;
  strict is unaffected.

  **The corpora need no change — an earlier scoping note of mine said otherwise and was
  wrong.** `_process_events` appends one label row per event mention
  (`processor.py`, training path), so `structure[1]` is already a list of instance rows
  and `structure[0]` is the instance count; the count loss already consumes it via
  `if task_type != "entities"`. The instance dimension is in the supervision today. What
  is missing is only that `compile_record_specs` refuses event groups, so the record head
  — which *does* carry instance queries — is never given them. Remaining work is therefore
  spec compilation (events have a structural anchor: field 0 is always the trigger),
  greedy-side ownership, and a retrain to benefit.
- ⚠ **Arm-comparability caveat (must settle before Phase A):** the joint path threads the
  engine `threshold` through as `mention_threshold`, but four greedy-side threshold
  behaviours are **not** mirrored:
  1. **adaptive thresholding** (`boundary_settings.adaptive_threshold`) — greedy-only;
  2. **null-abstention** (`abstention_threshold` on `null_logits`) — greedy-only;
  3. **per-relation-type thresholds** from `metadata["relation_metadata"]` — greedy looks
     them up per relation; `candidate_score_set_to_problem` centered edges at a fixed
     `decision_threshold=0.5`. ~~Moot for a single-threshold Re-DocRED eval~~ —
     **that call was wrong, and it was the dominant confound.** Fixed 2026-08-10.

     The fixed 0.5 did not merely differ from greedy's per-relation lookup; it meant the
     joint arm ignored the eval threshold *entirely* for edge selection, because utility
     is centered there and the optimizers only take positive-utility edges. Measured
     relation recall across thresholds 0.5 → 0.1: greedy **0.0461 → 0.4134**, joint
     **0.1498 → 0.1591**. The joint arm was pinned to one operating point and looked
     "calibration-insensitive" rather than broken.

     `decision_threshold` now threads from the eval threshold. Record **role edges bypass
     it** (`pre_scored_edges`), preserving decision C below: a scalar role's utility is
     ABSENT-relative and has no probability cutoff to be centered on. Enforced by test,
     not comment. The per-relation-type lookup itself is still unported.
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


---

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


---

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

### The record head is ~5x slower to train than the rest (2026-08-10, unresolved)

The warm-start run (structure + NER added to `mmbert-137k` with 30% replay) trains
CORRECTLY -- loss 9.10 -> 4.35 by step 39 -- but at **4.6 samples/s against the curve's
22** on the same H100 and the same model. ETA 14h for 3 epochs rather than ~3h. Killed
after ~$6 rather than paying 4.6x the estimate for a result we can get later.

What was measured, so the optimisation does not start from scratch:

| signal | value | reading |
|---|---|---|
| GPU utilization | **0%** over 10 samples (one 6%) | the GPU is idle, not saturated |
| GPU memory | 12.8 GB resident, one compute app | the model IS on the device |
| CPU | one core at ~105%, load average 1.0 | a single serial bottleneck |
| dataloader workers | 12 spawned, **all idle** | NOT a data-loading bottleneck |
| `num_workers` 0 -> 12 | 5.0 -> 3.2 s/it | small gain, so collation is not the cause |

That combination -- idle GPU, one pegged core, idle workers -- is the signature of many
small GPU kernels behind heavy Python-side work, not of compute or I/O.

**The prime suspect is the record path**, by elimination: this run differs from every
curve arm in exactly one way, it carries `json_structures` supervision. The 137K mix had
none (its `text2json` corpus supervises entities despite the name), so `record_decoder`
was never exercised during the curve, and the curve is where the 22 samples/s came from.
`record_instance_queries: 32` per document, times multi-instance documents, is where to
look first.

Not yet distinguished, and worth separating before changing anything:

1. target-graph construction for records (collator side, should be parallel but the
   workers are idle, which itself needs explaining), versus
2. the record decode/loss path in the training step (main process, would match the
   symptom exactly).

The frequent `gold capacity exceeded (max_gold_per_query=32, overflowing queries={0: 72})`
warnings show NER documents are also expensive to build targets for, so the two candidates
overlap and a profile -- not a guess -- should settle it.

Everything needed for the relaunch is built and committed: `data/warmstart_mix.train.jsonl`
(84,279 records, 70/30, shuffled) and `joint-boundary-warmstart-struct.yaml`.

### Warm start: records ARRIVE, the location field does not (2026-08-10)

Adding structure + NER to `mmbert-137k` with 30% replay, 3 epochs, 82 minutes, 51.4
samples/s, zero non-finite losses, ~$6. Both runs calibrated to threshold **0.3** and — the
check that retracted the curve above — **identical support on every family**, so this
comparison is valid.

| capability | mmbert-137k | warm start | delta |
|---|--:|--:|--:|
| event | 0.328 | **0.349** | +0.021 |
| event_argument | 0.098 | **0.118** | +0.020 |
| event_trigger | 0.751 | **0.759** | +0.008 |
| classification | 0.631 | 0.631 | 0.000 |
| entity | 0.586 | 0.580 | −0.006 |
| relation | 0.170 | 0.154 | −0.016 |
| event_type | 0.956 | 0.935 | −0.021 |

**30% replay held.** The worst regression is 2.1% relative, against a run that added two
task families. Events improved, which was not the goal and is a free result.

**The added capability arrived.** The base cannot emit records at all; the warm start can:

    "At least 41,000 deaths ... in Turkey, while 5,800 ... in Syria."
      mmbert-137k  -> None
      warm start   -> [{dead: '5,800'}, {dead: '41,000'}]     TWO instances

**But `location` is never filled** — the heterogeneous-field goal, and the reason the corpus
was rebuilt at all. Multi-instance emission works; the non-numeric field does not.

**A separate discovery, which invalidates every earlier "records return None" measurement.**
The boundary record path requires `record_metadata`, and `Schema().structure(name)` does not
emit it. Only a DECLARED record schema does:

    s.structure("casualty_report", mode="natural", anchor="dead")
     .field("dead", dtype="str", cardinality="required_one")

With the plain form both models return `None`; with the declared form the warm start
returns records. So the earlier conclusion that "`mmbert-137k` cannot do `[C]` record
extraction" was measured with a schema that could never have worked — the model may have had
more record competence than credited, though it still returns `None` under the declared
schema, so the *comparison* stands even though the *method* was wrong.

> **This was a SECOND instance of the §0c defect, and it is now FIXED (2026-08-19).** The
> `runtime.py` fix did not close it: that one dropped a key `build()` had produced, this
> one was `build()` never producing it. `StructureBuilder._auto_finish` now defaults to
> `mode="natural"` anchored on the first declared field — the same choice
> `_store_record_metadata` already made for a caller who set mode and omitted anchor.
> Verified end-to-end on the 137k checkpoint: the plain and declared forms return
> identical records where the plain form previously returned `None`. So the measurements
> quoted just above were taken against a schema that could not work, and can now be
> retaken. This matters most for the EKF/disaster line, which builds its schemas through
> exactly this API.

**Unresolved: why `location` did not learn.** Inspecting collated targets suggested no record
targets were built for the `json_structures` training format, which would explain it — but
the same probe reports `start_targets = None`, which cannot be true of a run that trained.
The probe is not reading the right structures, so the mechanism is NOT established and is
recorded as open rather than guessed at.

### Record mode A/B: `natural` works, `anchorless` learns nothing (2026-08-10)

Single-variable by construction: both mixtures byte-identical with `record_metadata`
stripped (same content hash, 84,280 records), both arms on ONE 2xH100 box with a GPU each,
identical hyperparameters, both calibrated to threshold 0.3, identical blind-test support.
The only difference is `{"mode": "natural", "anchor": "dead"}` versus
`{"mode": "anchorless"}`.

**Record extraction** (`probe_records.py`; `loc filled` is the discriminator, since the
blind test carries no structure data and scores both arms alike):

| model | instances | value | **loc filled** | loc correct |
|---|--:|--:|--:|--:|
| mmbert-137k (base) | 0/9 | 0/9 | 0/9 | 0/9 |
| warm start, no `record_metadata` | 6/9 | 5/9 | **0/9** | 0/9 |
| **natural** | **7/9** | **6/9** | **6/9** | **4/9** |
| anchorless | 1/9 | 1/9 | 0/9 | 0/9 |

**`location` fills for the first time.** It was 0/9 in every prior configuration, which
confirms the diagnosis: the field never learned because the corpus emitted no
`record_metadata`, so `compile_record_specs` returned nothing and the record head was never
supervised. Declaring the mode fixed it. Not solved, though -- 6/9 filled but only 4/9
correct, so it binds the wrong place about a third of the time.

**`anchorless` collapsed**, and by more than expected. The prior evidence was a model never
TRAINED on anchorless failing to decode it, which was explicitly discounted as evidence
about the model rather than the mode. Training on it did not rescue it. The likely reason
is that a casualty record genuinely HAS a natural anchor -- the figure itself -- so removing
it leaves nothing to key instances on.

**Capability cost, and the two arms explain each other** (same baseline, same support,
threshold 0.3):

| capability | base | natural | anchorless |
|---|--:|--:|--:|
| event_argument | 0.098 | **+0.021** | +0.002 |
| event_trigger | 0.751 | +0.014 | +0.017 |
| entity | 0.586 | -0.015 | -0.026 |
| event_type | 0.956 | -0.017 | +0.004 |
| **relation** | 0.170 | **-0.037** | -0.002 |

`natural` pays 0.037 on relations (-22% relative) while `anchorless` pays 0.002 and is flat
everywhere. That is not two different regressions -- it is one arm learning a new task and
displacing capacity, and one arm learning nothing to displace anything with. **The relation
cost is the price of the capability**, which makes it a trade to price rather than a bug to
fix.

**Where to look first if that price is too high.** `task_lr` is 5.0e-4 and was tuned in the
curve for COLD heads. Here the relation head is warm and receives only 8% of the mixture --
few gradients at a high rate, which is a recipe for drift. That targets the regression more
directly than `encoder_lr`, which acts on the shared trunk where relations are not the only
thing at stake. Untested.
