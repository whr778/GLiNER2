# joint_ie × Head-Init Scaling on the Boundary Head — Design

Status: design + build. Date: 2026-08-07 (revised 2026-08-08). Companion to
[[BOUNDARY_DECODE_AND_EKF.md]] (the boundary decode map) and the mmBERT head-init scaling
finding ([[mmbert-head-init-finding]]). Sibling line to [[EKF_MHT_DESIGN]] — a *different*
route to dense document-level extraction: a global typed-constraint decode instead of a
tracker.

> ## ⚠⚠ SUPERSEDED TWICE. The "clean" column below is ALSO pre-repair (2026-08-18)
>
> A **second, larger** contamination was found on 2026-08-18: 45 corpora shipped
> train/val/test that overlapped each other, and this config additionally paired
> regenerated `data/*.train.jsonl` against FROZEN `data/scaling_joint/*.val.jsonl`
> from 2026-08-08 — 252 train-in-val and 22 val-in-test documents, mostly
> `events_biotech` and `text2json`.
>
> So the numbers presented below as **"clean" are not clean**. Both columns of that
> table are superseded. All 45 corpora were repaired (21,553 records dropped, 0.94%),
> the scaling slices were rebuilt from the repaired sources, and all four points now
> gate CLEAN via `check_leakage.py --config`.
>
> **The curve was re-run from scratch on repaired data and is COMPLETE (2026-08-19).**
> Use these numbers, not anything below. Pool shifted to 136,772 records (was 136,787).
>
> | head | 10k | 40k | 100k | 137k |
> |---|--:|--:|--:|--:|
> | event_type | 0.8138 | 0.7503 | 0.7979 | **0.9841** |
> | event_trigger | 0.2259 | 0.3285 | 0.4032 | **0.6953** |
> | classification | 0.0974 | 0.4838 | 0.5518 | **0.6336** |
> | entity | 0.2779 | 0.4323 | 0.4795 | **0.5158** |
> | event | 0.1475 | 0.2232 | 0.2643 | **0.2688** |
> | relation | 0.0058 | 0.0175 | 0.0486 | **0.2071** |
> | event_argument | 0.0130 | 0.0502 | 0.0815 | 0.0692 |
> | structure | 0.0294 | 0.0568 | 0.1102 | **0.1119** |
>
> The `structure` row is CORRECTED (2026-08-19) and read 0.0000 across the
> board here until then -- a decode bug, not a head defect. Unlike every
> other row it is measured on the full test splits for ALL FOUR points and
> at each model's own swept record threshold. See §0c before quoting it.
>
> **Read the 100k -> 137k jump as RECALL UNLOCKING, not capability.** Only 1.37x the
> data but event_trigger +0.29, relation +0.16, event_type +0.19. The precision/recall
> split gives it away: event_trigger at 100k was P=0.58/R=0.31, at 137k P=0.62/R=0.79.
> The smaller points were not bad at the task, they were WITHHOLDING predictions.
> Anyone reading this as smooth capability scaling will draw the wrong conclusion about
> what more data buys.
>
> event_argument strict FELL 0.0815 -> 0.0692 while relaxed rose 0.2758 -> 0.5064 --
> the same inversion. More arguments proposed, approximately right far more often,
> exact spans lagging. Calibration, not regression.
>
> 137k `best` is the EPOCH-4 checkpoint (eval_loss 1.3107/1.0936/0.9996/0.9592; epoch 5
> did not improve). Models: `whr778/gliner2-joint-boundary-mmbert-{10k,40k,100k,137k}-clean`,
> all private. ~$42 total.
>
> The historical warning is kept below because the *reasoning* still applies and the
> two-effects-confounded point is still worth understanding.
>
> ## ⚠ Historical (2026-08-15): the FIRST contamination
>
> The splits leaked. `SplitWriter` drew one random **per row**, so a document emitted more
> than once scattered across train/val/test. On this config's aggregated splits that put
> **1,080 documents (7.03% of the blind test) inside train** — text2json 791,
> sentence_rex 210, events_biotech 58, docee 12, docfee 8, mendeley_ed 1.
>
> **Do not quote the base reference below** (`entity 0.586 / relation 0.170 /
> event_type 0.956 / event_argument 0.098`). Re-measured on the decontaminated blind test,
> the same checkpoint scores:
>
> | metric | clean | contaminated |
> |---|--:|--:|
> | entity strict | **0.6306** | 0.586 |
> | relation strict | **0.1573** | 0.170 |
> | event_type strict | **0.9365** | 0.956 |
> | event_argument strict | **0.1014** | 0.098 |
>
> Two effects are confounded and the delta is **not** a contamination estimate:
> decontamination lowers scores, but the test was also recomposed — text2json is now
> emitted as `json_structures` rather than entities, removing its pseudo-entity types
> (`aces`, `originalpostlink`) from the entity test population, which raises entity F1.
>
> Curve *shapes* across {10k, 40k, 100k, 137k} are probably still directionally right
> (every point shared the same contaminated test), but any absolute number, and any
> comparison against a differently-built model, needs re-measuring. Contamination is now
> gated automatically by `tools/train/train.py`; see
> [`../train/TRAINING.md`](../train/TRAINING.md) §3.

## 0b. Warm start with 30% EXACT replay: forgetting reversed into a gain (2026-08-19)

Warm-started the clean 137k base on cc_news + synthetic (21,354 records) with 9,151
records of replay drawn from **the base's own training pool**, then scored that
checkpoint on **the base's own blind test**:

| head | base 137k | warm-start | delta |
|---|--:|--:|--:|
| entity | 0.5158 | **0.6326** | **+0.1168** |
| event | 0.2688 | **0.3644** | **+0.0956** |
| event_trigger | 0.6953 | **0.7490** | **+0.0537** |
| event_argument | 0.0692 | 0.0975 | +0.0283 |
| classification | 0.6336 | 0.6394 | +0.0058 |
| event_type | 0.9841 | 0.9447 | -0.0394 |
| relation | 0.2071 | 0.1245 | -0.0826 |
| structure | 0.1119 | 0.1060 | -0.0059 |

The `structure` row was 0.0000/0.0000 until 2026-08-19 (decode bug, §0c) and the
warm-start delta must be read knowing the arm CUT record-head supervision by 93%
-- see §0c, "What the warm-start actually did to structure".

**Context: the zero-replay arms lost 23%, 32% and 39%** of general-domain entity F1
(three fine-tunes of `fastino/gliner2-base-v1` on the same corpora, no replay, in
monotonic order of training volume). With 30% replay the model GAINED on the original
task while learning a new one.

Why this arm is stronger evidence than those: we own this base AND its pool, so replay
is **exact** rather than a proxy. For base-v1 we do not have the original training data
and had to stand in `pile_ner_def`.

Qualifications that must travel with the number: different architecture from the
base-v1 arms (mmBERT boundary vs DeBERTa span), so it is not a controlled contrast;
part of the gain is genuinely NEW learning, since cc_news supplies real entity signal;
and protection is **not uniform** -- relation -0.083 and event_type -0.039 both
declined, which is the open thread.

Model: `whr778/gliner2-warmstart-137k-realsynth-replay30` (private). Config:
`tools/train/config/warmstart-137k-realsynth-replay30.yaml`. Replay built by
`tools/train/build_137k_replay.py` (proportional across all 13 pool corpora, seed 42).

## 0c. `structure` = 0.0000 was a DECODE BUG. The head works and scales (2026-08-19)

**RESOLVED. The earlier text in this section was wrong and is retracted below.** It read:
five independent zeros (10k / 40k / 100k / 137k / warm-start) mean the record head is
defective, so stop buying structure data and instrument the head. Instrumenting the head
was the right call. The conclusion drawn from the zeros was not: the head was never
broken, and it had never once been asked a question it could answer.

### The defect

`runtime.py` rebuilt each schema through `Schema.from_dict(...).build()` -- which *does*
produce `record_metadata` -- then copied only `json_structures` and `json_descriptions`
out of the result. `record_metadata` was dropped on the floor. Downstream,
`compile_record_specs` returns `{}` when it gets no metadata, so no `RecordSpec` was
compiled, the record head decoded **nothing**, and **no error was raised**. The
extraction came back empty and `structure` scored exactly 0.0000. One line, five
poisoned measurements, no traceback. Fixed in `d754132`; regression-tested in
`tests/test_record_metadata_roundtrip.py`, which exercises the real
`_build_schema_dicts_and_metadata` (an earlier version re-implemented the logic inline
and passed with the fix stashed -- it asserted that a copy works, not that the shipped
code does one).

A second, smaller problem sat on top: `record_anchor_threshold` defaults to **0.5**,
above where this head is confident. It is *not* the reason the metric read zero --
post-fix, 0.5 still scores 0.0654 and 0.0760 at 100k and 137k -- but it costs 32-100% of
attainable F1 depending on scale, and nothing had ever calibrated it (`threshold_sweep`
moves the general decision threshold and never touches the record cutoffs).

### The head scales, and it always did

Re-scored from the SAME checkpoints -- nothing retrained -- by
`tools/train/sweep_record_thresholds.py`, strict exact-match on `(name, field, value)`:

**All five checkpoints on ONE test set** -- the full test splits, 856 structure-bearing
records, support 4,245. Quote this curve, not the per-card numbers:

| | 10k | 40k | 100k | 137k | warm-start |
|---|--:|--:|--:|--:|--:|
| structure strict F1 | 0.0294 | 0.0568 | 0.1102 | **0.1119** | 0.1060 |
| precision | 0.0302 | 0.0783 | 0.3030 | 0.2365 | 0.2102 |
| recall | 0.0287 | 0.0445 | 0.0674 | 0.0733 | 0.0709 |
| at record threshold | 0.03 | 0.05 | 0.10 | 0.10 | 0.10 |

A monotone curve across a 13x data range that **flattens hard between 100k and 137k**
(+0.0017, and 100k is the more precise model at 0.303 vs 0.237). This is the ordinary
shape of a head learning from 7,754 supervised records. There was never an anomaly to
explain -- and note that the extra 37k records bought recall (0.067 -> 0.073) at the cost
of precision, the same recall-unlocking trade the other heads show at that step.

Recall is the binding constraint everywhere: even the best model recovers 7.3% of gold
triples. That, not the head's existence, is the real open question.

> Two caveats on the table. **10k's best sits at the bottom edge of the grid (0.03)**, so
> its 0.0294 is possibly understated -- left as-is rather than re-run, since precision was
> already collapsing to 0.030 there and a lower cutoff buys nothing real. And **the model
> cards report each config's OWN test set** (10k/40k/100k carry subsampled slice test sets
> of 148 records, support 758, giving 0.0238 / 0.0552 / 0.1043), so that every row on a
> given card is read on one population. Card numbers and this table are both correct and
> are not interchangeable.

### What the warm-start actually did to structure

The retracted text called the warm-start "+45% structure supervision (3,494 records)".
**The count is right and the conclusion is backwards.** Those 3,494 records carry
`json_structures` but **no `record_metadata`**, and on the training path
`Structure.get_record_metadata()` returns `None` unless `mode` is set, so
`compile_record_specs` builds no spec and `_build_sample_records` emits no targets. They
supervise the record head with exactly **zero** records.

| source | structure records | reaching the record head |
|---|--:|--:|
| cc_news_haiku45 | 1,033 | 0 |
| synthetic_haiku45_5k | 996 | 0 |
| synthetic_sonnet5_1k | 1,465 | 0 |
| replay_137k30 | 519 | 519 |
| **warm-start total** | **4,013** | **519 (12.9%)** |
| *137k base pool (text2json)* | *7,754* | *7,754 (100%)* |

So the arm did not add 45% more structure supervision. It **cut it by 93%**, from 7,754
records to 519. Measured on the base's own blind test the warm-start scores 0.1060
against the base's 0.1119 -- a 0.0059 dip under a 15x reduction in record-head
supervision, which is a strikingly good showing for 30% replay, not a failure to add
capability. (Scored on the base's test set: at the time of measurement the warm-start's own test set
was unscorable, because all 452 of its structure records carry no `record_metadata` and
nothing decoded without it. **That changed the same day** -- the builder now defaults to
natural mode, so those records are scorable and the warm-start is being re-scored on its
own test set. Note the anchor there is *synthesized* from the gold's first field rather
than declared, which is a weaker guarantee than the 137k set's explicit metadata.)

### Consequences

- **The cc_news and synthetic converters emit `json_structures` without
  `record_metadata`.** Any future arm claiming to add structure capability from those
  corpora will add none. Fixing it means assigning `mode` and `anchor` per structure type
  -- a data-design decision, not a silent patch. Open in TODO.md.
- `build_warmstart_mix.py`'s docstring claim that text2json supervises entities rather
  than structures describes an OLDER state: it now emits `json_structures`, and all 7,754
  carry metadata.
- Anything reporting `structure` at the 0.5 default is reporting a miscalibration.

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
> data-volume law about mmBERT. Greedy arm only.
>
> On the beam arm: it was described here as "unmeasured", but until 2026-08-10 it was
> **unrunnable** — mention keys dropped the relation type, so any schema with two relation
> types raised `node candidate ids must be unique`. First run on Re-DocRED that same day;
> see §4c.

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
| **137K** | **boundary (this work)** | **0.192** | 0.823 | **0.963** |

All boundary points calibrated to threshold **0.3**, so they are mutually comparable.

> ### ⚠ 2026-08-15: the boundary curve is FLAT WITHIN NOISE. Do not read its shape.
>
> The 137K point was added, and with it a **control**: the published 137K recipe, same
> base, same 15 epochs, re-run on current code. It scores **0.2151** against the
> published **0.192** — **+0.023 from a re-run alone**
> ([[lambda-rams-warmstart-run]], arm `rams-clean-a-base137k`).
>
> So **single-run variance on RAMS argument F1 is at least ±0.02**, and every boundary
> point here is one seed with no measured floor. The consequences, stated exactly:
>
> | claim | status |
> |---|---|
> | boundary beats span at 10K (0.177 vs 0.050, **3.5×**) | **SAFE** — 0.127 is 6× the variance |
> | the head-init curve is largely a SPAN-head property | **SAFE, and strengthened** — span climbs 0.108 across the range (5× variance); boundary's whole 10K→137K spread is 0.177–0.215, i.e. *one run's variance*. It does not measurably climb at all. |
> | boundary "moves +14% and is nearly flat" | **restate** — it is not a shallow trend, it is **flat within noise**. The +14% is an artefact of reading one seed per point. |
> | boundary 10K (0.177) beats span ~100K (0.158) | **MARGINAL** — +0.019 sits *at* the variance. Directionally supported by the 10K-vs-10K gap; do not cite the number alone. |
> | arguments/triggers "peak at 100K" | **RETRACTED** — never claimed in this file, but it was briefly concluded from the 137K dip. There is no turn; −0.010 is half a variance. |
>
> The honest summary is stronger than the one it replaces: **more Stage-A volume past
> 10K buys the boundary head nothing measurable on RAMS arguments.** Any future claim
> about this curve's shape needs ≥2 seeds per point first.
>
> ### Does an intermediate multi-task stage help? No.
>
> Same run tested a three-stage chain — `mmBERT → 137k joint → mix_natural → RAMS` —
> against the two-stage published recipe. All three arms swept to threshold 0.3:
>
> | metric (thr 0.3) | A: 137k base | B: via mix_natural | C: via event-weighted mix_natural |
> |---|--:|--:|--:|
> | event_argument (S) | **0.2151** | 0.2104 | 0.2133 |
> | event_trigger (S) | 0.8130 | 0.7991 | **0.8197** |
> | event_type (S) | 0.9280 | **0.9493** | 0.9455 |
> | event (S) | 0.5156 | 0.5200 | **0.5410** |
>
> Arguments span **0.005** across all three — a fifth of the variance. An intermediate
> `mix_natural` stage (34% structures, 36% NER) neither helps nor hurts the event
> downstream, which is itself worth knowing: warm-starting through a broad multi-task
> corpus does **not** cost event capability the way the casualty fine-tune's zero-replay
> narrowing did.
>
> One delta is worth a second seed: **event_type, B +0.021 and C +0.018 over A**, same
> sign in two independent treatment arms and approaching the variance scale. If a broader
> intermediate stage helps event *typing*, that is a cheap win — but n=1 per arm.
>
> C (event-weighted intermediate stage, `task_loss_weight_scope: all`, events at 12.5%
> of the gradient) is **not** distinguishable from B. The event weighting that showed
> +0.013 on `mix_natural`'s own blind test does not carry through a further fine-tune.
>
> Hygiene, checked before the run: RAMS splits mutually disjoint, and `mix_natural`
> shares **zero** documents with any RAMS split, so B and C never saw the RAMS test set.

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
   (0.050 → 0.158) and is still climbing; boundary is **flat within measurement noise**
   (0.177 → 0.202 → 0.192, spread 0.025 against ±0.02 single-run variance). The span head
   spends the whole curve recovering from a low floor; the boundary head starts near its
   ceiling and stays there. *Originally written as "boundary moves +14%" — that trend does
   not survive the control; see the box above.*

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

The full Re-DocRED arm, at matched thresholds (the cards disagree because the three
models calibrated to 0.1 / 0.3 / 0.1):

| threshold | 10K base | 40K base | 100K base |
|--:|--:|--:|--:|
| 0.1 | 0.1637 | 0.1777 | **0.1905** |
| 0.3 | 0.0890 | 0.1288 | **0.1302** |
| 0.5 | 0.0301 | 0.0422 | **0.0437** |

Monotonic at every operating point — more base volume does help relations — but the
increments are small: **+8.6%** then **+7.2%**, for 10× the Stage-A data.

Set against the base curve, that is the interesting part. Without a Re-DocRED fine-tune
the bases score 0.007 / 0.012 / 0.073; with one they score 0.164 / 0.178 / 0.191. So the
**downstream fine-tune supplies almost all of the relation capability, and base volume
contributes ~16% on top of it.** Relations and events therefore fail in opposite ways:

- **Events** — the *architecture* carries it. The boundary head reaches at 10K what the
  span head needs >100K for, and its curve is nearly flat (§4b).
- **Relations** — the *downstream data* carries it. Base warming moves the number
  6× when there is no fine-tune, and barely 16% once there is one.

The practical read for §1's two-faced thesis: an event model wants the right head, a
relation model wants the right fine-tune, and neither is fixed by pouring more mixed
Stage-A data into it.

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

### The curve completes (2026-08-10) — 12/12 arms, and the warm start saturates

All twelve arms finished on one H100 (~23h wall clock). Strict micro-F1, each family at
its own calibrated threshold — **which is the first thing to check, not the last**:

| family | 10K | 40K | 100K | 137K | thresholds |
|---|--:|--:|--:|--:|---|
| mmbert (cold base) | 0.010 | 0.039 | 0.079 | **0.098** | 0.3 / 0.3 / 0.3 / 0.3 — matched |
| rams (warm start) | 0.177 | 0.191 | **0.202** | 0.192 | 0.3 / 0.3 / 0.3 / 0.3 — matched |

(`event_argument`. Base `relation` over the same points: 0.007 → 0.012 → 0.073 →
**0.170**; base `entity`: 0.311 → 0.402 → 0.459 → **0.586**; base `event_type` reaches
**0.956**.)

> **RETRACTED 2026-08-10, same day.** The "divergence" claimed here was an artifact of
> comparing rows evaluated on DIFFERENT blind tests, and the cold-base 137K point is not
> comparable to its own row. See "The support check that should have come first" below.
> What survives is the RAMS row alone: it saturates from 100K.

~~**The headline is a divergence.** The cold base is still climbing at 137K with no
plateau; the RAMS warm start is flat from 100K.~~ The warm start **is** flat from 100K
(0.202 → 0.192, consistent support 2,016) — treat that as saturation rather than decline:
single seed, ~5% relative, inside seed noise. Everything said here about the COLD BASE
climbing, and about the two arms having different exhaustion points, is withdrawn.

### The support check that should have come first

`test_metrics.json` records a `support` per family -- the number of gold instances the
metric was computed over -- and comparing it across arms is a two-line check that was not
run until after the curve was written up. It should have been the first thing done, because
it invalidates half of it:

| family (strict support) | mmbert 10k/40k/100k | mmbert 137k | rams (all) | redocred (all) |
|---|--:|--:|--:|--:|
| entity | 7,896 | **66,624** | -- | 10,705 |
| relation | 902 | **5,281** | -- | 17,348 |
| event_type | 433 | **4,004** | 848 | -- |
| event_argument | 3,527 | **20,845** | 2,016 | -- |

The mmbert arms were run across multiple attempts (`status.attempt1..4.tsv`) spanning the
9 Aug fix that added the missing event `test:` keys, so the first three used the small
blind test and 137K used the corrected one. **The cold-base row is therefore not a curve** --
three points on one test set and a fourth on another -- and the mmbert-versus-rams
comparison never shared a test set at all.

Valid: the RAMS row (support 2,016 throughout) and the Re-DocRED row (10,705 / 17,348
throughout). Invalid: the cold-base 100K→137K step, and every cross-row comparison.

The lesson is exactly the matched-threshold lesson one level up. Thresholds were checked
because a previous result had been overturned by them; **support** was not, and it is the
same class of error -- two numbers that look comparable and are not. A curve config should
assert constant support across its points and fail if it moves.

### Re-DocRED's "noise" was the threshold alternating — matched-threshold rule, again

The Re-DocRED family looked erratic: relation 0.176 → 0.136 → 0.207 → 0.176, entity
0.564 → 0.656 → 0.622 → 0.694. Neither is monotone, and the obvious reading is noise.

It is not noise. The calibrated thresholds **alternate**: 0.1, 0.3, 0.1, 0.3. Split the
series by threshold and both halves rise monotonically:

| | 10K (0.1) | 100K (0.1) | | 40K (0.3) | 137K (0.3) |
|---|--:|--:|---|--:|--:|
| relation | 0.176 | **0.207** | | 0.136 | **0.176** |
| entity | 0.564 | **0.622** | | 0.656 | **0.694** |

So Re-DocRED scales with base data like everything else, and the non-monotonicity was an
artefact of reading four points at two operating points. **This is the second time the
matched-threshold rule has changed a conclusion in this experiment** — the first was the
apparent warm-start regression above.

Two process fixes follow, both cheap:

1. `metric_sweep: true` (select each checkpoint at its own best threshold) is right for
   shipping a single model and **wrong for a curve**. A curve config should pin one
   threshold across its points, or emit both.
2. `test_metrics.json` records no threshold, so comparability cannot be checked from the
   metrics alone — it took a separate pull of `threshold_sweep.json` to establish that
   mmbert and rams were in fact matched. The threshold belongs *in* the metrics file.

The cross-model warning already added to `model_card.py` is what prompted the check here,
and it earned its place.

## 4c. Phase A actually ran (2026-08-10) — and the beam is not the story

First run of the beam arm, on `joint-boundary-redocred-137k`: 96 relation types, one trained
model, eval-time `decode_mode` switch, full 500-doc Re-DocRED test at threshold 0.5.

| | greedy | joint (W=16) |
|---|--:|--:|
| relation strict F1 | 0.0740 | **0.1803** |
| entity strict F1 | 0.6960 | 0.6786 |

**That +0.106 is not a beam win, and it must not be quoted as one.** 0.5 is near greedy's
worst operating point; §4b's own table has greedy at **0.176** at threshold 0.3, which is
a tie with the joint arm. **This is the third time the matched-threshold rule has changed
a conclusion in this experiment.** It should now be treated as a standing rule rather than
a lesson relearned: *no arm comparison is readable until both arms are at their own
swept threshold.*

Three findings survive that caveat.

**(a) Beam width should be 1.** Slice sweep over W ∈ {1,2,4,8,16,32,64}, relation strict F1:

| W | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|--:|--:|--:|--:|--:|--:|--:|
| F1 | **0.2406** | 0.2290 | 0.2260 | 0.2211 | 0.2170 | 0.2152 | 0.2058 |

Monotonically decreasing. Widening drops predictions 157 → 117, of which **18 were
correct** — 45% precision on the dropped set against 61% overall, so precision rises while
F1 falls. Entity metrics are byte-identical at every width: `_finish_nodes` admits every
positive-score node regardless of beam state, so width touches only edges.

This is score-vs-F1 divergence. The beam maximizes the objective *better* as it widens —
`beam.py:94` even keeps greedy as a floor, so the score is monotone — and the objective is
not F1. **A better search on a mis-specified objective is worse output.**

Independent support: OneIE ([Lin et al. 2020](https://aclanthology.org/2020.acl-main.713/))
used θ=10 with β_v = β_e = 2 capping branching at 2 per step; the released package defaults
to 5. Two independent global-IE decoders landed at 5–10 where our optimum is 1. The
mechanism differs — their β caps the *label* dimension, and their beam has no greedy floor
so a narrow width there risks falling *below* the local baseline — but the direction agrees.

**(b) The gain is the formulation, not the search.** W=1 barely searches and wins outright.
The working contrast is **independent thresholding vs constrained joint selection**, not
greedy vs beam. Decision 5 frames Phase A as "paired greedy vs beam"; that framing is
mis-specified and the finding should be reported under the corrected one.

**(c) A β-style label cap is NOT the next move.** It was considered and rejected: β is a
pruner, and the joint arm sits at P=0.61 / R=0.15 — precision is 4× recall, so removing
candidates targets the axis already being won. The span-dimension caps also already exist
(`relation_heads_per_type=32`, `relation_tails_per_type=32`, `relation_pair_cap=128`), and
compute is not binding (W=64 ran in the same wall clock as W=1). The genuinely uncapped
label dimension — how many of the 96 relation types one span may join — is real but cutting
it costs recall. On the event side the analogue would bite only on **list** roles (scalar
roles already get one-per-slot from `exclusion_keys`), and the known event failure is
under-generation (anchorless: 1/9 instances), not over-generation.

Running the arm is also what exposed the fixed `decision_threshold=0.5` — see the
arm-comparability caveat, item 3, where this document had predicted the issue and called
it "moot". Wall clock 1.5× greedy on a clean slice.

**Best-vs-best (slice, each arm at its own best threshold) — settled 2026-08-10 after the
threshold fix.** Both arms peak at 0.2; greedy is unaffected by the fix (it only touches
`joint_decode`), so its pre-fix curve stands.

| threshold | greedy | joint W=1 |
|---|--:|--:|
| 0.1 | 0.2785 | 0.3270 |
| **0.2** | **0.2835** | **0.3357** |
| 0.3 | 0.2270 | 0.3264 |
| 0.4 | 0.1648 | 0.2828 |
| 0.5 | 0.0980 | 0.2406 |

**Joint wins by +0.052 (+18% relative) at best-vs-best**, and beats greedy at every
threshold on the grid. Real, but a third of the +0.106 the fixed-0.5 comparison implied.
Remaining: confirm on the full 500-doc test.


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
