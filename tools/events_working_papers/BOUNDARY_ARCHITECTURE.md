# The Boundary Architecture — How It Works, End to End

Status: reference (implementation-verified 2026-09-08 against `merge/main-20260805`;
counts, §6 cardinality, §11 decode result and §13 items 10-11 re-checked against the code
that day — every number in §12 was recounted rather than carried forward).
Companion to [[COUNTING_LAYER]] (why the span head is a dead end) and
[[BOUNDARY_DECODE_AND_EKF]] (where global inference plugs in). Where this document
and a code comment disagree, the code wins — every claim below was checked by
running the code, not by reading comments.

## Contents

1. [Why it exists](#1-why-it-exists)
2. [The core idea: boundaries, not spans](#2-the-core-idea-boundaries-not-spans)
3. [The pipeline](#3-the-pipeline)
4. [The query protocol (markers and layout)](#4-the-query-protocol-markers-and-layout)
5. [NER / entities](#5-ner--entities)
6. [Structures (json_structures)](#6-structures-json_structures)
7. [Relations](#7-relations)
8. [Events](#8-events)
9. [Classifications](#9-classifications)
10. [Losses](#10-losses)
11. [Decode modes: greedy vs joint beam](#11-decode-modes-greedy-vs-joint-beam)
12. [What in GLiNER2 uses this today](#12-what-in-gliner2-uses-this-today)
13. [Gotchas that have cost real money](#13-gotchas-that-have-cost-real-money)

---

## 1. Why it exists

The span architecture (`GLiNER2` / `SpanExtractor`) enumerates candidate spans up to a
fixed `max_width` (default 20 subwords) and scores each one. Two consequences:

- **A hard length cap.** Anything longer than `max_width` is unrepresentable. Document
  level arguments ("us , and the world") routinely exceed it.
- **A hard instance cap.** The count-first structure decode caps records per document
  (the "span 19-instance cap" of [[COUNTING_LAYER]]), which makes dense relation sets and
  multi-instance events impossible regardless of model quality.

The boundary head removes both. It never enumerates spans, so there is **no width axis
anywhere in the model** — `model.py`'s docstring states it plainly: *"Half-open
`[start, end)` coordinates throughout; there is no width axis."* A `max_width` in a
boundary config is not merely ignored, it is a category error (the validator names it
`span_head.max_width`).

## 2. The core idea: boundaries, not spans

For a text of `n` word tokens there are `n + 1` **boundaries** `0..n`. Boundary `i` sits
between token `i-1` (its left) and token `i` (its right). The first boundary uses a
learned BOS left-state, the last a learned EOS right-state (`encoding.py`).

A span is then a *pair of boundaries* `[start, end)`, and the model predicts:

- `start_logits[q, i]` — does query `q` start at boundary `i`?
- `end_logits[q, j]` — does query `q` end at boundary `j`?
- `inside_logits` — is a token inside a `q` span? (kept as a **cumulative prefix sum of
  length `L+1`**, never an `[L, L]` matrix)

All three are scaled dot-product attention between projected boundary/token states and
projected query states (`heads.py`). Cost is linear in `L` per query; the largest tensor
materialized is `[B, Q, Ks, end_block_size]` (`proposal.py`), never `[L, L]`.

**Why start/end are multi-label BCE and never a softmax:** nested and overlapping spans
legitimately share a boundary ("Bank of England" and "England" share an end). A softmax
over positions would force them to compete. `losses.py` makes this explicit.

**There is deliberately no constraint on `end - start`.** `proposal.py`: *"a start at `0`
may pair with an end at `L`."* That is the whole point.

## 3. The pipeline

```
text ──► encoder (mmBERT/DeBERTa/…) ──► token states
schema ──► marker tokens ──► query states           (one query per extractive field)
                    │
                    ▼
        BoundaryEncoder: token states ──► n+1 boundary states     (encoding.py)
                    │
                    ▼
        BoundaryQueryHead: start / end / inside marginals         (heads.py)
                    │
                    ▼
        Proposal: top-K starts, top-K ends per query, then
        conditional end-given-start (and start-given-end, if
        bidirectional_proposals) scored in streaming blocks       (proposal.py)
                    │
                    ▼
        Rerank: one scalar logit per surviving candidate           (scoring.py)
                    │
        ┌───────────┼───────────────┬────────────────────┐
        ▼           ▼               ▼                    ▼
    entities    relations       records/instances     classifications
    (mentions)  (typed edges)   (natural/latent/       (own head, no
                (relations.py)   anchorless)            candidate slot)
                                 (records.py)
```

Two candidate-pool modes (`candidate_pool`):

- **`per_query`** (default) — candidates are proposed per query.
- **`shared`** (`pool.py`) — boundary marginals stay query-conditioned but span pairs are
  formed **once per document**, then reranked per query. Internally `[B, C, Q]`;
  `PooledCandidates.to_candidate_batch` is the explicit adapter back to the public
  per-query `[B, Q, C]` contract. This is what makes document-level decoding affordable.

**Reranker prior convention (Finding 7, worth not re-breaking):** the reranker's prior is
the proposer's *marginal-free* endpoint compatibility (`proposals.compat_logits`), **not**
`proposals.logits` — the latter already folds in the start/end marginals and would
double-count them.

## 4. The query protocol (markers and layout)

The schema is serialised into marker tokens; each **extractive** marker becomes one query,
and query `q` owns candidate slot `q`. Verified marker map:

| marker | meaning | extractive? |
|---|---|---|
| `[P]` | parent / group name (event type, relation name, structure name) | no (names the group) |
| `[E]` | entity type | **yes** |
| `[C]` | **json-structure field** | **yes** |
| `[R]` | **relation `head` / `tail`** | **yes** |
| `[V]` | event role (trigger + arguments) | **yes** |
| `[L]` | classification label | **no** — classifications take no candidate slot |

> **Correction worth flagging:** earlier notes in this program described `[C]` as
> "classifications" and `[R]` as "json-structure fields". That is backwards. `[C]` is
> structure fields, `[R]` is relation endpoints, and classification labels are `[L]`.
> `_EXTRACTIVE_MARKERS = ("[E]", "[C]", "[R]", "[V]")` in `boundary_preprocessing.py`
> is the authority; `[L]` is correctly absent.

**Every marker that takes a query slot must be in `_EXTRACTIVE_MARKERS`**, or the layout
under-counts queries while `query_states`/`query_mask` still carry one entry per marker —
producing a shape crash at gold injection. `[V]` was missing once; every event group
produced zero layout queries and boundary+events training was impossible.

Real outputs from the processor (`collate_fn_inference(..., architecture="boundary")`):

```
NER        ( [P] entities ( [E] person [E] org ) )
           queries: (entities, entities, person), (entities, entities, org)

STRUCTURE  ( [P] order ( [C] item [C] buyer ) )
           queries: (json_structures, order, item), (json_structures, order, buyer)

RELATION   ( [P] works_for ( [R] head [R] tail ) )
           queries: (relations, works_for, head), (relations, works_for, tail)

EVENT      ( [P] Attack ( [V] trigger [V] target [V] instrument ) )
           queries: (events, Attack, trigger), (events, Attack, target),
                    (events, Attack, instrument)

CLASSIF    ( [P] sentiment ( [L] pos [L] neg ) )
           queries: (none — no candidate slot)
```

Note the event group: **`trigger` is auto-prepended as field 0.** The user schema is
`{"Attack": ["target", "instrument"]}`; the trigger query is synthesised.

`ext_specs` enumerates exactly the extractive queries in candidate-slot order, so
`query_id == candidate query index`. `_layout_from_ext_specs` (`engine.py`) derives the
`QueryLayout` from it rather than from `batch.query_layouts` — only the fast-routing path
populates the latter, so deriving from `ext_specs` keeps mention keys and edge keys
referencing the same source by construction.

## 5. NER / entities

**Schema** `{"entities": {"person": "", "org": ""}}`
**Tokens** `( [P] entities ( [E] person [E] org ) )` — one query per type.

**Training.** Gold surfaces are aligned to word tokens, converted to half-open boundary
pairs, and become `MentionTarget(query_id, start, end)`. Supervision is per-query
multi-label BCE on start/end plus the pair/rerank losses (§10). A surface that cannot be
aligned is either an error (`on_missing_surface="raise"`) or is skipped
(`"skip"`, implied by `error_policy` other than `raise`) — skipping loses that *mention*,
not the record.

**Evaluation.** `compute_metrics` reads gold `entities` and predicted `entities`, scoring
strict `(label, surface)` and relaxed (label exact + surface overlap) micro/macro F1.

**Extraction.** `engine.py::_extract_from_batch` walks each spec with
`task_type == "entities"`, resolves its candidates to character-anchored spans
(`_query_spans`), and emits `sample["entities"] = [ {type: [spans]} ]`. Abstention
(`null_logits > abstention_threshold`) yields `[]` for that type — distinct from
"decoded and found nothing".

## 6. Structures (`json_structures`)

**Schema** `{"json_structures": [{"order": {"buyer": "", "item": ""}}]}`
**Tokens** `( [P] order ( [C] item [C] buyer ) )`

Structures are where the **record/instance machinery** actually engages — but only when
the schema declares it:

```python
{"json_structures": [{"order": {...}}],
 "record_metadata": {"order": {"mode": "natural", "anchor": "buyer"}}}
```

Without `record_metadata`, `compile_record_specs` returns `{}` and the group decodes as
independent per-field spans. With it, you get a `RecordSpec` and true multi-instance
records. Verified:

```
plain           record_specs: [{}]
record/natural  record_specs: [{0: ('order', 'natural', ['item', 'buyer'])}]
```

Three modes (`records.py` — *"Instance Formation and Record Disambiguation"*, replacing
count-first decoding):

| mode | instance seed | use when |
|---|---|---|
| **natural** | every detected **anchor** candidate seeds one instance; each non-anchor field candidate is scored against every instance with an explicit `ABSENT` alternative | a field naturally identifies the record (e.g. `buyer`) |
| **latent** | no declared anchor — a learned selector scores each candidate as a potential seed, supervised only by record grouping | records exist but no field is a natural key |
| **anchorless** | document-conditioned learned instance queries cross-attend candidate states, predicting object/`NO_OBJECT` plus per-field pointers | neither of the above |

**Decode thresholds are separate from the extraction threshold.** `decode_group` uses
`record_anchor_threshold` (0.5) and `record_field_threshold` (0.5) from the *saved config* —
the `threshold=` you pass to `batch_extract` does **not** move them. A threshold sweep that
only varies the extraction threshold will not change record output at all.

**A field's `dtype` declares its CARDINALITY, and cardinality reaches much further than
output shape.** `dtype: str` compiles to `OPTIONAL_ONE` (scalar), anything else to
`ZERO_OR_MORE` (list); an anchor is always `REQUIRED_ONE`. That one flag selects the
training loss (`_scalar_field_nll`, a softmax over candidates plus `ABSENT`, against
`_list_field_bce`), the occurrence policy (`_apply_occurrence_policy` returns early for
list fields), the joint beam's utility (`logit - absent` against bare `logit`) and its
exclusivity slot, and finally the emitted shape (`{"text": ...}` against `[{"text": ...}]`).

**Until 2026-09-08 the dtype never arrived.** `compile_record_specs` has always accepted
`field_dtypes`, and `build_boundary_batch_metadata` has always declared
`field_dtypes_list` — which **no caller ever passed**. So every non-anchor field compiled
`ZERO_OR_MORE` whatever its schema said. Only the greedy decoder looked correct, because it
re-read the dtype at format time (`is_scalar = fspec.cardinality.is_scalar or
dtype == "str"`) and patched the shape back. Dtypes now travel *with* the schema
(`Schema.build()` emits `field_dtypes`; `structure_field_dtypes` also reads the raw
`[{"name": ..., "dtype": ...}]` form), the processor passes them, and that rescue is gone
because it is now provably redundant. See §11 for what it cost.

## 7. Relations

**Schema** `{"relations": [{"works_for": {"head": "", "tail": ""}}]}`
**Tokens** `( [P] works_for ( [R] head [R] tail ) )` — two queries per relation type.

Relations **reuse the entity mention candidates** rather than introducing a second
extraction representation. Pair generation is *typed and capped* (`relations.py`): for each
relation type keep the top-`Rh` head-typed and top-`Rt` tail-typed mentions and score their
capped cross product — `O(Rh·Rt)` per type with fixed caps, **never the `O(N²)` all-pairs
matrix**. This is what makes dense Re-DocRED-style documents tractable.

**Training.** Gold `(head_span, tail_span)` pairs become `edge_targets`; `relation_loss`
supervises the typed pair scores.
**Evaluation.** Strict `(name, head, tail)`; relaxed = name exact + head/tail overlap.
**Extraction.** `_decode_relations` emits `sample[relation_name] = [...]`, which the runtime
formatter routes into `relation_extraction`.

## 8. Events

**Schema** `{"events": {"Attack": ["target", "instrument"]}}`
**Tokens** `( [P] Attack ( [V] trigger [V] target [V] instrument ) )`

This is the subtlest task, and the one where the implementation and the intent diverged.

**Events are supervised as MENTIONS, not as records.** An events schema never produces
`record_metadata` (that is built only from `json_structures`, in `training/data.py`), so
`compile_record_specs` returns `{}` for event groups — **in training as well as inference**.
Verified: a RAMS training record collates to `record_specs: [{}]`, `targets.records: None`,
but **14 mentions across 3 documents**. So:

- `enable_records: true` is **inert for events**. The record head never sees them.
- What the model actually learns is one extractive query per trigger and per role — i.e.
  event extraction is per-role span extraction that happens to share a group name.
- Multi-instance separation ("two attacks in one document") is therefore **not currently
  learnable for events** — that genuinely needs the record head.

**Extraction** assembles events from those mention queries (`engine.py::_decode_events`),
mirroring the span engine's shape:

```json
{"Attack": [{"triggers": ["bombed"],
             "arguments": [{"role": "target", "entity": "the depot"}]}]}
```

Field 0 of a group is the trigger; the rest are roles; a group with no trigger span is
dropped (gold arguments are keyed by trigger, so a trigger-less instance cannot match).
One instance per event type — the mention path has no instance dimension.

**This assembly did not exist until 2026-08-09.** Before it, `_extract_from_batch` skipped
every non-entity query on the assumption the record head would decode it. Since the record
head is inert for events, nothing did — so **every event metric read 0.0000 at every
threshold, down to 0.01**, on models that had in fact learned events. Fixing the assembly
alone took one unchanged checkpoint from `0.0000` to argument F1 `0.182` / trigger `0.764`
/ type `0.946` ([[JOINT_IE_SCALING]] §4b).

**Evaluation** pairs gold `events` with predicted **`event_extraction`** (they are
deliberately different key names) and scores four families: `event_type`,
`event_trigger`, `event_argument`, and a combined `event`.

## 9. Classifications

`( [P] sentiment ( [L] pos [L] neg ) )`. Classifications run on their own head over query
states and **take no candidate slot** — which is why `[L]` is absent from
`_EXTRACTIVE_MARKERS`. `_decode_classifications` writes the label (or label list) directly
into the sample. `classification_loss` is a separate term.

## 10. Losses

`_compute_losses` returns a dict; `total_loss` is the weighted sum. Terms:

| term | what it supervises |
|---|---|
| `start_loss`, `end_loss` | boundary marginals, multi-label BCE (never softmax) |
| `inside_loss` | inside-span consistency |
| `proposal_loss` | listwise: did the proposer surface the gold pair? |
| `pair_loss` | candidate pair correctness |
| `soft_iou_loss` | soft-IoU over candidate pairs (`candidate_pair_loss`) |
| `rerank_listwise_loss` | listwise reranking of surviving candidates |
| `classification_loss` | classification head |
| `record_object_loss`, `record_field_loss` | instance existence + field assignment (structures only, see §8) |
| `relation_loss` | typed relation edges |
| `null_loss` (`abstention_loss`, weight **0.2**) | a per-query gate whose target is **1 for an ABSENT query** |
| `count_loss` (`count_log_rate_loss`, weight **0.2**) | how many spans a query should yield; supervises a count of **zero** for an absent one |

All are **masking-aware and empty-query safe**: denominators use `clamp_min(1)`, so a query
with no positive span still contributes finite negative supervision rather than `0/0`.

### Two terms were live and starved: the ABSENT-QUERY machinery

`null_loss` and `count_loss` above, plus `negative_query_ratio` (**0.5**) and
`max_negative_queries_per_batch` (**64**) at `models/boundary/model.py:845`, all act on
**absent queries** — a label offered in the schema with no gold in that document.

**Until 2026-09-16 there were none.** `InputExample.from_dict` built the entity menu from the
gold dict's keys and one `Event` per gold event, so every query carried gold: measured, **0
absent queries in 574** from a real training batch. The selection block above has therefore
run in every model this project has trained and has always selected from an empty set, and
`abstention_loss`'s positive class was empty. Classification was the sole exception — it
carries `labels` (the full menu) plus a separate `true_label`, which is why its precision is
real where `event_type`'s is 1.0000 by construction.

The consequence is measurable at decode: given a schema of only ABSENT event types, the
incumbent fires on **63 of 100 documents**, and its real `event_type` precision against a
full menu is **0.5521**, not the 1.0000 the blind test reports.
See [[LABEL_NEGATIVES_PLAN]] and [[EVENT_ARGUMENT_DIAGNOSIS]] §4h/§4i.

**An absent entity query is `{"label": []}`** — the label mapped to an empty list, the shape
`GuideScores.inject` already emitted for GIST rivals, so the collator, target builder and
losses accept it end to end. Events needed real work: the training list is menu and answer
key at once and `_process_events` SKIPS an event with empty triggers, so absent types ride an
`absent_events` key emitting `labels.append([0, []])`. Relations use `absent_relations` (a
name list — the inference shape `{name: {"head": "", "tail": ""}}` is read as a GOLD pair of
empty surfaces), and absent structures must carry `record_metadata` or `compile_record_specs`
builds no spec and nothing decodes them.

### Three places with NO GRADIENT, and only one of them is a defect

Easy to conflate, so stated separately (checked 2026-09-17):

| component | gradient? | verdict |
|---|---|---|
| **Hungarian matching** (record loss) | **yes, flows** | correct |
| **`shared_pool_builder`** under `candidate_pool: per_query` | no — never called in the loss | by design; measured 0.000e+00 grad norm against 8.890e+00 under `shared`, and that pair is the A/B's gate |
| **the joint beam** (`decode_mode: joint`) | no — eval-time only | by design, but see below |

**Hungarian is implemented correctly** and is worth checking rather than assuming, because
this is exactly where a blocked gradient would hide. `field_nll` is computed from the
grad-carrying `assign_logits` FIRST, and only then does a `with torch.no_grad():` block build
the matching cost and run the assignment (`records.py:1230-1245`). The assignment is
non-differentiable and is supposed to be; the loss it selects is not. Standard DETR pattern.
`linear_sum_assignment` detaching its cost matrix is part of that, not a bug.

**`shared_pool_builder` gets no gradient under `per_query` simply because nothing calls it in
the loss** — its only other call site is behind `if diagnostics:` and feeds stats. Its 64
tensors sit at initialisation in every `per_query` checkpoint.

**That fact was misfiled for months, and the correction is worth stating precisely, because
the two claims it ran together are both real and only one of them applies.**

*It cannot be swept at EVAL* — true, and unchanged. Flipping the flag over a trained
checkpoint scores freshly-initialised weights and reads as catastrophic when the honest
answer is "untrained". `candidate_pool` is correctly absent from `_EVAL_TIME_BOUNDARY_KEYS`
and stays absent.

*It is STRUCTURAL* — **false**, and it was listed as such, which blocked the only legitimate
way to ask the question: TRAIN an arm with the flag on. Measured 2026-09-17 by building the
head both ways — 340 tensors under `per_query`, 340 under `shared`, none added, removed or
reshaped. The module is constructed unconditionally at `model.py:246`, and the comment there
says exactly that. A flag whose tensors are in every checkpoint cannot be structural; the
sentence this paragraph used to end with contained its own refutation.

The confusion is a general one and worth naming: **"changes the shapes" and "invalidates a
warm start scientifically" are different properties, and only the first belongs in a guard
that exists to prevent a state-dict mismatch.** An untrained submodule is a handicap to
report, not a load error to refuse. The cost of collapsing them was one A10 hour on an A/B
whose treatment arms never started.

**The joint beam is the interesting one.** It decodes JOINTLY over candidate scores trained
GREEDILY — nothing in training ever optimises for the beam's constraint satisfaction, so the
search is layered on scores that never knew it was coming. That is a genuine train/test
mismatch, and §11's measured result is consistent with it: **joint matches greedy on all seven
heads at 2.5x the wall clock**. A beam that buys nothing is what one expects when the scores
beneath it were never trained toward joint decoding. The remedies — a beam-aware or structured
loss, or a differentiable relaxation — are research directions, not config changes.

### The loss has no task axis

Every query — entity, relation, event trigger, event role, structure field — flows through
the **same** terms. The decomposition above is by *mechanism*, not by task, so "is the event
signal too small?" is not answerable by reading any of these numbers. That is a real
difference from the span architecture, whose loss decomposed by task.

Three settings exist to see and change the balance (all inert at their defaults):

| setting | effect |
|---|---|
| `report_task_losses` | split every query-typed term into per-task **contributions** that sum back to the term. Diagnostic only, no gradient effect. |
| `task_loss_weights` | scale a task's whole term — magnitude |
| `task_loss_weight_scope` | what that reaches: `span` (start/end/pair, **18.5%** of the loss) or `all` (adds inside, soft-IoU, rerank, proposal, abstention, count — **94.3%**) |
| `task_pos_weights` | scale positives against negatives *inside* a task's queries — direction. start/end/pair/inside only; not soft-IoU (fractional targets), and **ignored** by the `asymmetric_focal` marginal path, which never calls `_safe_bce`. |

Measured on a converged warm-start checkpoint: entities hold **77.2%** of the training
gradient, json_structures 11.5%, events **6.6%**, relations 2.5%. Use
`tools/train/probe_task_losses.py` before tuning any of the above — a dose sweep that
cannot reach most of the objective produces a null that says nothing about the hypothesis.

Two terms behave oddly enough to note. `count_loss` is Poisson NLL with `full=False`
(`exp(x) - t*x`), which is **unbounded below**, so a per-task contribution can be negative
and upweighting a task can *lower* the reported value while its gradient scales correctly.
`abstention_loss` barely responds to an event weight because event queries almost always
have a mention to find (events hold 0.00001 of its 0.00449 mass).

Diagnostics: the trainer records *which* terms went non-finite and reports them on flush
(`… -- offending terms: start_lossx12, end_lossx12, …`). When every boundary term goes
non-finite together it indicates a shared upstream (encoder/scores), not one loss's
normalisation — that is exactly how the sdpa+bf16 defect in §13 was localised.

## 11. Decode modes: greedy vs joint beam

`decode_mode` (config, **eval-time**) selects the arm over *one trained model*:

- **`greedy`** (default) — per-query set prediction; records via `decode_group`; relations
  via typed capped pairs.
- **`joint`** — candidates become a `JointProblem` and a global beam picks a
  constraint-consistent assignment (`joint_ie/`). In joint mode records come out of the
  beam via role edges, so `_decode_records` is skipped to avoid double emission.

Baking `decode_mode` into a *training* config would make the two arms different models and
void the comparison — it belongs at eval only.

**`joint_beam_width` should be 1.** Measured on Re-DocRED (JOINT_IE_SCALING §4c): relation
F1 decreases monotonically from W=1 to W=64, and entity metrics do not move at all, because
`_finish_nodes` admits every positive-score node regardless of beam state — width touches
only edges. The default is still 16; a change of default should wait for best-vs-best.

> **Live discrepancy, stated so it is not rediscovered:** this section recommends **1**, the
> shipped default is **16** (`configuration.py:118`), and **no config in the tree overrides
> it** — so every model here trains and evaluates at 16 while the measurement says 1 is
> marginally better. The gap is small (structure moves 0.0018 across 4/16/64) which is why it
> has not been worth a migration, but "the default is 1" is NOT true today.

**`abstention_threshold` is an eval-time override too** (added 2026-09-16). It ships at
**0.5**, is read at decode (`engine.py:390`, `965`), adds and removes no tensors, and now sits
in the same whitelist as `decode_mode` and `joint_beam_width` — `eval.py
--abstention-threshold`. It is worth sweeping on any model trained WITH label negatives: the
gate only has a positive class once absent queries exist, and a newly-trained gate at a
shipped default is the exact shape of the stage-0 gate that ran its whole life at 0.5 and
needed 0.998.

**The threshold reaches edge selection** (fixed 2026-08-10). `decision_threshold` sets where
utility crosses zero and the optimizers take only positive-utility candidates, so a
hard-wired 0.5 does not merely miscalibrate — it makes the whole decode ignore
`--threshold`. Record **role edges deliberately bypass** it (`pre_scored_edges`): a scalar
role's utility is the ABSENT-relative `logit_c - logit_ABSENT`, which has no probability
cutoff to be centered on.

**The measured answer.** One checkpoint (`eb16-rebuild-tr`), 18,786-record blind test,
`decode_mode` the only variable: **all seven heads inside the ±0.02 floor**, at ~2.5× the
wall clock (8.5 min against 22). A 4/16/64 width sweep moves structure by 0.0018, and
*narrower* is marginally better — the opposite of a search-capacity story. So the beam buys
nothing here and costs 2.5×.

**THE NULL WAS MEASURED WITH `event_records` OFF, AND THAT SCOPING MATTERS (2026-09-17).**
`eb16-rebuild-tr` decodes events through the MENTION path. With `event_records: true` events
are RECORDS -- and joint mode **skips the record decoder entirely** via an early return
(`engine.py:250-259`): "in joint mode records come out of the beam via role edges; running the
record head too would double-emit them."

Measured per document on `eb16-eventrecords-tr` (`event_records: true`), 60 cmnee documents,
`probe_beam_disagreement.py`:

| | |
|---|---:|
| documents where joint and greedy DIFFER | **42 of 60 (70%)** |
| items greedy produced that joint dropped | **70** (29 of them CORRECT) |
| items joint added that greedy lacked | **16** (6 correct) |
| **net for joint** | **-23 correct items** |
| **triggers: dropped / added** | **35 / 0** |

**The beam is NOT inert -- it differs on 70% of documents -- and it is net negative.** The
dominant mechanism is not substitution but DELETION: 70 items dropped against 16 added, and for
triggers it adds none at all. The beam's role-edge path recovers less than the record decoder
it replaces.

Where it does substitute, right-rates are close (41.4% greedy-only against 37.5% joint-only) --
the coin-flip signature, meaning the candidate scores carry little information about which
constraint-consistent assignment is correct. That half is what a beam-aware or structured loss
would address, but it is the SMALLER effect, so it is not the first thing to build.

**Practical consequence: do not pair `decode_mode: joint` with `event_records: true`** until the
role-edge path is shown to recover what `_decode_records` does. The null above does not license
it -- that null was measured on a model where events never went through records.

**The first version of that answer had structure at −0.0454, and it was the plumbing in
§6.** With `dtype: str` compiling `ZERO_OR_MORE`, the ABSENT-relative utility and the
exclusivity slot never engaged for any structure field, and greedy's format-time rescue
masked the same defect on its own side. Connected: greedy 0.1208 → **0.1640**, joint
0.0754 → **0.1582**, deficit −0.0454 → **−0.0057**. The joint arm gained nearly twice what
greedy did, because greedy had the rescue and joint had nothing. **87% of the deficit was
a parameter no caller passed.**

A null at parity is a better negative than a loss: it cannot be explained away as a defect,
and this one nearly was — in the other direction.

## 12. What in GLiNER2 uses this today

**Core (`gliner2/models/boundary/`)** — 20 modules: `model.py` (the head bundle +
`BoundaryExtractorModel`), `encoding.py`, `heads.py`, `proposal.py`, `scoring.py`,
`pool.py`, `records.py`, `relations.py`, `losses.py`, `content.py`, `rotary.py`,
`engine.py` (decode/extract), plus small adapters.

**Depends on boundary:**

| module | how |
|---|---|
| `gliner2/auto.py` | `AutoExtractor` dispatches `architecture="boundary"` — the only correct loader |
| `gliner2/configuration.py` | `BoundaryHeadSettings` (92 fields), `attn_implementation`, `decode_mode`, `joint_beam_width` |
| `gliner2/processor.py` | marker emission, `_add_boundary_metadata`, record-metadata **and field-dtype** plumbing (§6) |
| `gliner2/processing/boundary_preprocessing.py` | `_EXTRACTIVE_MARKERS`, layouts, targets, record specs |
| `gliner2/processing/targets.py`, `records.py`, `layouts.py` | target/spec compilation |
| `gliner2/training/trainer.py` | boundary defaults to bf16, per-term non-finite diagnostics |
| `gliner2/training/matching.py` | record matching cost |
| `gliner2/training/lora.py` | LoRA target modules |
| `gliner2/inference/engine.py`, `runtime.py` | shared extract path + output formatting |
| `gliner2/joint_ie/candidate_scores.py` | boundary candidates → `JointProblem` |

**Training configs:** 77 YAMLs declare `architecture: boundary` and 11 declare `span`;
13 more inherit the architecture from a `pretrained:` checkpoint. No live config resolves
its architecture by silent default any more — `from_encoder` defaults to `span`, so a
config with `encoder:` and no `architecture:` built a span model without saying so, and
eleven were in that state (commit `c394ae8`).
**Tests:** 50 files under `tests/models/boundary/`, including golden-parity, overfit,
invariants, and joint-decode suites.

## 13. Gotchas that have cost real money

1. **FlashAttention 2 is required on ModernBERT encoders, for correctness.** Measured on
   1×H100, 60 steps: `sdpa`+bf16 goes non-finite at step 15 (2.0 samples/s); `sdpa`+fp32 is
   clean (8.9); **FA2+bf16 is clean at 22.0**. Changing either variable removes the NaN, so
   it is the *interaction*. mmBERT is ModernBERT and is built for FA2 + unpadding. FA2 comes
   from the Hub kernel registry (`kernels>=0.12,<0.13`); prebuilt `flash-attn` wheels stop
   at cp313/torch2.9.
2. **FA2 encoders need autocast at inference too.** Weights load fp32 by design; training
   wraps the forward in autocast, inference did not, so every `batch_extract` raised
   *"FlashAttention only support fp16 and bf16"*. `BaseExtractorModel._encoder_autocast()`
   supplies it. Not reproducible off-GPU.
3. **Load with `AutoExtractor`, never `GLiNER2`.** `GLiNER2` *is* the span class; on a
   boundary checkpoint it dies at `config.max_width`.
4. **A `metric_for_best` pointing at a structurally-zero metric silently pins `best/` to
   epoch 1** — `0.0 > 0.0` is false forever. This happened with
   `eval_event_argument_strict_micro_f1` while §8's assembly was missing.
5. **A `boundary_head` override can reach `model.config` and still not reach the head.**
   `BoundaryHead` holds its **own** settings reference, built in `__init__` from the
   checkpoint's config, and copies `hard_negatives_per_positive` /
   `minimum_hard_negatives` out of it. Rebuilding only `model.boundary_settings` left
   every knob the head reads through `self.settings` — the soft_iou/rerank/proposal/count
   weights, `boundary_negative_weight`, `negative_query_ratio`, `task_loss_weight_scope` —
   pinned at the checkpoint value. Measured: a config setting `scope: "all"` produced
   `"all"` on the model and `"span"` on the head, i.e. a treatment arm identical to its
   control. Verify on `model.boundary_head.settings`, not on `model.config`.
6. **Structures are never scored by the blind test.** `_schema_from_gold` builds no schema
   for `json_structures`, so a structure-only record produces an empty schema and
   `compute_metrics` skips it entirely — 35.1% of `mix_natural`'s val. Structure quality
   comes from `tools/train/probe_records.py`, and a structure corpus can look "trained"
   while contributing nothing measurable.
7. **CUDA `Error 802: system not yet initialized` with a perfectly healthy `nvidia-smi`
   is a dead host, not a config problem.** Driver fine, both GPUs listed, no XID errors,
   modules loaded — and every context creation fails. `nvidia-fabricmanager` refusing to
   start with "Nothing to do" plus `lspci` showing zero NVSwitch and
   `GPU Fabric GUID: N/A` is the signature. A module reload and a full restart both
   changed nothing. Terminate and relaunch, ideally in another region.
8. **Record thresholds are not the extraction threshold** (§6) — sweeping one does not move
   the other.
9. **`error_policy` and `on_missing_surface` are different knobs.** The first governs
   malformed *records* in `_collate_batch`; the second governs surface *alignment*. Eval
   ignored the second until 2026-08-09 and aborted on any unalignable val mention.
10. **A parameter that exists, is threaded, is documented — and that nothing ever passes.**
   `build_boundary_batch_metadata(field_dtypes_list=...)` was declared, forwarded to
   `compile_record_specs`, and consumed correctly by `_default_cardinality`. No caller
   supplied it, so the whole chain ran on the default and every `dtype: str` field
   compiled as a list. Nothing raised, no test failed, and the *greedy* decoder looked
   right because it re-derived the dtype at format time and patched the shape back — which
   is what kept the defect invisible while it changed the joint decoder's output, its beam
   utilities, and the training loss selection. **Grep for a keyword parameter's call sites
   before trusting that it does anything**; a default that is never overridden is
   indistinguishable from a hard-coded constant. Fixed 2026-09-08 (§6).
11. **A structure with no `record_metadata` trains nothing and says nothing** — valid
   input, no error, and the rows still count as supervision in every composition print.
   60,948 rows across 13 models sat in that state. The corpora were repaired in place, and
   the *producer* was fixed separately a day later: `tools/data/synthetic/validate.py` had
   gone on emitting `json_structures` with no metadata, so every newly generated corpus
   reproduced the defect. `tools/data/audit_corpora.py` has a `STRUCT` check for it.
