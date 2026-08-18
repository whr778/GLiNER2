# Per-task event loss weighting on the boundary architecture — evaluation plan

> **⚠ 2026-08-18: every joint_ie / 137k number in this document predates a data repair
> and is superseded.** 45 corpora shipped overlapping train/val/test, and the scaling
> configs additionally paired regenerated train files against frozen val slices
> (252 train-in-val, 22 val-in-test). All were repaired, the slices rebuilt, and the
> four scaling points now gate CLEAN. The curve is being re-run from scratch — see
> `JOINT_IE_SCALING.md` and [[lambda-137k-curve-restart]]. Numbers below are kept as
> the record of what was measured and believed at the time; do not compare new results
> against them.

Phase 3 successor to [`EVENT_LOSS_PLAN.md`](EVENT_LOSS_PLAN.md), which is the **phase 2
(span) plan** and is kept as history. That plan is fully implemented on
`origin/mmbert_training` and does not port; this one states what the phase-3 analogue is,
and how to find out whether it does anything.

---

## 1. Status, stated honestly

**External evidence is the reason to run this.** The user reports that a separate event
loss was used and heavily tested on another system, where it outperformed every other
approach. That is primary-source and is not reproducible from this repo — record it as the
motivating prior, not as a local result.

**What this repo can say, and it is thinner than it looks:**

| claim | status |
|---|---|
| phase 2 implemented a separate event loss bucket | **true** — `event_struct_loss` / `event_struct_pos_weight`, `origin/mmbert_training:gliner2/model.py:56,479` |
| phase 2 *ran* with it | **false** — zero `.yaml` on that branch sets `event_struct_pos_weight`. Its configs set a **global** `struct_loss: bce_posweight` across all tasks |
| it exists on this branch | **false** — zero references anywhere in `gliner2/`. Dropped in the port onto main's boundary rewrite, silently, not by decision |
| PAPER_0 claims it exists | **true and now wrong** — `PAPER_0_FOUNDATION.md:106`, "A dedicated event loss path (§6) lets event supervision be tuned independently" |

So there is **no local measurement either way**. This is not "restore a thing that worked
here"; it is "run the experiment phase 2 set up and never ran, on an architecture where it
has to be built differently."

## 2. Why phase 2's implementation cannot be ported

Phase 2's span loss decomposed **by task**:

    entities → struct_loss | relations → struct_loss + count | events → struct_loss + count

so routing events into their own accumulator with their own variant and `pos_weight` was a
local edit. Note the plan's own words: the split itself was "a **pure bucketing change** —
the tensors are the same". The gradient lever was the *follow-on* per-task variant, not the
separation. Anyone repeating this should not expect the bucket alone to change training.

The boundary loss decomposes **by mechanism**, not by task
(`models/boundary/model.py:795`):

    total = w.start·start + w.end·end + w.pair·pair + w.inside·inside
          + soft_iou + rerank + proposal + consistency + abstention + count

Every query — entity, relation, event trigger, event role — flows through the same terms.
There is no task axis in the loss at all, and `_compute_losses`
(`models/boundary/model.py:573`) never receives `batch`, so no task-type signal reaches it.

## 3. The phase-3 analogue, and why it is small

Not a separate loss path — a **per-query task weight**. Three facts make it cheap:

1. `QuerySpec` already carries `task_type` (`models/base.py:59`).
2. `build_boundary_batch_metadata` already walks `task_types` while assigning `query_id`
   (`processing/boundary_preprocessing.py:383`), so the query→task map exists at
   construction time; it is simply never materialized.
3. **`_reduce` already reduces `[B, Q, N]` with a `query_mask` and has a `per_query` mode**
   (`models/boundary/losses.py:39`). The hook point exists — this needs a `[B, Q]` weight
   tensor multiplied in before the final mean, not a new loss function.

### Build steps (inert by default)

1. Materialize `query_task_ids: [B, Q]` (int8 over a fixed task-type vocabulary) in
   `build_boundary_batch_metadata`; carry it on the batch.
2. Thread it into `_compute_losses` and into `_reduce` as an optional per-query weight.
3. Add `task_loss_weights: Dict[str, float]` to the boundary settings, **default all 1.0**.
4. Emit per-task diagnostic buckets (`event_pair_loss`, `entity_pair_loss`, …) alongside
   the existing mechanism buckets. This is the phase-2 "bucketing" half: no gradient
   effect, but it is what makes the A/B readable, and it is worth landing regardless.

**Scope for run 1: span terms only.** Leave `count_loss` unweighted, matching phase 2's
explicit choice to leave `count_loss` combined. One variable.

**Note two things that will otherwise be misread:** classification groups emit no
extractive queries (they are skipped in `build_boundary_batch_metadata`), so a
`classifications` weight is a no-op on the span loss and would need the classification head
separately. And the weight must be applied **per query, not per sample** — one sample
carries queries of several task types.

## 4. The experiment

**Corpus: `mix_natural`** (84,280 records; 7.6% events, plus entities, relations,
classifications, structures). It is the only multi-task corpus here where cross-task
interference has actually been observed — the warm start pays relation −0.016 at 30% replay
while events gain +0.021. **RAMS is the wrong corpus**: 100% events, so it is structurally
incapable of showing a cross-task trade.

**Base: `whr778/gliner2-joint-boundary-mmbert-137k`, fixed across every arm.** Clone
`warmstart-natural.yaml` and change *only* `task_loss_weights`. Same `task_lr` 5e-4, same
`encoder_lr` 2e-5, same 3 epochs, same threshold sweep.

### Arms — a dose-response, not a two-arm win

| arm | event weight | purpose |
|---|--:|---|
| A | 1.0 | control; must be bit-identical to today's code |
| A' | 1.0, second seed | **noise floor** |
| B | 0.5 | down-dose |
| C | 2.0 | up-dose |
| D | 4.0 | up-dose, far |

**Why a sweep and not A-vs-B.** The hypothesis is a *capacity trade*, and a single
two-arm win at one seed is exactly the shape this repo has been burned by — item 12's
+0.0119 is provisional for that reason. Four dose points give a trend, and they give the
sharp test below, which no single comparison can.

### The primary readout

Not "does C beat A". It is the **sign of the correlation between event and non-event
deltas across the dose axis**:

- **Trade confirmed** — event metrics rise monotonically with the weight while
  relation/entity fall. Then there is a knee, and the deliverable is where it sits.
- **Not a trade, a free win** — event metrics rise and non-event metrics do not fall
  beyond the A/A' noise floor. This is what the external evidence predicts, and it is the
  most valuable outcome.
- **No lever** — nothing moves beyond noise. Loss balance is not the mechanism here, and
  the negatives are attributable elsewhere (§5). A clean negative, and cheap.

Secondary: the per-task diagnostic buckets from step 4 should *show* the reweighting —
`event_pair_loss` share of total rising with the weight. If it does not, the plumbing is
wrong regardless of what the metrics say.

## 5. Separating this from the two explanations already on record

Both are held constant by construction, so neither can explain a *between-arm* difference:

- **Head-init** — the stated cause of the CASIE Tier 2 collapse (0.0036 vs 0.2998) and the
  MAVEN flat result. Every arm warm-starts from the same checkpoint, so head-init is
  identical across arms.
- **`task_lr` tuned for cold heads** — TODO item 5's explanation for the warm-start
  relation regression (−0.037). Held fixed at 5e-4 across arms.

This design shows whether loss weight is **a** lever. It does not show it is the **best**
lever; a `task_lr` × `task_weight` 2×2 would, and is a follow-up, not run 1.

## 6. Pass/fail, fixed before spending

1. **Inertness gate (free, local, before any GPU).** With all weights 1.0, total loss must
   reproduce the pre-change value to <1e-6 on a fixed batch, and every mechanism bucket
   must match. If this fails, stop — every arm difference would be plumbing.
2. **Noise floor from A vs A'.** Any |Δ| below it is unreadable. Do not adopt a prior
   guess; measure it. Absent a reason to think otherwise, expect it to be near the 0.01
   that item 12's cautionary result sits at.
3. **Every arm at its own swept threshold** before any comparison is read — the project's
   standing rule, promoted from lesson because it changed a conclusion three times.
   `eval.threshold_sweep: true` is already set in the warmstart configs.
4. Report per-capability, not a single aggregate. The whole question is which capability
   moves which way.

## 7. Cost and sequencing

Five arms × ~1.7 h training + per-epoch eval and blind test. On `gpu_2x_h100_sxm5`
($8.38/h) that is three waves of two, roughly **8 h wall clock, ~$65**. On `gpu_1x_h100`
($4.29/h) sequential, ~2× the wall clock at similar total cost.

**Sequencing:** do not start until the GIST A/B finishes and its checkpoints are pulled —
same corpus, same base, and the box is currently occupied.

**One economy, with a condition.** The GIST *control* arm now running is
`warmstart-natural.yaml` with implicit weights of 1.0, i.e. arm A. It can serve as a
control replicate **only if** gate 1 proves the change inert, since otherwise it was
trained by a different codebase. If gate 1 passes, this drops to four new arms (~$52).

## 8. Traps

- **Pull `best/` off the box before terminating.** Item 12 is an entire experiment left
  unreadable because its checkpoints went with the instance.
- `uv pip install -e .` must be `-e ".[local]"`, or transformers resolves to 5.13 against
  a core `kernels<0.13` pin, FA2 never hooks, and bf16 mmBERT goes non-finite around step
  50 after running 11× slow. Set `GLINER2_STRICT_ATTN=1` so the fallback raises.
- The Lambda API is behind Cloudflare and rejects `urllib` with a 403 that reads exactly
  like a bad key. Use `curl`.

## 9. If run 1 shows a lever

The faithful phase-2 analogue is not a flat multiplier but a per-task **`pos_weight`**
inside the span BCE (`_safe_bce`, `losses.py:32`) — phase 2's lever was
`event_struct_pos_weight`, a positives-vs-negatives reweighting for event queries
specifically, not a scalar on the whole term. Run the flat weight first because it is one
variable and tests whether balance matters at all; escalate to per-task `pos_weight` only
if it does.

## 10. RESULT of run 1 — the flat weight is a null lever, and why

Four arms, all swept to threshold 0.3 on val, best-vs-best. Dose on `task_loss_weights.events`:

| metric | 0.5 | 1.0 (A) | 2.0 | 4.0 | A-vs-A' floor |
|---|--:|--:|--:|--:|--:|
| entity fair | 0.6156 | 0.6248 | 0.6209 | 0.6242 | 0.0097 |
| relation strict | 0.1793 | 0.1439 | 0.1180 | 0.1561 | **0.0407** |
| event_trigger fair | 0.7414 | 0.7527 | 0.7552 | 0.7541 | 0.0128 |
| event_type strict | 0.9686 | 0.9531 | 0.9399 | 0.9337 | 0.0139 |
| event strict | 0.3342 | 0.3433 | 0.3585 | 0.3553 | 0.0079 |

By §4's readout this is **"no lever"**: nothing is monotone in the dose, no trade appears
(the two event metrics disagree with each other), and the only movement clearing its floor
is `event_type` getting *worse* as event weight rises.

**The A-vs-A' noise floor is the reusable output.** Relation strict has a ±0.041
run-to-run band on this corpus, which is 4× the 0.01 §6 guessed and larger than every
relation delta in the table. It also voids the GIST A/B's relation claim (−0.033).

### Why it was null — measured, not inferred

`tools/train/probe_task_losses.py` on the control (seed43/final, 100 batches, train path,
buckets reconciling to 3.4e-07 with a residual of exactly 0):

> **Corrected.** A first pass bucketed only start/end/pair/inside and reported
> "events 1.6%, entities 17.2%, task-blind 76.4%". Those were shares of the four
> *bucketed* terms, and the 76.4% called task-blind is task-attributable — it was
> simply unmeasured. The numbers below bucket all nine query-typed terms.

| task | share of total training gradient |
|---|--:|
| **entities** | **77.2%** |
| json_structures | 11.5% |
| **events** | **6.6%** |
| relations | 2.5% |
| not query-typed (classification, consistency) | 2.2% |

Entities at 77.2% are the imbalance, not events at 6.6%. **`query_weights` reaches only
start/end/pair** — three call sites, 18.5% of the loss — so the strongest arm (w=4.0)
moved events from 6.6% to 10.6% of the gradient while three quarters of it sat untouched.
The null was close to mechanically guaranteed; it is not evidence that loss balance is the
wrong idea.

### Doses for run 2, computed rather than guessed

Event mass splits pos 0.01368 / neg 0.01067 (positive fraction **0.56** — hard-negative
mining has already trimmed the negatives, so the "negatives dominate" prior was wrong).
`pos_weight` k multiplies the event term by `(k*pos + neg)/(pos + neg)`:

Every candidate treatment on one comparable axis — events' share of the **whole**
gradient (control 6.6%):

| treatment | events' share | verdict |
|---|--:|---|
| flat w=2 | 8.0% | **tested, null** |
| pos_weight k=4 | 9.1% | *below* a dose already null — dropped |
| flat w=4 | 10.6% | **tested, null** |
| pos_weight k=8 | 12.2% | first dose past the tested range |
| flat w=2, `scope: all` | 12.5% | pairs with k=8 at matched share, different mechanism |
| pos_weight k=16 | 17.8% | real test of the direction hypothesis |
| flat w=4, `scope: all` | 22.2% | the magnitude question, reach fixed |

Arms: `warmstart-natural-evpw{08,16,32}.yaml` and `warmstart-natural-evwide{2,4}.yaml`.
Liveness verified end-to-end through `_build_model` — `events_pair_loss` ×5.57 and
`events_start_loss` ×5.59 at k=8, with entity/relation/json_structure and every
unreached term at ×1.000 exactly.

**Regime matters more than the dose.** The event positive fraction is 0.562 at
convergence (balance at k≈0.8) but **0.052 at cold-start initialization** (balance at
k≈18). `pos_weight` corrects a negative-dominated imbalance; at warm start that
imbalance has already been resolved by training, so k>1 does not correct it so much as
create the opposite one. The mechanism belongs in the cold-start rebuild.

### The larger finding: the weight reaches 18% of the loss, and could reach 94%

Every term, with its configured weight applied, as a share of the 1.51291 mean total:

| term | weighted | share | reached by `query_weights` today? | shape |
|---|--:|--:|---|---|
| pair | 0.22978 | 15.2% | **yes** | elementwise `[B,Q,C]` |
| start | 0.02501 | 1.7% | **yes** | elementwise `[B,Q,L+1]` |
| end | 0.02474 | 1.6% | **yes** | elementwise `[B,Q,L+1]` |
| proposal | 0.52090 | 34.4% | no | listwise → `[B,Q]` |
| rerank | 0.33884 | 22.4% | no | listwise → `[B,Q]` |
| soft-IoU | 0.17428 | 11.5% | no | elementwise `[B,Q,C]` |
| count | 0.08429 | 5.6% | no | per-query scalar `[B,Q]` |
| inside | 0.02317 | 1.5% | no | elementwise `[B,Q,L]` |
| abstention | 0.00449 | 0.3% | no | per-query scalar `[B,Q]` |
| consistency | 0.00061 | 0.0% | no | not query-typed |
| record / classification / relation | 0.08680 | 5.7% | no | separate heads |

Reachable today **0.27953 = 18.5%**; reachable if extended to the other six query-typed
terms **1.42550 = 94.2%**. That is the 5.1× figure.

**It is not the hard change the shape column suggests.** `proposal_listwise_loss`,
`reranker_listwise_loss`, `abstention_loss` and `count_log_rate_loss` all build a `[B,Q]`
per-query loss and then reduce it as `(loss * keep).sum() / keep.sum()` — structurally
identical to `_reduce`'s `per_query` branch. A per-query weight is exactly as well-defined
there as it is for start/end/pair, and the unweighted-denominator decision already argued
for `_reduce` carries over unchanged. `soft_iou` is a `candidate_pair_loss` call that
already *accepts* `query_weights` and simply is not passed it.

Still a **different experiment** from run 2, and one that needs a measurement first:
events' share of those six terms is currently an *estimate* (~8%, from events holding 8.0%
of bucketed mass and 7.6% of queries), not a number the buckets produce. Bucketing
soft-IoU costs one argument; proposal/rerank need their `[B,Q]` loss surfaced. Measure,
then decide — the same order that turned run 1's null into a diagnosis.
