# Per-task event loss weighting on the boundary architecture — evaluation plan

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
