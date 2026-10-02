# The life of a sample: eb18 (and eb19 candidates), traced

**Status:** working paper, 2026-10-02. **eb18 is a PROVISIONAL base** (its LR ran out at epoch 5 with
the selection metric still rising); numbers here describe the mechanism, not a final model.

## Purpose

This follows ONE real training record from its JSONL line to the optimizer step. At each stage it
gives the real values the pipeline held, and the loss equation exactly as the code implements it.
The documented flow and the actual flow drift apart. This paper is the place to compare them, and
four corrections below came from doing exactly that.

Evidence grades used throughout:
- **one record** / **one batch**: a single traced example. It shows the mechanism, not a rate.
- **measured**: a sample-based rate with its n given.

## Reproduce

```
uv run python tools/train/trace_sample_life.py --corpus casie            # S0-S7 on one real record
uv run python tools/train/measure_gradient_shares.py --batches 8 [--absent-reduction separate]
uv run python tools/train/measure_absent_dilution.py                     # pi_P over the real mix
uv run python tools/data/adjudicate_absent_events.py --per-corpus 50     # is "absent" true?
```

`trace_sample_life.py` has two self-checks:
- **S6 gate:** the weighted sum of the logged terms must reproduce the model's own loss.
- **S7 gate:** the gradient shares must sum to 1.

---

## 1. Trace diagram (eb18-balanced, one CASIE record)

Real values come from `trace_sample_life.py`, seed 0, eb18 checkpoint, CPU, train mode.

```
S0  RAW JSONL  data/casie.train.jsonl
    575 chars: "The boss of a company held to ransom by computer hackers ..."
    gold: entities Person x5, Organization x2, Capabilities, PaymentMethod, Money
          events   Cyber.Ransom x3  (triggers 'held to ransom' | 'demanded money' | 'demanded')
        |
S1  LABEL TRANSFORM   labels_file: labels/unified-full.yaml        (no change for this record)
        |
S2  SLIDING WINDOW    max_len 4096 subwords, stride 3072  ->  1 chunk (114 subword tokens)
        |
S3  ExtractorDataset.__getitem__   (runs in a worker; re-samples on EVERY call)
    - schema sampling: remove_events_prob 0.2 may drop a gold type (query AND gold together)
    - NegativeLabels.inject: +1 absent entity label, +1 absent event type from the same corpus
      here: + Device (entity, empty gold), + Cyber.Databreach (event, empty gold)
        |
S4  COLLATOR    107 words, 218 subword ids, 24 queries
    - an EVENT TYPE expands to a trigger query + one query per role:
      Cyber.Databreach -> 12 queries (trigger + 11 roles), all ABSENT (no gold)
      Cyber.Ransom     -> 6 queries (trigger gold 4, roles gold 1-5)
    - 13 of 24 queries carry no gold (Device + all 12 Databreach)
    - mention_mask [B,Q,G] = gold spans per query; record_specs: natural groups
      (event_records: true -> anchor = trigger field)
        |
S5  FORWARD (train mode)
    encoder mmBERT-base, FA2, bf16 on GPU
      -> boundary marginals: start/end logits [B,Q,L+1], inside logits [B,Q,T]
      -> proposal: top-128 starts x conditional ends, per query
         + GOLD INJECTION with prob p_inj(t): 1.0 for the first 15% of planned optimizer
           steps, then linear to 0.25 (trainer defaults; eb18 overrides none)
         -> up to 768 candidates per query (training_candidate_budget)
      -> pair scorer (compat + start marginal + end marginal), null head, count head
      -> record head, natural mode: instances = anchor (trigger) candidates;
         assign logits per field = [ABSENT column | candidates of that field's query]
      -> relation scorer, classifier
        |
S6  LOSSES (section 2)      model loss 0.807443
    = weighted span terms 0.801163 + record_field 0.006280 + residual -1.49e-08   GATE PASSES
        |
S7  GRADIENT   |g| = 75.96 on this record; max_grad_norm 1.0 -> clip factor 0.0132
    trainer: loss/4 per micro-batch, 4 micro-batches summed, clip_grad_norm_(1.0),
    AdamW (encoder 2e-5, task 5e-4, weight decay 0.01), cosine_restarts, warmup 0.05
```

### S6 loss table (one record; head schedules at their step-0 defaults, see Finding 3)

| term | value | weight | contribution | gradient norm on encoder | share of the whole-model gradient |
|---|---:|---:|---:|---:|---:|
| start_loss | 0.02474 | 1.0 | 0.02474 | 0.915 | 0.0071 |
| end_loss | 0.02045 | 1.0 | 0.02045 | 0.905 | 0.0087 |
| pair_loss | 0.19418 | 1.0 | 0.19418 | 11.65 | 0.1160 |
| inside_loss | 0.05584 | 0.5 | 0.02792 | 1.46 | 0.0121 |
| soft_iou_loss | 0.40840 | 0.2 | 0.08168 | 4.41 | 0.0292 |
| rerank_listwise_loss | 0.52928 | 0.3 | 0.15878 | 24.29 | 0.2920 |
| proposal_loss | 0.98217 | 0.3 | 0.29465 | 33.06 | 0.3947 |
| consistency_loss | 0.00148 | 0.1 | 0.00015 | 0.017 | 0.0001 |
| abstention_loss | 0.03061 | 0.2 | 0.00612 | 8.70 | 0.0579 |
| count_loss | -0.03754 | 0.2 | -0.00751 | 10.96 | 0.0807 |
| record_field_loss | 0.00628 | 1.0 | 0.00628 | 0.361 | 0.0015 |
| record_object_loss | 0.0 | 1.0 | 0 | none (constant) | 0 |
| classification_loss | 0.0 | 1.0 | 0 | none (no classification queries here) | 0 |

Share is defined as share_k = ⟨g_k, g⟩ / |g|², where g_k = ∇(w_k L_k).
- The shares sum to 1 **by linearity, for any weights**. No normalisation of the weights is needed.
- A share is a projection, not a cosine. It can be negative, or greater than 1.

---

## 2. Every loss term, as implemented

**Notation**
- b = sample, q = query, ℓ = boundary position (L+1 of them), t = token, c = candidate.
- BCE(x, y) = softplus(x) − x·y.
- **global reduction** = Σ(elementwise·keep) / max(Σ keep, 1), pooled over the whole micro-batch (`losses.py:56-100`).
  - In global mode, `_reduce` **ignores `query_mask`**.
  - With `query_weights`, the numerator is weighted and the denominator is not.
  - eb18 sets no `query_weights` (`task_loss_weights` is None), so w_q = 1.

**Shared masks** (`model.py:782`, `:854`)
- keep[b,q,ℓ] = boundary_mask[b,ℓ] ∧ query_mask[b,q]
- absent[b,q,ℓ] = keep ∧ ¬ any_g mention_mask[b,q,g]  (the query has no gold mention)

### 2.1 start / end boundary BCE (`model.py:856-873`, `losses.py:189-212`)

e[b,q,ℓ] = BCE(s[b,q,ℓ], y[b,q,ℓ]) · (1 if y = 1 else `boundary_negative_weight` = 1.0)

**pooled** (eb18):

$$L_{start} = \frac{\sum_{keep} e}{|keep|} = \pi_P\,\bar L_P + \pi_A\,\bar L_A,\qquad \pi_P = \frac{N_P}{N_P+N_A}$$

**separate** (opt-in, `df162a0`):

$$L_{start} = \texttt{present\_loss\_scale}\cdot\bar L_P + \texttt{absent\_loss\_weight}\cdot\bar L_A$$

- Weight in the total: 1.0, hard-coded in `DEFAULT_LOSS_WEIGHTS` (`model.py:105`). It is not settable from YAML.
- The end loss is the same with end targets.
- **Measured** π_P on eb18's mix (6,765 records, 20,000 simulated micro-batches of 4):
  - negatives on: mean 0.713, p10 0.493, p90 0.887;
  - negatives off: mean 0.862.

### 2.2 inside (`model.py:1019-1026`, `losses.py:470-493`)

- Balanced BCE over inside logits [b,q,t].
- keep = text_mask ∧ query_mask; global mean.
- Weight 0.5, hard-coded.

### 2.3 candidate labels and hard negatives (`losses.py:251-316`, `:402-435`)

- **Label:** z[b,q,c] = 1 iff the candidate (s,e) exactly equals a gold (s,e) of that query, i.e. by **span identity**. The candidates include injected gold.
- **Hard negatives:** per query, keep all positives plus the top-k negatives by detached pair logit, with k = max(5·n_pos, 8).
  - An absent query keeps 8 negatives (`hard_negative_keep_all_when_absent=False`).
- The guide veto is inert in eb18 (no `guide_scores`).

### 2.4 pair (`model.py:934-964`, `losses.py:438-467`)

$$L_{pair} = \frac{\sum_{eff} BCE(p_{bqc}, z_{bqc})}{|eff|},\qquad eff = valid \wedge (z \vee hard)$$

- Weight 1.0, hard-coded.
- **`negative_query_ratio` (0.5) is INERT under global reduction.** The sampled query mask reaches `_reduce` only as `query_mask`, which global mode ignores.
  - Every absent query contributes its 8 hard negatives, every step.
  - A subagent probe confirmed it: pair loss 0.700418 with the full mask and with the sampled mask. It differs under `per_query` and `sum`.

### 2.5 soft-IoU (`model.py:967-982`)

$$L_{iou} = \frac{\sum_{eff} BCE(p_{bqc}, \max_g IoU(c, g))}{|eff|}$$

- Same `eff` set and the same logits as pair.
- Weight 0.2·σ_iou(t), with **σ_iou(t) = max(1 − t/20000, 0)** (`trainer.py:1417-1421`). It is off after 20,000 optimizer steps.
- **Double counting:** an exact positive gets target 1 in both pair and soft-IoU.
- **Conflict:** an overlapping hard negative gets target 0 from pair and a fractional IoU from soft-IoU, on the same logit.

### 2.6 rerank listwise (`losses.py:715-752` → `proposal_listwise_loss`)

$$\ell_{bq} = \operatorname{LSE}_{c\in valid} p_{bqc} - \operatorname{LSE}_{c\in G_{bq}} p_{bqc},\quad G = \{c: z>0.5\},\qquad L_{rerank} = \frac{\sum_{q:G\neq\emptyset}\ell_{bq}}{\#\{q: G\neq\emptyset\}}$$

- Weight 0.3.
- A query with no gold contributes exactly 0.
- `absent_negatives_in_denominator` is False in eb18. It was measured only at `start_top_k` 16 (absneg2/4); see the catalog note.

### 2.7 proposal listwise: the injected-only gold defect (`model.py:1028-1065`)

The same listwise form as 2.6, but on proposer logits (compat + start marginal + end marginal), with

$$G^{prop}_{bq} = \{c : \texttt{is\_gold}_c\}\quad\text{where is\_gold marks only the INJECTED copy}$$

`assemble_candidates` (`proposal.py:343-450`) works like this:
1. Gold spans are injected with a ceiling score.
2. After de-duplication, only the injected copy keeps `is_gold`.
3. A gold span the proposer found **naturally**, but that this step's injection did not sample, survives as an ordinary candidate. It sits in the **denominator as a negative**, while the pair and rerank losses label the same span **positive**. Because the proposal logit includes the start/end marginals, this also pushes down the boundary scores of real gold.

**Probe** (one real CASIE batch, eb18, 26 gold-bearing candidates by span identity):

| p_inj | proposal-loss gold | identity-gold labelled NEGATIVE in proposal loss | proposal loss |
|---:|---:|---:|---:|
| 1.0 | 26 | 0 | 1.6172 |
| 0.25 | 4 | **22** | 2.0605 |
| 0.0 | 0 | 26 | **0.0000** (no query has gold, so the term switches itself off) |

eb18 ran with p_inj below 1.0 for the last 85% of its planned steps, ending at 0.25. So the
contradiction was active in the term with the largest gradient share (one record: 0.39 of the
whole-model gradient).

**Fix (built, opt-in): `boundary_head.proposal_gold: identity`.** It labels proposal-loss gold by span identity, the same z as pair and rerank.
- The default `injected` is bit-identical to eb18.
- A log line from inside the loss reports the mode that executed and the cumulative count of gold candidates under each definition.

Traced after the change on a real eb18 CASIE batch with 16 gold candidates (the schema re-sampled, so this batch differs from the one in the table above):

| mode | p_inj | proposal-loss gold | identity-gold labelled NEGATIVE | proposal loss |
|---|---:|---:|---:|---:|
| injected | 1.0 | 16 | 0 | 2.322924 |
| injected | 0.25 | 2 | **14** | 4.255371 |
| injected | 0.0 | 0 | 16 | 0.000000 |
| identity | 1.0 | 16 | 0 | 2.322924 |
| identity | 0.25 | 16 | 0 | 2.322924 |
| identity | 0.0 | 16 | 0 | 2.322924 |

The two modes agree to 8 decimals at p_inj 1.0. Under `identity` the proposal loss no longer depends on p_inj for gold that was proposed naturally. Tests: `tests/models/boundary/test_proposal_gold.py`, mutation-checked.

### 2.8 consistency (`losses.py:755-808`)

$$\hat P^{start}_{bq\ell} = 1-\prod_{c\in valid,\ s_c=\ell}(1-\sigma(p_{bqc})),\qquad L_{cons} = \tfrac12\left[\frac{\sum_{k_s}(\hat P^{start}-\sigma(s))^2}{|k_s|} + \text{end}\right]$$

- k = (some candidate reaches ℓ) ∧ keep.
- The target σ(marginal) is **not detached**, so the loss pulls on both the pair logits and the marginal logits.
- Weight 0.1·min(t/2000, 1), a warmup from 0 (`trainer.py:1450-1460`).

### 2.9 abstention and count (`losses.py:811-859`)

- **Abstention:**
  $$L_{null} = \frac{\sum_q BCE(n_{bq}, \mathbb 1[q\text{ has no gold}])}{\#\text{valid } q}$$
  Weight 0.2.
  - At decode it gates entity and extractive queries only. `_decode_records` (events) never reads it.
- **Count:**
  $$L_{count} = \text{mean}_q\,(e^{r_{bq}} - y_{bq}\, r_{bq})$$
  Poisson NLL with `full=False`, so the value **can be negative** (−0.0375 on the traced record). Weight 0.2.

### 2.10 record loss, natural mode = events (`records.py:1432-1521`, `model.py:2436-2547`)

**Object loss:** identically 0 (`records.py:1468-1473`).

**Field loss, per gold record r:**
- First, the gold trigger is resolved against the model's own anchor candidates.
  - If it is neither proposed nor injected, **the record trains nothing**.
  - Since p_inj ≤ 1, this happens more often as injection anneals.
- Otherwise:
  - **Trigger (scalar):** ℓ = −log Σ_{c∈T} softmax(A_{i,f})_c, where A = [ABSENT | candidates]. If no column matches the gold, the target is the ABSENT column.
  - **Role (list):** ℓ = mean over the field's candidates (up to 768) of BCE(A_{i,f}[1+c], 1[c matches a gold value]).
    - The ABSENT column is excluded.
    - A gold filler that was not proposed is silently dropped.
  - ℓ_r = (1/F)·Σ_f ℓ_{r,f}.

**Batch aggregation:**

$$L_{field} = \frac{\sum_g\sum_{r}\sum_f \ell_{r,f}}{\sum_g n_g F_g}$$

A uniform mean over (record, field) cells, so groups with more roles weigh more.

**Normalisation mismatch:** a scalar cell is one log-prob, while a list cell is a mean BCE over up to 768 mostly easy negatives.

**Total:** `record_loss_weight` (1.0) · (L_obj + L_field).

### 2.11 relation and classification

- **Relation:**
  $$L_{rel} = \frac{\sum_{mask} BCE(x_k, y_k)}{|mask|}$$
  y_k is exact edge identity. Weight 1.0 (`model.py:1993-2075`).
- **Classification:** BCE over choices, divided by the label count, weight 1.0 (`model.py:1918-1969`).

### 2.12 silent guards

- `_finite_loss_term` (`nan_to_num`) zeroes **each** of the 10 head terms on its own, plus span_total, cls, every record value and relation. There is no log at that level.
- The trainer's non-finite counter sees only the combined scalar.
- The checks are forward-only: bf16 has no gradient-NaN guard.
- Record groups that raise TargetCapacityError, ValueError or IndexError are skipped without a log (`model.py:2533`).

---

## 3. The compound equation (eb18, one micro-batch, optimizer step t)

$$
\begin{aligned}
L(t) =\;& L_{start} + L_{end} + L_{pair} + 0.5\,L_{inside}
 + 0.2\max(1-\tfrac{t}{20000},0)\,L_{iou} + 0.3\,L_{rerank} + 0.3\,L_{prop}\!\left[p_{inj}(t)\right]\\
&+ 0.1\min(\tfrac{t}{2000},1)\,L_{cons} + 0.2\,L_{null} + 0.2\,L_{count}
 + L_{cls} + 1.0\,(0 + L_{field}[p_{inj}(t)]) + 1.0\,L_{rel}
\end{aligned}
$$

- Every term passes through `nan_to_num`.
- p_inj(t) = 1.0 for the first 15% of planned steps, then linear to 0.25. It enters through:
  - the candidate positives (pair, rerank, soft-IoU),
  - the proposal gold mask,
  - the record anchor gate.

**The update:**

$$\theta \leftarrow \text{AdamW}\Big(\theta,\ \operatorname{clip}_{1.0}\Big(\sum_{m=1}^{4}\nabla_\theta \tfrac{L_m(t)}{4}\Big)\Big)$$

with encoder LR 2e-5, task LR 5e-4, and cosine_restarts.

**Reading gradient shares under AdamW.** AdamW divides each coordinate by its own running RMS.
- A module touched by one term alone steps at full size, whatever its global share. The record decoder gets gradient only from `record_field`.
- Shares are therefore meaningful **within a parameter group**, where the terms compete for the same coordinates.
- Global-norm clipping and accumulation do not change shares: clipping scales uniformly, and accumulation is linear.

**Per-group shares, one training-shaped batch** (synthetic_events_capped, sentence_rex, docee_zh, docee; step-0 head schedules; each group sums to 1.0000; residual 0.0):

| group | rerank | pair | proposal | count | abstention | soft_iou | inside | start | end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boundary_head | 0.333 | 0.319 | 0.257 | 0.005 | 0.0001 | 0.083 | 0.001 | 0.0003 | 0.0003 |
| encoder | 0.298 | 0.178 | 0.195 | 0.212 | 0.041 | 0.039 | 0.016 | 0.0045 | 0.0040 |

### Gradient shares over 8 batches (`pooled`, eb18's loss)

**Setup:** `measure_gradient_shares.py --batches 8`. Each batch is 4 chunks from the natural mix. The run uses step-0 head schedules (p_inj 1.0, soft-IoU 1.0, consistency 1.0). Residual is 0.0 on every batch, and every group sums to 1.0000.

Shares are mean (min, max). Single-term groups (classifier, record_decoder, relations) are 1.0 by construction.

| term | boundary_head | encoder |
|---|---:|---:|
| rerank_listwise | 0.297 (0.126, 0.577) | 0.314 (0.081, 0.430) |
| proposal | 0.260 (0.075, 0.506) | 0.248 (0.093, 0.345) |
| pair | 0.263 (-0.136, 0.605) | 0.147 (0.107, 0.178) |
| soft_iou | 0.137 (-0.063, 0.616) | 0.045 (0.027, 0.092) |
| count | 0.038 | 0.109 (0.015, 0.294) |
| classification | 0 | 0.113 (0.000, 0.480) |
| abstention | 0.000 | 0.009 |
| relation | 0 | 0.008 |
| record_field | 0.007 | 0.002 |
| inside | 0.0007 | 0.005 |
| **start** | **0.0005** | **0.0023** |
| **end** | **0.0004** | **0.0020** |
| consistency | -0.0002 | -0.0001 |

**Reading it:**
- Three listwise and pairwise candidate terms (rerank, proposal, pair) set about 0.82 of the boundary head's direction and 0.71 of the encoder's.
- The start/end boundary terms set under 0.1% of the boundary head and about 0.4% of the encoder. That is the budget fix #2 (`absent_reduction`) acts on.
- Pair and soft-IoU go **negative** on some batches. On those batches they opposed the rest of the update in the boundary head, which fits the pair/soft-IoU target conflict in 2.5.

**`absent_reduction: separate`** (same 8 batches, residual 0.0 on all) raises the boundary terms by about 1.5×:

| group | term | pooled | separate |
|---|---|---:|---:|
| boundary_head | start | 0.0005 | 0.0008 |
| boundary_head | end | 0.0004 | 0.0006 |
| encoder | start | 0.0023 | 0.0034 |
| encoder | end | 0.0020 | 0.0031 |

Every other term is unchanged to the third decimal. Fix #2 does what its equation says, on a budget too small to be likely to move a score: under 0.4% of the encoder's direction.

### Late-training state: the contradicted term dominates

**Setup:** p_inj 0.25, soft-IoU 0, consistency 1. This is where eb18 spent most of its steps. Both modes ran on the same 8 seeded batches, with residual 0.0 on all of them. Values are mean (min, max).

| term | boundary_head, injected (eb18) | boundary_head, identity (fix) | encoder, injected (eb18) | encoder, identity (fix) |
|---|---:|---:|---:|---:|
| proposal | **0.590** (0.372, 0.828) | 0.304 (0.121, 0.440) | **0.491** (0.272, 0.688) | 0.304 (0.207, 0.402) |
| rerank_listwise | 0.223 | 0.367 | 0.243 | 0.350 |
| pair | 0.172 | 0.291 | 0.105 | 0.131 |
| count | 0.013 | 0.034 | 0.087 | 0.108 |
| record_field | 0.0008 | 0.0056 | 0.0006 | 0.0014 |
| start / end | 0.0003 / 0.0003 | 0.0007 / 0.0007 | 0.0018 / 0.0017 | 0.0023 / 0.0022 |

**What this shows:**
- In eb18's late state, the proposal loss set **59% of the boundary head's update direction and 49% of the encoder's**. At p_inj 0.25 that term labels most naturally found gold as negatives (14 of 16 in the 2.7 trace).
- Early (p_inj 1.0) it was about a quarter of the update. The contradiction grows into the dominant term exactly when injection anneals.
- Under `proposal_gold: identity` the proposal share returns to 0.30, close to its early level, and rerank and pair regain theirs.

This is mechanism, not outcome: the Phase 2 fast A/B (catalog, `p2fast`) measures whether it moves scores.

---

## 4. eb19 candidates, as deltas to section 3

Each candidate is opt-in and none is adopted yet.
1. **`absent_reduction: separate`** (built, `df162a0`)
   - Replaces π_P·L̄_P + π_A·L̄_A in L_start/L_end with s·L̄_P + λ_A·L̄_A.
   - Touches start/end only. These carried 0.0003 (boundary_head) and about 0.004 (encoder) of their groups on one batch, so the expected effect is small.
2. **Proposal gold by span identity** (built, `proposal_gold: identity`)
   - In L_prop, G^prop_bq = {c : z_bqc > 0.5}.
   - This removes the dependence of L_prop on p_inj(t) for gold that was proposed naturally.
3. **Per-corpus soft target for absent queries** (idea)
   - In L̄_A and the absent rows of L_pair, set y = ε_c instead of 0.
   - ε_c would be ranked from the adjudication below.

---

## 5. Findings from tracing

1. **Gold reaches 100% of training candidates only for the first 15% of steps.**
   - p_inj then anneals to 0.25.
   - An earlier statement that "eb18 already reaches 100% of gold triggers in training" was true only for that window. It is corrected here.
2. **`negative_query_ratio` sampling is inert under the global reduction.** All absent queries contribute to every term that sees them.
3. **The trace's S6 weights are step-0 values.**
   - In training, soft-IoU decays to 0 by 20,000 steps and consistency warms up over 2,000.
   - A warm start resets t, so it re-enables soft-IoU and restarts injection at 1.0.
4. **The proposal loss trains against naturally found gold whenever p_inj < 1** (2.7, probe table).
5. **Absent-event negatives that really occur** (Haiku 4.5 judge, 50 random injections per corpus):
   - wikievents 34%, maven 28%, cmnee 20%, cc_news_haiku45 14%, synthetic 12%, cc_news_events 10%, casie 6%, rams 6%, duee 0%.
   - The judge over-calls fine-grained taxonomies, so these are upper bounds.
   - The lexicon detector's recall against the judge is mostly 0.00-0.33.
   - Catalog row: 2026-10-02 "Absent-EVENT negatives adjudicated".
6. **Dilution of the boundary loss:** π_P 0.862 → 0.713 with negatives on. The worst corpora are rams_merged 0.372 and cc_news_events 0.420. Catalog row: 2026-10-02 "Absent queries dilute the boundary loss".
7. **Gradient magnitude is not loss magnitude.**
   - On one record, proposal and rerank supplied 0.69 of the gradient direction from 56% of the loss (0.453 of 0.807).
   - start + end supplied 0.016 of the direction from 5.6% of the loss.

## 6. Open questions to check

- **Injection schedule × record anchor gate.**
  - From 15% of training onward, a gold trigger that is neither proposed nor injected trains no argument record at all.
  - How many event records went untrained per epoch in eb18? This is measurable with `_note_anchor_gate`.
- **Soft-IoU conflict:** the same logit gets target 0 from pair and a fractional IoU from soft-IoU on overlapping hard negatives, until step 20,000.
- **Consistency's undetached target:** is pulling the marginals toward the noisy-OR of the pair scores intended?
- **`nan_to_num` silence:** a term that went non-finite and was zeroed for many steps would leave no trace. Add a per-term counter?
- **Hard-coded weights:** start/end/pair/inside (1, 1, 1, 0.5) cannot be set from a config, so any A/B on them needs code.
- **Record field normalisation:** a scalar NLL cell is averaged with a list cell that is a mean over up to 768 BCEs, so role cells are diluted relative to trigger cells.
- **Abstention unused for events at decode,** while it separates absent from present near-perfectly on one record (P(null) 1.000 vs 0.001).
