# Code Review & Failure Mode Evaluation: Event Head, Coref Links, and Link Junctions

**Date**: 2026-10-22  
**Target Investigation**: Event head collapse, trigger-to-trigger links (`TriggerLink`), and trigger-to-argument links (`LinkJunction`).  
**Entrypoint**: [tools/train/train.py](file:///Users/williamroe/development/git/GLiNER2/tools/train/train.py)  

---

## 1. Executive Summary

Empirical autopsies (notably in [tools/events_working_papers/EB22C_SCORE_LEVEL_AUTOPSY.md](file:///Users/williamroe/development/git/GLiNER2/tools/events_working_papers/EB22C_SCORE_LEVEL_AUTOPSY.md)) revealed that the apparent "collapse" in event evaluation metrics is primarily an **unanchored score-level drift**:
- Gold triggers remain ranked at or near the top of proposed candidates (gold reach remains 0.96–1.00).
- However, raw candidate scores drift downward across training by ~0.30–0.40 on the probability scale (or ~1.5 to 2.0 in logit space).
- Because downstream decoding gates on a hard-coded absolute anchor threshold ($p > 0.30$, equivalent to $\text{logit} > -0.85$), this modest downward shift causes the number of passing triggers to drop to near zero at inference, collapsing precision, recall, and downstream argument extraction.

Beyond this calibration drift, code analysis across the training entrypoint, model architecture, and loss functions identified **six distinct structural bugs and supervision imbalances** that actively destabilize trigger and argument representations.

---

## 0. Verification (2026-10-22, against the code at 5d9f419 and the measured runs)

| # | issue | verdict | evidence |
|---|---|---|---|
| 1 | head encoders on `encoder_lr` | **CONFIRMED** (impact unmeasured) | `boundary_head.boundary_encoder` (463,360) + `boundary_head.candidate_encoder` (197,376) = 660,736 head params match `"encoder" in name` and train at 2e-5 |
| 2 | junction self-link on field 0 | **PARTLY CONFIRMED** | field 0 IS supervised (`_instance_field_loss` loops every field) and `LinkJunction.forward` adds a link to it; the quoted decode code ("Field 0 is the anchor itself") does NOT exist and `decode_group` also loops every field -- "discarded at inference" is unverified |
| 3 | all-negative trigger coref loss | **REFUTED as stated** | the quoted `_coref_link_loss` is not our code; ours pairs DIFFERENT gold records (distinct events, correctly negative) + k hardest false triggers; measured: the trigger-link gradient is ~7% of the entity gradient and orthogonal to entity/event (|cos| <= 0.12, gradient-conflict run) |
| 4 | other gold triggers excluded from column negatives | **CONFIRMED, scoped, partly mitigated** (12-agent verification 2026-10-22) | exclusion is DOUBLE-enforced (`trained` set :1877-1878 AND the gold-span overlap filter :1753-1754/:1762) and reaches only SAME-TYPE sibling events (one record group per event type). Exercised in 94% of CASIE, 81% of WikiEvents, 32% of CMNEE argument terms; NEVER in MAVEN (0 argument gold) and 5.5% of DuEE. Mitigation is partial: B's row BCE puts target 0 on A's argument only when it is among B's 8 hardest wrong columns (hard_k=8); a column at logit 0.5 outranked by 8 wrong columns at 1.5 gets NO gradient though it clears the 0.3 decode threshold, and decode is per-row (roles are non-exclusive), so A and B BOTH emit it. A naive 'siblings as negatives' fix is WRONG where siblings share the filler: 34% of CMNEE and 52% of DuEE sibling terms. Reaches the trigger level only through the shared encoder (record-loss encoder gradient 24-51, measured) |
| 5a | arguments dropped when the trigger is not proposed | **CONFIRMED, already documented** | records.py "THE SUPERVISION GATE" / `_note_anchor_gate("anchor_not_proposed")`; gold injection is 1.0 for the first 15% of planned steps, then anneals to 0.25 |
| 5b | absent schemas give the anchor gate no negative | **PARTLY REFUTED** | true for the RECORD head (`record_object_loss` was 0 on all 725 logged steps), but the anchor gate decodes the PAIR score, and absent event queries ARE negatives in the pair/proposal/rerank span losses |
| 6 | unanchored proposer prior drives a downward drift | **PARTLY REFUTED** | listwise shift-invariance is true; but the level OSCILLATES both ways (CMNEE gold logit -4.33 at 20k, +1.56 at 30k), entity levels swing as much (prior a minor part of the entity moves), and separate event scorers / task_lr 4e-5 were NULL inside the seed floor |

Most consequential that this review did not have: the seed-to-seed floor is LARGE (two identical setups differ by 0.083 mean event_type F1 over 20k-30k), and no training or eval loss tracks the swings (best |Spearman r| 0.22) -- the readout at 0.3 flips well-ranked scores. Path fix: `gradient_conflict.py` is in `tools/train/`, not `tools/events_working_papers/`.

---

## 2. Identified Issues Requiring Investigation

### Issue 1: Parameter Grouping Bug in `_create_optimizer` (Head Encoders Starved of LR)

> **Verdict: CONFIRMED** (routing). 660,736 head parameters train at `encoder_lr` 2e-5. The impact ("representation lag ... gradient instability") is NOT measured; note the task_lr 4e-5 A/B never touched these modules.
**File**: [gliner2/training/trainer.py#L1989-L2002](file:///Users/williamroe/development/git/GLiNER2/gliner2/training/trainer.py#L1989-L2002)

#### The Code
```python
for name, param in self.model.named_parameters():
    if not param.requires_grad:
        continue
    if "encoder" in name:
        encoder_params.append(param)
    else:
        task_params.append(param)
```

#### Mechanism & Impact
In `GLiNER2Boundary`, the head contains:
- `boundary_head.boundary_encoder`: BiLSTM / Transformer / Linear encoder over token states.
- `boundary_head.candidate_encoder`: Candidate span projection / encoder.

Because the check uses substring matching (`if "encoder" in name:`), both `boundary_encoder` and `candidate_encoder` are routed to `encoder_params` rather than `task_params`. In typical configurations:
- `encoder_lr` (backbone, e.g., DeBERTa/ModernBERT): `2e-5`
- `task_lr` (head modules): `5e-4` (25× higher)

These randomly initialized head encoders are trained at the fine-tuning backbone rate (25× slower than the rest of the head). Because all trigger and argument candidate representations are produced by `candidate_encoder` over `boundary_encoder` states, the candidate representations evolve with severe inertia compared to the fast-moving scoring heads (`prior`, `inst_w`, `link`). This representation lag forces the scoring heads to overcompensate, creating gradient instability.

---

### Issue 2: `LinkJunction` Contaminates Anchor (Trigger) Field 0 with Artificial Self-Link Gradients

> **Verdict: PARTLY CONFIRMED.** Field 0 is supervised and the junction adds a link to it (records.py 314-326, 1260-1270). The quoted decode block does not exist in our code and `decode_group` loops every field including 0, so "discarded at inference" is unverified. Impact unmeasured.
**Files**:
- [gliner2/models/boundary/records.py#L862-L866](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/records.py#L862-L866)
- [gliner2/models/boundary/records.py#L510-L536](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/records.py#L510-L536)
- [gliner2/models/boundary/records.py#L1054-L1064](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/records.py#L1054-L1064)
- [gliner2/models/boundary/records.py#L1968-L1972](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/records.py#L1968-L1972)

#### The Code
```python
assign_logits = self._assign_logits(inst_states, fq, field_cand_states)
if self.link is not None and spec.mode == "natural" and inst_states.shape[0] > 0:
    link = self.link(inst_states, fq, field_cand_states, anchor_spans, field_spans)
    assign_logits = [a if l is None else torch.cat([a[:, :1], a[:, 1:] + l], dim=-1)
                     for a, l in zip(assign_logits, link)]
```

#### Mechanism & Impact
1. `field_cand_states[0]` corresponds to the anchor (trigger) field.
2. In `LinkJunction.forward`, the loop runs over `enumerate(zip(field_cand_states, field_spans))` starting at index 0. It computes cross-bilinear and distance-weighted logits between the trigger instances and candidate triggers.
3. The resulting `link[0]` is concatenated with `assign_logits[0]`.
4. In `_instance_field_loss`, field 0 is supervised as a scalar field (`_scalar_field_nll`) against the gold trigger target.
5. **Inference mismatch**: During decode ([records.py#L1054-L1064](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/records.py#L1054-L1064)), `assign_logits[0]` is completely discarded:
   ```python
   # Field 0 is the anchor itself -- copy from the record instance
   fields_out[spec.field_names[0]] = {
       "text": text[rec.anchor_span[0]:rec.anchor_span[1]],
       "span": [rec.anchor_span[0], rec.anchor_span[1]],
       ...
   }
   ```
6. **Result**: During training, backpropagation expends representational capacity training `LinkJunction` to predict trigger-to-trigger self-links for field 0 that are never used during inference, injecting spurious gradients into `inst_states` and `fq[0]`.

---

### Issue 3: Asymmetric All-Negative Bias in Trigger-to-Trigger Coref Link Loss

> **Verdict: REFUTED as stated.** The quoted function is not our code (ours: `_coref_link_loss(group, records, anchor_qid, span_indices, k)`, records.py 1802). Its negatives are different gold events and the k hardest false triggers, which should be negative. Measured: the trigger-link gradient is small (~7% of entity) and orthogonal to entity and event gradients.
**File**: [gliner2/models/boundary/records.py#L1807-L1842](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/records.py#L1807-L1842)

#### The Code
```python
def _coref_link_loss(self, logits: torch.Tensor, coref_map: list[int]) -> torch.Tensor:
    pos, neg = [], []
    for i in range(len(coref_map)):
        for j in range(i + 1, len(coref_map)):
            if coref_map[i] == coref_map[j]:
                pos.append(logits[i, j])
            else:
                neg.append(logits[i, j])
    if not pos and not neg:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    pairs = pos + neg
    target = torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device=logits.device)
    return F.binary_cross_entropy_with_logits(torch.stack(pairs), target)
```

#### Mechanism & Impact
- In standard event extraction datasets (e.g., CASIE, ACE05, DuEE), documents overwhelmingly feature single-mention events per sentence or per document.
- When an example has no coreferent event mentions (`len(pos) == 0`), `target` is entirely zeros: `[0.0] * len(neg)`.
- The binary cross-entropy loss unconditionally drives `logits[i, j]` negative.
- Because $\text{logits}_{i,j} = \frac{z_i \cdot z_j}{\sqrt{d}} + \text{dist}_{i,j}$ where $z = W(\text{inst\_states})$, this loss applies a continuous repulsive gradient on trigger instances belonging to the same event type.
- This directly conflicts with the anchor classification gate, which requires triggers of the same event schema to map near the shared schema query vector $q$.

---

### Issue 4: Competing Gold Triggers Excluded from Trigger-to-Argument Column Loss

> **Verdict (verified 2026-10-22 by 4 independent read-only tracers + 8 blind skeptics; claims C1/C2 survived 0/2 refutations each): CONFIRMED as a code fact, with four qualifications the review omits.**
> 1. The exclusion is enforced TWICE: the `gold_all`/`trained` set (records.py:1877-1878, :1760) AND, independently, `_negative_instances` drops any instance overlapping ANY record's gold trigger span (:1753-1754, :1762). Removing `gold_all` alone changes nothing. The negatives are the k=8 highest-`object_logits` non-gold instances picked ONCE per group and reused for every column -- not "the K hardest false triggers in that argument's column" as configuration.py:131 says (docs drift).
> 2. Scope: one record group per EVENT TYPE (model.py:2707-2708), so only same-type sibling events are affected; a different-type trigger IS an eligible negative. Exercised (upper bounds, whole records): CASIE 94.2% of (event, role, argument) terms, WikiEvents 80.8%, CMNEE 32.2%, DuEE 5.5%, MAVEN 0% (77,993 events, 0 with arguments -- the column loss never fires on the corpus that is ~80% of the in-run event signal).
> 3. "Mitigated" holds only partly. `_instance_field_loss` -> `_list_field_bce` (:1235-1241) with eb22c's `record_role_hard_negatives: 8` puts target 0 on A's argument column X in B's row ONLY when X is among B's 8 highest-scoring wrong columns. Traced with the real function: X at logit 2.5 gets gradient; X at 0.5 outranked by 8 wrong columns at 1.5 gets NONE although sigmoid(0.5)=0.62 clears the 0.3 decode threshold. And decode is per ROW for roles (non-exclusive): A and B BOTH emit X if both clear 0.3; only `global_decode`'s beam conflict penalty (0.5) then arbitrates. (`_negative_role_loss` never runs in eb22c: `record_negative_instances` absent -> 0.)
> 4. The obvious fix is wrong as stated. Same-type siblings legitimately SHARE the filler in 34.1% of CMNEE and 51.8% of DuEE sibling terms, so "add other gold triggers as column negatives" would teach the junction to deny true co-ownership. A correct fix adds, for column c of record A's field f, only sibling gold instances whose own gold for f does NOT contain c (bypassing the overlap filter for those), keeping the k non-gold negatives. `_column_loss` ignores `record_coreferent_ownership` (it already treats every seeded mention as a co-owner), and the five train files hold 0 multi-mention events, so co-ownership is moot here.
> Scope of effect: argument binding (event_argument precision on same-type multi-event documents: CASIE and WikiEvents, where English arguments score ~0). Not the trigger level directly -- emission is gated by the pair logits and the record loss puts no gradient on the pair scorer/proposer parameters -- but it reaches the pair logits INDIRECTLY through the shared encoder (the record loss's encoder gradient is 24-51, comparable to entity's; the gradient-conflict run found it never opposed to the event gradient).
**File**: [gliner2/models/boundary/records.py#L1877-L1882](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/records.py#L1877-L1882)

#### The Code
```python
gold_all = {i for _, insts in owners for i in insts}
false = _negative_instances(group, records, anchor_qid, gold_all, k)
```

#### Mechanism & Impact
- `_column_loss` asks: "Given an argument candidate $c$, which instance $i$ owns it?"
- In multi-event documents, `gold_all` pools all gold instances across all events.
- When selecting negative candidate instances (`false`) to compete for argument candidate $c$, any instance in `gold_all` is excluded.
- **Consequence**: If an article has Event 1 (Trigger A, Attacker X) and Event 2 (Trigger B, Victim Y), Trigger B is in `gold_all`. For Attacker X, Trigger B is **never** included in `false`.
- The column loss never trains Trigger B *not* to claim Trigger A's arguments. Cross-instance competition between valid triggers is omitted, leaving argument-trigger assignment under-constrained.

---

### Issue 5: Silent Argument Drop on Missed Triggers & Absent Schema Negatives

> **Verdict: 5a CONFIRMED (already documented as the supervision gate); 5b PARTLY REFUTED.** The record head gets no absent-schema negative (`record_object_loss` 0 on all 725 logged steps), but the anchor gate decodes the pair score, which IS trained with absent event queries as negatives.
**Files**:
- [gliner2/models/boundary/records.py#L1925-L1934](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/records.py#L1925-L1934)
- [gliner2/training/trainer.py#L1563-L1570](file:///Users/williamroe/development/git/GLiNER2/gliner2/training/trainer.py#L1563-L1570)

#### The Code
```python
if inst_idx is None:
    # Gold anchor wasn't proposed; anchor gate target=0 for this schema
    # would be wrong because the event IS present, we just missed the trigger.
    _note_anchor_gate("anchor_not_proposed")
    continue
```

#### Mechanism & Impact
1. **Missed Trigger Cascade**: If a gold trigger does not appear in the top proposed boundary candidates, `inst_idx is None`. The record head logs `anchor_not_proposed` and skips the instance entirely. All argument field assignment losses, link junction losses, and coref losses for that event are dropped.
   - During early training or as gold trigger injection decays ([trainer.py#L1563](file:///Users/williamroe/development/git/GLiNER2/gliner2/training/trainer.py#L1563)), any temporary dip in proposer recall eliminates record head supervision, starving the head of training signal.
2. **Absent Schema Invariance**: When an event schema is absent from a document, `object_count = 0` and `object_loss = 0`. The record head's anchor gate is only trained on non-trigger candidates within *active* event groups. It receives zero negative gradient suppressing the anchor gate on unprompted/absent event schemas.

---

### Issue 6: Unanchored Proposer Prior and Downstream Threshold Sensitivity

> **Verdict: PARTLY REFUTED.** Shift-invariance of the listwise loss is true. But the level oscillates in both directions rather than drifting down, entity levels swing as much with the prior a minor part, and both treatments aimed at it (separate event scorer, task_lr 4e-5) were null inside the seed floor.
**Files**:
- [gliner2/models/boundary/scoring.py#L143-L154](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/scoring.py#L143-L154)
- [gliner2/models/boundary/proposal.py#L149-L162](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/proposal.py#L149-L162)
- [gliner2/models/boundary/losses.py#L107-L125](file:///Users/williamroe/development/git/GLiNER2/gliner2/models/boundary/losses.py#L107-L125)

#### Mechanism & Impact
- In `SparseBoundaryPairScorer`, the logit score for a candidate span is:
  $$\text{score} = \text{compat} + \text{prior} + a_{\text{start}} + b_{\text{end}}$$
- The prior is trained via `proposal_listwise_loss` (softmax cross-entropy over candidates). Softmax is shift-invariant ($\text{softmax}(x) = \text{softmax}(x + c)$), providing zero constraint on the mean logit level.
- Only pointwise BCE penalizes absolute score magnitude, but its loss weight and gradient are often overwhelmed by listwise and relation losses.
- Downstream event decoding relies on a fixed probability threshold:
  $$\sigma(\text{logit}) > 0.30 \implies \text{logit} > -0.847$$
- Across training, the prior mean can easily drift by $-1.5$ logits while preserving 100% of candidate ranking. Once the top candidate drops from $\text{logit} = -0.5$ to $\text{logit} = -1.2$, downstream event metrics collapse to zero despite ranking accuracy remaining high.

---

## 3. Recommended Investigation Plan

Following the repository's core guidelines (trace first, prove root causes with code and logs before altering implementations):

1. **Verify Parameter Group Routing**:
   Run a minimal inspection script with `uv run python` to print parameter group assignments from `trainer._create_optimizer()`:
   ```python
   for group in optimizer.param_groups:
       print(group["lr"], [name for name, p in model.named_parameters() if any(p is p_g for p_g in group["params"])])
   ```
   Confirm whether `boundary_head.candidate_encoder` and `boundary_head.boundary_encoder` are in the `encoder_lr` group.

2. **Measure `LinkJunction` Field 0 Impact**:
   Inspect gradient magnitudes flowing through `link[0]` into `fq[0]` and `inst_states` during training to confirm if field 0 self-linking is distorting trigger representations.

3. **Audit Coref Loss Distribution**:
   Instrument `_coref_link_loss` to log the ratio of batches with `len(pos) == 0`. Verify whether the pure negative loss is actively driving trigger representations apart.

4. **Trace Competing Trigger Ownership in `_column_loss`**:
   Check multi-event documents in the dataset and evaluate whether excluding `gold_all` from negative instances allows rival triggers to claim out-of-event arguments without penalty.

5. **Profile Proposer Score Trajectories**:
   Use [tools/events_working_papers/gradient_conflict.py](file:///Users/williamroe/development/git/GLiNER2/tools/events_working_papers/gradient_conflict.py) or evaluate checkpoint logits to measure the mean logit shift of gold triggers relative to the 0.30 threshold across checkpoints.
