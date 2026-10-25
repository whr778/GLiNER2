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

*Results update 2026-10-22 18:50 UTC: Issue 1 A/B DONE (NULL, out of eb22d); Issue 4 fix built, its A/B DONE (NULL at one epoch, out of eb22d); the geometry/distance findings that grew out of this review (junction geometry channel, coref distance bias) have their own catalog rows and arms (`eb22c-geomgelu-trace`, `eb22c-corefdist-trace`).*

| # | issue | verdict | evidence |
|---|---|---|---|
| 1 | head encoders on `encoder_lr` | **CONFIRMED; impact MEASURED: NULL** (A/B `eb22c-encfix-trace`, one epoch, 2026-10-22) | `boundary_head.boundary_encoder` (463,360) + `boundary_head.candidate_encoder` (197,376) = 660,736 head params match `"encoder" in name` and train at 2e-5 **A/B RESULT:** `encoder_lr_backbone_only: true` (9390f36) trained the two head encoders at task_lr for one epoch (16,586 steps, ~$24); proof line `[optim] ... head encoders (30 tensors) on task_lr`. Threshold-robust read vs the control, seed43 and sep: mean strict trigger F1 4k-12k maven 0.050 (0.045-0.055), cmnee 0.307 (0.303-0.385), casie 0.046 (0.043-0.051), duee 0.167 (0.077-0.317); 16k on the controls' 12k->20k path; in-run gate means type 0.106 / trigger 0.014 vs control 0.204 / 0.040 and seed43 0.114 / 0.026. Every delta inside the three-run spread. The misgrouping is real by construction but changed nothing measurable; NOT taken into eb22d (user decision 2026-10-22). Catalog row `eb22c-encfix-trace` |
| 2 | junction self-link on field 0 | **CONFIRMED (corrected 2026-10-25); MEASURED: the term OPPOSES the role gradient early in training** | field 0 IS supervised (`_instance_field_loss` loops every field) and `LinkJunction.forward` adds a link to it. CORRECTION: the 2026-10-22 line here said `decode_group` loops every field so 'discarded at inference' was unverified; `decode_group` DOES skip it (records.py:1082-1091, the anchor branch copies the instance's span and `continue`s, upstream 2e8a44e 2026-07-28), and no other reader exists (merge and column loss read list fields; natural matching is by anchor span). MEASURED 2026-10-25 (`probe_anchor_field.py`): anchor vs role gradient at step 1k norm ratio 0.43-0.71 with cos -0.71 (encoder layer 0) to -0.16 on every shared group, -0.88 on the junction; at 33k mostly solved (NLL 0.002) and orthogonal. Catalog 2026-10-25 |
| 3 | all-negative trigger coref loss | **REFUTED as stated** | the quoted `_coref_link_loss` is not our code; ours pairs DIFFERENT gold records (distinct events, correctly negative) + k hardest false triggers; measured: the trigger-link gradient is ~7% of the entity gradient and orthogonal to entity/event (|cos| <= 0.12, gradient-conflict run) |
| 4 | other gold triggers excluded from column negatives | **CONFIRMED, scoped, partly mitigated** (12-agent verification 2026-10-22); **FIX BUILT, A/B DONE: NULL at one epoch** | exclusion is DOUBLE-enforced (`trained` set :1877-1878 AND the gold-span overlap filter :1753-1754/:1762) and reaches only SAME-TYPE sibling events (one record group per event type). Exercised in 94% of CASIE, 81% of WikiEvents, 32% of CMNEE argument terms; NEVER in MAVEN (0 argument gold) and 5.5% of DuEE. Mitigation is partial: B's row BCE puts target 0 on A's argument only when it is among B's 8 hardest wrong columns (hard_k=8); a column at logit 0.5 outranked by 8 wrong columns at 1.5 gets NO gradient though it clears the 0.3 decode threshold, and decode is per-row (roles are non-exclusive), so A and B BOTH emit it. A naive 'siblings as negatives' fix is WRONG where siblings share the filler: 34% of CMNEE and 52% of DuEE sibling terms. Reaches the trigger level only through the shared encoder (record-loss encoder gradient 24-51, measured) **FIX:** `record_link_column_siblings` (d6cf209): same-type sibling gold triggers become column negatives only for the columns their own gold does not hold (a sibling sharing the filler is never a negative); bit-identical when off; traced on a real CASIE record (3 Cyber.Phishing events: +16 sibling pairs over 8 terms, gold won 7 untrained). **A/B** `eb22c-colfix-trace` (one epoch, H100): the term is live in-run -- 1,936 sibling pairs beside 4,664 false pairs over the first 1,000 groups, sibling contrast 0.52 at the start -> 0.78 by 475k groups; trigger / type reads inside the band through 10k; CASIE argument precision (the pre-set read) 0.01-0.02 vs the controls' 0.01-0.07 so far; **RESULT (18:40 UTC):** NULL. On the full CASIE val (137 same-type groups, 39,384 sibling pairs) the held-out sibling contrast of the trained arm ends at 0.774 at 16k, where the three untrained originals converge (0.776-0.780 at 20k); the false-trigger contrast 0.775 inside the spread; CASIE argument F1 at the floor for every run through 16k. The exclusion exists, the term learns on training batches, and geometry + content already do the sibling separation on held-out data. Flag stays off in eb22d. Catalog row `eb22c-colfix-trace` |
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

> **Verdict: CONFIRMED (corrected 2026-10-25).** Field 0 is supervised and the junction adds a link to it. The quoted decode block does not exist verbatim, but `decode_group` skips the anchor row all the same (records.py:1082-1091), so the row is trained and never read. Measured 2026-10-25: early in training (step 1k) its gradient is 43-71% of the role terms' on the shared weights and points against them (cos -0.16 to -0.71; junction -0.88); by 33k it is solved and orthogonal. Arm proposed, not run (catalog 2026-10-25).
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

### Issue 7: 81 of 334 Tensors Receive No Training Signal (found 2026-10-23, tracing the corefdist resume)

**Status: MEASURED, informational (no fix proposed).** Reading `checkpoint-epoch-1/training_state.pt` of
eb22c-corefdist-trace (a full epoch, 16,586 steps): 70 of the 334 optimizer parameters have NO Adam moments at all
and 11 more have moments that are exactly zero. Both groups are structural, not a defect of the run:

- **70 tensors, 892,417 params (0.28%)**: `boundary_head.shared_pool_builder.*` and `boundary_head.shared_pool_scorer.*`.
  They are built unconditionally (`model.py` ~329-335) but enter the graph only under `candidate_pool == "shared"`
  (`model.py` ~564); every eb22c config runs `candidate_pool: per_query`, so their grads are `None` and AdamW skips
  them. They sit at init in every checkpoint.
- **11 tensors, 813,570 params (0.26%)**: `record_decoder.instance_embed`, `object_head`, `latent_seed_head`,
  `q_proj`, `k_proj`, `v_proj` -- the LATENT / ANCHORLESS instance paths of `forward_groups_dense`. The dense forward
  computes those branches for every group and selects them with `torch.where(modes == 1 / 2)`
  (`records.py` ~640-672), but every record spec in training is `mode="natural"`: events via
  `schema.py:762` (`mode="natural", anchor="trigger"`) and JSON structures via `runtime.py:2386`
  (`record_mode = "natural"` whenever records are enabled; the first field is the anchor). `latent` and `anchorless`
  are reachable only through an explicit `Schema.structure(..., mode=...)`, which no corpus uses. The branches are
  masked, so their gradients are exact zeros; AdamW still applies decoupled weight decay to them every step.

Structures DO train through the natural path: in-run structure strict micro F1 0.000 -> 0.147 over epoch 1
(support 308). Impact on the eb22c programme: none on the metrics; ~1.7 M dead parameters in every checkpoint and
in the warm-start name space. If the latent/anchorless modes are never going to be trained, removing them from the
dense forward would also drop two masked branches from every record step.

### Issue 8: The Relation Loss Never Sees Most of Its Gold Pairs (TRACED 2026-10-23)

**Status: TRACED on the 30k control; fix BUILT as `relation_keep_gold_pairs` (off by default); overfit matrix RUN on an A10 and the one-epoch arm `eb22c-relfix-trace` DONE 2026-10-23 -- NEGATIVE** (robust relation F1 inside or below the two-control band at every step, below both controls on the sparse corpora at 16k, plus a CMNEE / DuEE trigger cost; mechanism: an un-rebalanced ~3.5x relation loss). Not in eb22d. [[EXPERIMENT_CATALOG]] 2026-10-23.
`_relation_loss` scores the pairs `RelationPairGenerator.generate_batched` keeps: endpoints must clear
`relation_argument_proposal_threshold` (0.2 in eb22c, upstream default 0.0), rank in the top `relation_heads/tails_per_type`
(32) of their relation's member queries by ENTITY pair probability, and the pair must rank in the top `relation_pair_cap`
(64) by product score. Gold mentions are all in the candidate pool (gold injection works), but the control scores them at
0.003-0.17, so on 8-record batches (`tools/events_working_papers/trace_relation_gold_coverage.py`): biored 5 of 30 gold
pairs reach the loss as positives, professorbob_re 4 of 27, sentence_rex 5 of 6. Threshold 0 alone recovers little
(8 of 30, 11 of 21): the ranking caps drop the rest. The missing gold is ABSENT from the loss (neither positive nor
negative). This is the shape of "cannot fit dense corpora" (TODO 40): the one corpus that fits (sentence_rex) is the one
whose gold reaches the loss. Fix: append every gold pair the ranking dropped to the scored pair set, training only,
non-structural; proof line `relation pairs: gold kept by the proposal ranking K, appended A of G gold pairs`. Traced:
biored seed 0, flag off relation_loss 0.078 with 8 of 29 gold; flag on "kept 8, appended 21 of 29", loss 0.276.

### Issue 9: Schema Dropout Removes 20% of Gold Relation Instances per Collate (TRACED 2026-10-23)

**Status: TRACED; controllable today by `training.schema_dropout: false`; NOT set in any eb22 base.** Upstream's
`SamplingConfig` (gliner2/processor.py) is live in every training collate: `remove_relations_prob` 0.2 drops each gold
relation ITEM independently (`_process_relations`), `swap_head_tail_prob` 0.2 swaps the head/tail field order per type,
`remove_events_prob` 0.2 drops each gold event TYPE (its mentions leave the schema), `remove_json_structure_prob` /
`remove_json_field_prob` 0.2, `synthetic_entity_label_prob` 0.2. Measured: the same 8 biored records collated 10 times
emit 26-32 gold pairs of 37 alignable (mean 0.78); one record's single relation is emitted on some collates and not
others. A dropped instance whose type keeps other instances is no longer gold, so if the generator proposes that pair
the loss scores it as a NEGATIVE: a fifth of the relation labels flip sign between collates ("positives below negatives
for 250 steps", 2026-10-19). A dropped event type simply leaves the sample (fewer positives, no false negatives). The
flag was added 2026-09-16 (2b379e1) after three false readings in small A/B batches and judged "sensible regularisation
at 174k records"; that predates knowing the drop is per instance for relations and that the relation head cannot fit.
Also found: the `sampling: {}` block in the gate3 configs is wired to nothing (inert).

### Issue 10: The Classification Head Collapses to One Label on News-Style Tasks (MEASURED 2026-10-23)

**Status: MEASURED on the 30k control; the sampling is NOT the cause** -- `eb22c-clsfix-trace` (all three classification sampling probabilities -> 0) DONE 2026-10-23, NULL: the probes equal the control at 12k and 16k on every task. The news one-label prior lives in the mix; label imbalance across the news classification tasks is the next suspect ([[EXPERIMENT_CATALOG]] 2026-10-23; TODO 48). `batch_extract` with each record's own gold menu,
100 val records per task: chfinann_event (5 labels, 25k training records) accuracy 0.99 with predictions spread over
the classes; cc_news_haiku45 'topic' (16 labels, in the mix) accuracy 0.16 with 'politics' predicted 69 times;
cc_news_haiku45 'sentiment' (3 labels) 0.47 with 'neutral' 96 times; masakhanews_eng 'news topic' (7 labels, NOT in
the mix) 0.14 with 'politics' 93 times. The same encoder's frozen label-token states carry the signal: a fresh
768 -> 1536 -> 1 head trained on them reaches 0.71 blind accuracy on masakhanews_eng (FG_GEGLU.md 3.5). So the
deficit the memory records (9-27% vs upstream's 82%) is the TRAINED head's behaviour on news-style tasks, a prior for
one label ('politics', 'neutral') over content, while a well-represented closed task (chfinann) is fine.

Measured alongside, from the training collate (masakhanews_eng, 200 records x 5 collates, the SamplingConfig
defaults): the menu shrinks to 5.75 of 7 labels on average, the true label is ABSENT from the menu in 10.0% of collated
records (all-zero gold: trained as "none of these"), 30.8% carry synthetic names with descriptions, 17.2% synthetic
names WITHOUT descriptions (nothing to learn from), i.e. 74.2% usable supervision per step. Upstream trains under the
same defaults, so this cannot be the whole deficit; whether it is part of it is testable with
`tools/train/overfit_classification.py` (control vs `--no-schema-dropout`, 64 records x 400 steps), which ran on an A10
(2026-10-23 18:42-18:52 UTC): with the sampling ON the head still fits 64 records (masakhanews 0.92, scientific_text 0.83
accuracy at 400 steps), so the full-run collapse is NOT the sampling alone; with it OFF the fit is faster and steadier
(masakhanews 1.00 by step 250, 0.95 at 400; scientific_text 0.92 vs 0.83; macro F1 0.83 vs 0.67). The sampling is a
fit cost on top of whatever drives the one-label prior in the mix (label imbalance across the news classification
tasks is the next suspect: 'politics' and 'neutral' are the labels it defaults to).

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
