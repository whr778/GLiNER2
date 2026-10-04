# Existence head for natural-mode event records: specification

**Status:** spec only, 2026-10-04. Nothing built. **Build after the p3arg result** (see Prerequisite).

## 1. The problem, measured

In natural mode, an event instance *is* a trigger candidate:
- `forward_group` sets `inst_states = anchor_states` and `object_logits = anchor_logits` (the trigger query's pair logits). See `records.py`, natural branch.
- Decode keeps an instance iff `sigmoid(object_logits) >= anchor_threshold` (`decode_group`).
- Exclusive role fields give each argument candidate to the **strongest instance only**, ranked by that same object probability. That is the "each subject belongs to one king" rule.

Whether an event exists is therefore the trigger span score alone. Nothing about the event trains it:

| Measurement | Result | Source |
|---|---|---|
| Natural-mode object loss | identically 0 | `compute_group_loss` |
| Instance hypotheses receiving ANY record-head loss | 65 of 60,390 (0.11%) | `measure_record_supervision.py` |
| Gold triggers decoded but cut by the gate | ~31% | trigger-miss probe, eb18 |
| Argument evidence separating gold from false triggers below the gate | AUC ~0.5 (CMNEE reversed, 0.23-0.42) | `trace_argument_evidence.py` |
| Same, under relaxed (boundary-variant) matching | AUC 0.48-0.52 | same |

Design C (full normalisation) was measured as a **corner case**: the gold is trigger-keyed (0% multi-trigger events with arguments), and nested argument sets are mostly <=7% of same-type pairs. The broadly applicable piece of C is an **existence loss**. This spec gets it without C.

## 2. Design

Replace the natural-mode existence logit with

$$z_i = a_i + \mathbf{w}^\top \phi_i + b$$

- $a_i$ is the anchor (trigger) logit, unchanged.
- $\phi_i$ is the instance's **argument evidence**, computed from the assign logits $A_{i,f}$ that `forward_group` already produces for every instance and role field. No gold is used, so it is identical at train and decode time:
  - mean over role fields of $\max_c \sigma(A_{i,f,1+c})$;
  - max over role fields of the same;
  - share of role fields with $\max_c \sigma \ge 0.5$;
  - optionally a projection of the instance state $h_i$ (variant E1b).
- $\mathbf{w}, b$ form a new `existence_head` in the record decoder, **zero-initialised**. At a warm start, $z_i = a_i$ exactly, so decode is bit-identical at step 0 and the head learns only what the data supports.

Variant **E0** (parameter-free fallback): $z_i = a_i + \alpha(\bar e_i - c)$ with a fixed $\alpha$. No new weights, so no checkpoint-loading question, but $\alpha$ is a guess.

**Gradient flow.** The existence loss reaches both $a_i$ (trigger scores) and, through $\phi_i$, the assign logits. This is the intended two-way coupling: the king learns from his subjects, and the subjects are shaped by whether the king is real. Risk: arguments bend to please existence. Mitigation: `record_existence_detach_evidence: true` stops gradient into the assign logits, to be tested as its own arm.

## 3. Training target

Per natural event group:
- **Positives:** the instances seeded by a gold trigger. This is the same set the field loss trains, so a gold trigger that is neither proposed nor injected gives no positive.
- **Negatives:** the K highest-scoring false instances. This reuses `_negative_instances`, which skips any span overlapping a gold trigger, so boundary variants are not trained as "no event". Absent-type groups (no gold) contribute negatives only.
- **Loss:** BCE on $z_i$, with positives and negatives **averaged separately** and summed. That keeps class balance, and repeats the `absent_reduction` dilution lesson: neither side enters the other's denominator.

Settings, all opt-in with defaults that are bit-identical:

| Setting | Default | Meaning |
|---|---|---|
| `record_existence_weight` | 0 (off) | weight of the existence loss |
| `record_existence_negatives` | 8 | K false instances per group |
| `record_existence_detach_evidence` | false | stop gradient into the assign logits |
| `record_existence_gate` | false | decode gates on $\sigma(z_i)$ instead of $\sigma(a_i)$ |

The shared pool refuses these settings, as for `record_negative_instances`.

## 4. Decode

With `record_existence_gate: true`, `decode_group` uses $\sigma(z_i)$ for **both** the instance gate **and** the exclusive-allocation ranking. So the head changes which events survive **and** which instance claims a contested argument. The record anchor threshold must be re-swept on validation, since $z$ is on a new scale.

Gate on/off is a separate setting so an A/B can separate the two effects:
- **training effect:** representations improved by the loss, measured with the gate off;
- **decode effect:** the new gate itself, measured with the gate on.

## 5. Prerequisite: the evidence must exist first

$\phi_i$ is only useful if argument evidence separates gold from false instances. On eb18 it does not (AUC ~0.5), because the role loss was diluted about 768x and false instances were never trained. **p3arg** (role hard negatives + negative instances) targets exactly that. Plan:

1. Re-run `trace_argument_evidence.py` on the best p3arg arm's model.
2. If the below-gate AUC rises clearly (say >= 0.6), the existence head has something to read. Build it on top of that arm.
3. If it stays ~0.5, argument evidence cannot vouch for triggers even when trained. The head would reduce to a re-learned trigger score. **Stop** and reconsider.

## 6. How it gets proven (trace, code, trace, test)

- **Trace before:** on the real CASIE batch, $a_i$ for gold vs false instances, and $\phi_i$ per instance.
- **Trace after:** $z_i = a_i$ exactly at zero init; the existence loss is finite; the counts of positives and negatives; the loss scale against the field loss.
- **Tests that can fail:**
  - zero init means bit-identical decode;
  - positives are exactly the gold-seeded instances;
  - negatives never overlap a gold trigger;
  - positive and negative means are kept apart;
  - the shared pool refuses the settings;
  - the gate on/off switch changes which instances survive.
- **Proof line from inside training:** cumulative positive and negative counts, and the mean $\sigma(z)$ for each, with the usual backoff.
- **Fast A/B (p3arg design):**
  - control (x2, for the floor);
  - p3arg winner;
  - winner + existence (gate off);
  - winner + existence (gate on).
- **Readouts:**
  - trigger-miss probe (below-gate share, 31% today);
  - event_trigger and event_argument precision and recall by language;
  - argument-evidence AUC;
  - **every other head** (collateral).

## 7. Open items

- **Warm-start loading of the new parameters.** `from_pretrained` must accept a checkpoint without `existence_head` keys and keep their zero init. That needs checking before E1; E0 avoids it.
- **The trigger field** (identity map, P(ABSENT) = 0) is left untouched. Existence lives in $z$, not in that field.
- **Interaction with `proposal_gold: identity`.** Both change the trigger-score side. Test them on the same base and do not stack them untested.
