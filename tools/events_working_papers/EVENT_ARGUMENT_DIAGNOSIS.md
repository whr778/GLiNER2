# Why `event_argument` sits at 0.118, and what would move it

Status: **diagnosis, measured 2026-09-15** on `whr778/gliner2-eb16-rebuild-tr`, 18,786-record
blind test, greedy decode, threshold 0.5. Every number here is from that run's
`test_metrics.json` or from counting the test splits directly. Companion to
[[JOINT_IE_SCALING]] (Tier 2) and [[PAPER_0_FOUNDATION]] §10.

---

## 1. The model finds the arguments. It cannot attach them to the right event.

Same checkpoint, same predictions, three scoring keys:

| key | what it requires | F1 | P | R |
|---|---|--:|--:|--:|
| **strict** | (event_type, **trigger**, role, entity) | **0.1178** | 0.149 | 0.097 |
| **relaxed** | (event_type, role, entity) — trigger link dropped | **0.5783** | 0.701 | 0.492 |
| **fair** | relaxed + partial credit for boundary errors | 0.5737 | 0.707 | 0.483 |

**Dropping one element of the tuple multiplies the score by roughly five.** And
`fair ≈ relaxed` (0.574 vs 0.578) says partial boundary credit adds almost nothing on top
of relaxed — so boundaries are not where the loss is.

The error taxonomy over 18,557 gold arguments agrees:

```
COR          8,262   44.5%   correct
FN           9,059   48.8%   never proposed at all
FP           3,212
BES+BEL+BEO    988    5.3%   boundary errors
LE+LBE         248    1.3%   label errors
```

**Boundary and label errors together are 6.6%.** This is not a span problem and not a
typing problem. It is a *binding* problem, plus a large block of arguments that are never
emitted.

The per-event keys say the same thing from the other direction: **exact argument-set rate
0.0090** and **mean Jaccard 0.061** over 7,908 events. Almost no event is recovered
*completely*, while pooled relaxed micro sits at 0.578 — the model is getting many
individual arguments right and very few whole events right.

## 2. The mechanical cause: the mention path pools same-type events

On the mention path an events group is compiled as `[V]` trigger + role queries, and a
document gets **one instance per event type**. Two `Attack` events in one document collapse
to a single `Attack` trigger, so the arguments of the second either vanish or bind to the
first — and strict scoring keys on the trigger.

Measured on this blind test's own event corpora:

```
documents carrying events : 3,186
gold event instances      : 17,135
sharing a type in-document: 10,997   (64.2%)
```

by corpus: **cmnee 1,606 documents**, maven 346, casie 102.

**Two-thirds of gold event instances are structurally inexpressible by the decoder that
was measured.** The surviving third puts a ceiling of roughly `0.578 × 0.36 ≈ 0.21` on
strict argument F1; the observed 0.118 is the same order. That is consistency, not proof —
but it is the only hypothesis on the table that predicts the 5× strict/relaxed gap, the
negligible boundary error rate, and the near-zero exact-set rate simultaneously.

`compile_record_specs` already carries the upstream version of this number: the cap costs
**78.8% of gold instances on CASIE, 62.5% on WikiEvents, 38.3% on MAVEN, and 0.0% on RAMS**
(RAMS is 100% single-event documents — which is exactly why the RAMS-based argument curves
never surfaced this).

## 3. Was `event_records: true` ever configured? NO — and that is the answer to the obvious objection

Events *were* trained with mmBERT. The objection "so the head has seen events" is right
about the **mention** path and wrong about the **record** path, and the distinction is the
whole diagnosis.

`event_records` is a `boundary_head` setting, default **False**. Verified 2026-09-15:

| where | occurrences of `event_records` |
|---|--:|
| `config/base/eb16-rebuild-tr.yaml` | **0** |
| every config under `config/base/` | **0** |
| configs setting it **true** anywhere | **2** (both archived Tier 2 arms) |

So in every base run, events were supervised as mention-path queries and the **record head
was supervised on `json_structures` only. It has never been trained on an event.** That is
not an inference; it is what the configs say.

## 4. Why Tier 2 already failed, and what that does and does not prove

`event_records: true` routes events through the record head, which is multi-instance by
construction and lifts the cap. It was tried twice and both arms are archived:

- **CASIE Tier 2** — multi-instance events *worked structurally* and scored **0.0036
  against a 0.2998 control**.
- **MAVEN Tier 2** — trigger strict **−0.008**; nothing gained.

The recorded cause is head initialisation: the record head had no event competence to
decode with. **That is a confound, not a refutation.** Head-init is the largest single
effect this programme has measured — fresh → IE-pretrained heads moved RAMS arguments
**0.042 → 0.462**, eleven-fold ([[PAPER_0_FOUNDATION]] §10.5).

Switching the path and the head's competence at the same time measures their sum. The
arms did that, and the sum was negative.

## 5. What follows, in order

1. **Warm the record head on events before switching the path.** The Tier 2 arms changed
   the decode path while the head was naive; warming first separates the two.
2. **Re-run Tier 2 on CMNEE, not CASIE.** CMNEE dominates the affected mass here (1,606 of
   2,054 affected documents). CASIE was the previous venue and it is both smaller and
   harder.
3. **Report relaxed beside strict, always.** A metric that moves 0.118 → 0.578 on one key
   change has been measuring binding while being read as extraction. Any future claim about
   "argument quality" must say which it means.
4. **Re-read the argument curves with this in mind.** The head-init data-scaling curve
   (10k/40k/100k → 0.050/0.115/0.158) was run on **RAMS**, which is 0.0% affected. It
   therefore says nothing about the pooling ceiling, in either direction.

## 6. Caveats, stated because the number is quotable

- **64.2% is this mixture's number**, dominated by one Chinese corpus. It is not a
  universal property of event extraction, and a differently-composed test would give a
  different ceiling.
- **The ceiling arithmetic is consistency, not a proof.** Confirming it means running the
  decoder against gold with the cap lifted and seeing strict argument F1 move toward
  relaxed — that experiment has not been run.
- **This diagnosis does not explain the 9,059 never-proposed arguments** (48.8%). Pooling
  explains why correct arguments bind wrongly; it does not by itself explain absence.
  Recall at 0.492 *relaxed* means half the arguments are missing before binding is even
  considered. That is a second, independent problem and this document does not solve it.
- **The record head is entangled with work in flight.** The cardinality A/B trains the same
  head. If warming it on events is the real lever, the two are not independent and should
  be sequenced deliberately.
