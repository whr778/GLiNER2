# Closure validity: event arguments, the boundary head, and the October 22 review

**Date**: 2026-10-25
**Target**: Which closed argument and boundary results still stand, and which were closed by an instrument that could not fail.
**Entrypoint**: [tools/train/train.py](tools/train/train.py)
**Prior review**: [20261022_CODE_REVIEW.md](20261022_CODE_REVIEW.md)
**Checkpoint read**: `whr778/gliner2-eb22d` epoch 2, step 33,172, local copy `/Volumes/Development/tmp/eb22d_epochs/checkpoint-epoch-2`. eb22d is the live base. This review does not change it.

A closure is valid when it says: this treatment, on this instrument, did not clear its floor. It is not valid when the write-up then says the hypothesis is false.

---

## 1. Verdict

Most one-epoch NULLs are valid stops on a named treatment. Several of the closures that matter for event arguments are not valid as refutations. The boundary record head in natural mode is still one-directional, and that fact was not closed by any run.

Existence of an event instance is the trigger pair score. In natural mode `object_loss` is 0. Field loss supervises gold instances only, so a false trigger is never taught "no arguments." The column loss is the argument-to-trigger direction, and on English it is mostly a distance and order prior (geometry was 86–92% of the junction term on the eb22c 30k control). Coreferent triggers are a trained link whose decode merge is still opt-in. Coreferent arguments are string match ([COREFERENT_LINK_SPEC.md](COREFERENT_LINK_SPEC.md) §8). Cross-document association is not a trained head.

---

## 2. Instruments that could not fail

### 2.1 The anchor-gate refutation (diagnosis S13) does not measure the model's proposal

[EVENT_ARGUMENT_DIAGNOSIS.md](EVENT_ARGUMENT_DIAGNOSIS.md) §4f-i reports **0 of 362** event records dropped by the supervision gate, on `whr778/gliner2-eb16-eventrecords-tr`, and closes the gate as the recall floor.

[tools/train/probe_anchor_gate.py](../train/probe_anchor_gate.py) calls `model(batch)` in train mode and never calls `set_gold_injection_prob`. The head's default is `1.0` (`gliner2/models/boundary/model.py`, `_gold_injection_prob`). At `1.0`, `gliner2/models/boundary/proposal.py` keeps every in-range gold span. `anchor_not_proposed` then cannot fire except on a span the proposer cannot represent at all. A count of zero on that instrument is what the instrument prints when the gate is broken and when it is not.

What still stands comes from a different instrument. On the 2026-10-21 autopsy, gold-trigger **reach is 0.96–1.00** and rank is about 0 at inference, where injection is 0. A trained model proposes the gold trigger. That does not say training never drops argument supervision. After the first 15% of steps, injection anneals from 1.0 to 0.25 inside `GLiNER2Trainer._backward_one`. At step 33,172 of 82,930 the schedule is about **0.78**, not 0.25. That rate was unmeasured when this review was written. Section 6 records the measurement.

### 2.2 Option 4 is not refuted

[EVENT_ARGUMENT_DIAGNOSIS.md](EVENT_ARGUMENT_DIAGNOSIS.md) §4k says the 2026-09-21 re-run was void (epoch mismatch, different test sets) and then says "Option 4 is refuted," quoting event_argument strict F1 **−0.0300**. [TODO.md](TODO.md) row 43, written later, says the first run had no reachable negatives and a shared head, the re-run was void, and the verdict is regime-bound. A void comparison across a changed test set is not a result.

### 2.3 Option 2's summary is stronger than its measurement

[OPTION_2_TYPED_ROLE_CONSTRAINTS.md](OPTION_2_TYPED_ROLE_CONSTRAINTS.md) measured the typed-margin mask on held-out val: it fires on **0.339%** of candidate cells and moves the loss **0.28%** on average (median effect **0.00%**). The disallowed fillers are not the competitors. That is a valid reason not to spend the training run. The same section says the decode-time constraint was not tested. The diagnosis table's "decode-time a null on F1" overclaims that paper.

### 2.4 `candidate_pool: shared` is closed for entities only

Entity strict F1 **−0.0745** at 2 epochs and **−0.0435** at 4 epochs against a 2-epoch control stands. The entity head does not use the record loss. The event-argument "null" was taken while dense record loss was diluted by padding (catalog note, 2026-09-30, fix `426a38d`). Those event deltas are confounded.

---

## 3. Stops that are valid, and the claim they do not close

**encfix, colfix, geomgelu, clsfix** each had a control and an in-run proof line, and the deltas sat inside the eb22c seed spread. They are valid scheduling stops. They do not show the treatment is absent.

| arm | what the run can say | what it cannot say |
|---|---|---|
| **colfix** (`record_link_column_siblings`) | Held-out sibling contrast at one epoch is **0.774** against untrained controls **0.776–0.780**. Geometry already separates same-type siblings on held-out CASIE. | That sibling negatives cannot move argument F1. The pre-registered read, CASIE argument strict F1, was at the floor for every arm through 16k (own-best **≤ 0.01**). The originals reach **0.04–0.06** only at 20k–30k. The write-up first called that read unreadable, then published NULL on the substituted contrast. |
| **encfix** | One epoch at `task_lr` on the two head encoders (660,736 parameters) does not move trigger strict F1 or the score swing. | Anything about argument F1. Separately, those parameters moved less than weight-decay-only modules over 3.5k–30k at `encoder_lr`. That is a measurement of movement, not an A/B. |
| **geomgelu** | A width-64 GELU geometry channel does not clear the junction-gate floor in one epoch. | That geometry capacity cannot help. The rank-≤2 claim for the linear channel is math, and it does not depend on the run. |
| **clsfix** | Turning classification sampling off does not remove the one-label news prior in the full mix at one epoch. | That sampling is free. On 64 records alone, sampling off raised scientific_text accuracy **0.83 → 0.92** and macro F1 **0.67 → 0.83**. Different question. The one-label prior is in the mix. |

**relfix** is a valid negative on a bundle. `relation_keep_gold_pairs` and relation-dropout-off were changed together, and appending gold multiplied the relation loss about **3.5×** (0.078 → 0.276 on the traced batch). Robust relation strict F1 at 16k is below both controls on sentence_rex (**0.125** vs **0.198 / 0.179**) and professorbob_re (**0.022** vs **0.142 / 0.148**). The trace that the ranking keeps **5 of 30** biored and **4 of 27** professorbob_re gold pairs is untouched. Dropout-off alone, and keep-gold with `relation_loss_weight` about 0.3, were not run. The 16k CMNEE / DuEE trigger cost has no control read at 16k. DuEE zeros also appear in the original runs.

**Coref distance.** Removing the scalar `log1p(distance)` term raises held-out coref pair AUC in all **24** control cells (CMNEE **+0.037 to +0.111**). That subtraction is valid. The MLP training arm cleared its pre-registered band at 20k–30k, with the CASIE margin down to **+0.005** at 30k, and at 30k it no longer beat the controls with their scalar term removed. Epoch 1 of that arm was an H100 and epoch 2 an A100 resume. eb22d, the same recipe on one A100, reads like a control at the matched step (epoch 1 CMNEE pair AUC **0.748** vs the arm **0.883** vs controls **0.772 / 0.717**). At epoch 2 the MLP is ranking-neutral (removal **+0.006 / +0.002** AUC). `coref_link_distance: none` was never trained. Putting the MLP in eb22d was a decision on a lead that did not replicate.

**King of Sweden** closes a decode-time gate only. Argument-evidence AUC below the trigger gate is **0.45 / 0.54 / 0.47** (chance) on eb18, and CMNEE is reversed (**0.23–0.42**). Field loss trains gold records only, so false triggers never learn "no arguments," and those scores are uncalibrated. That is a valid negative for reading them at decode. It is the reason a training-side reverse link is still open, not a reason to drop one. Junction Phase 2 (a trigger's row aggregate feeding its existence score) in [JUNCTION_LAYER_SPEC.md](JUNCTION_LAYER_SPEC.md) §5 was never built.

**absneg** is a valid ship decision. The one legal comparison, absneg2, moved event_argument strict F1 **+0.0376** (0.1840 → 0.2215) against a seed sd of **0.0009–0.0020**, and classification strict-or-accuracy **−0.1977**. Two earlier attempts were void. absneg4 (roles only) lost the gain (event_argument strict F1 **−0.1027** vs the treatment, recall 0.1481 → 0.0681). Unscoped negatives move arguments, and the classification cost did not separate when the pool was scoped to roles.

**3b-i** (merge coreferent triggers by shared arguments) is a failed pilot. Argument-score rows had pair AUC **0.468** on 355 positive and 443 hard-negative pairs, below chance, and the merge was **0 of 6** correct. Enough to stop that rule. Not a sample that can refute argument-overlap merging as a class. Instance states separate the same pairs at AUC **0.709** untrained, which is why TriggerLink reads instance states.

---

## 4. Closures that stand

- **Score swing.** Gold stays proposed and ranked first while `sigmoid(pair) ≥ 0.3` (logit ≥ −0.85) flips the metric. A separate event scorer, run 30k steps against two identical controls, is a real NULL: mean event_type strict F1 differs by **0.083** between the two controls, and the separated arm sits inside that spread. `task_lr` 4e-5, stopped at 12k inside the oscillation, is a real NULL on the swing in that window. The cause of the level swing is open. "Inside a 0.083 floor" means the experiment cannot see a smaller effect.
- **Gradients.** Across 48 batches at five checkpoints there is no stable opposition between the entity gradient and the event gradient. The event span gradient is **3–5×** smaller than the entity gradient and the binding gradient. That magnitude was not treated. `task_lr` was the step-size treatment, and it was null on the swing.
- **Issue 3** as written is not `_coref_link_loss`. The measured trigger-link gradient is about 7% of the entity gradient and |cos| ≤ 0.12. That refutes the review's snippet. It does not measure how often a coref batch has no positive pair.
- **Issue 5b's mechanism.** Absent trigger queries reach training batches (about 20 per document where the taxonomy allows). eb22d epoch 2 under the news55 menu fires almost no types that are absent from the gold of the sample. The under-firing number (event_type recall **0.08** at the picked threshold 0.1, 150 val documents, epoch 2 of 5, no floor) is a direction, not a closed delta. The in-run val cannot see it: its menu is each document's own gold, so event_type precision is pinned at 1.
- **Issue 2's measurement, left open on purpose.** At step 1,000 the anchor-row gradient is 0.43–0.71 of the role gradient, cosine **−0.71** on encoder layer 0 and **−0.88** on the junction. By step 33,172 it is small and roughly orthogonal (anchor-row NLL median **0.002**). The pre-registered harmlessness rule failed at both steps. Decode never reads that row (`decode_group` copies the instance span and continues). The A/B has not run. Section 7 pre-registers it.
- **English arguments are under-learned relative to CMNEE**, as a direction. On 64 training documents, 400 steps, argument strict F1 at threshold 0.3: CMNEE **0.162 → 0.697**, CASIE **0.018 → 0.132**, WikiEvents **0.051 → 0.158**. Single runs, no floor. The gap is large enough to trust the direction and not the third digit. Offset gold did not close it (single runs, mixed: WikiEvents argument strict F1 at 0.1 **+0.074**, at 0.3 even).
- **Junction column AUC** rose **0.58 → 0.74 / 0.77** under p5link. The argument strict F1 **+0.035** sits near the single-run **±0.02** floor. Quote the AUC.
- **S14.** Eval always passed the span threshold into record decode, so `record_anchor_threshold` was never read. Byte-identical output at 0.5 and 0.02 proved it. Forcing the record gate to win later moved event_argument strict F1 **+0.0579** at 0.1. That gain is a CMNEE result.

---

## 5. Floors that do not transfer

The catalog header still quotes event_argument seed sd **0.0009–0.0020**. That floor is from an earlier regime. On the eb22c separation window the three runs' mean argument strict F1 already spans **0.009–0.017**. An eb22c argument delta inside 0.01 is not a result, and one outside the old floor is not automatically outside the noise.

In-run event curves are about 80% MAVEN, and MAVEN has no argument gold. An in-run event_argument number cannot close an argument claim. Per-corpus `diagnose_checkpoint` reads can.

Cross-base argument strict F1 (incumbent **0.0991** at a val-picked 0.2, eb17 **0.1752**, eb19 **0.1816** at threshold 0.5) mixes operating points and, where the test or the labels moved, is not one series. Quote a pair only when the test and the threshold are the same.

Relation strict F1 carries a run-to-run floor of **±0.041**. Single-run deltas carry **±0.02**. event_type mean strict F1 on eb22c carries a two-seed spread of **0.083** over 20k–30k.

---

## 6. Supervision-gate rate

Measured 2026-10-25 with [tools/train/probe_supervision_gate.py](../train/probe_supervision_gate.py) on the epoch-2 checkpoint, step 33,172 of 82,930. The tool uses the trainer's own data path, train mode, dropout off, 60 seeded training documents per corpus. It refuses to report if its argument replica disagrees with `records.anchor_gate_stats`. The lines below were printed, so the replica matched. Output so far: `/Volumes/Development/tmp/issue5a/gate_epoch2.json`. Seven corpora are still running.

Scheduled injection at this step is **0.7794**, not 0.25. `0` is the model's own proposal, the setting at which the gate can fire with no gold forced in. `0.25` is the end of the anneal.

Shares are of gold event records and of their gold arguments. `anchor_not_seeded` was 0 in both corpora. Every drop is `anchor_not_proposed`.

| corpus | p | records dropped | arguments dropped |
|---|---:|---:|---:|
| casie_off | 0.78 | 3 / 389 (**0.8%**) | 5 / 1,125 (**0.4%**) |
| casie_off | 0.25 | 13 / 389 (**3.3%**) | 30 / 1,125 (**2.7%**) |
| casie_off | 0.00 | 14 / 389 (**3.6%**) | 32 / 1,125 (**2.8%**) |
| cc_news_events_haiku45 | 0.78 | 0 / 143 (**0.0%**) | 0 / 795 (**0.0%**) |
| cc_news_events_haiku45 | 0.25 | 2 / 143 (**1.4%**) | 27 / 795 (**3.4%**) |
| cc_news_events_haiku45 | 0.00 | 3 / 143 (**2.1%**) | 37 / 795 (**4.7%**) |

On these two corpora the gate can fire, which S13's instrument could not show, and the share is a few percent. That is a footnote, not the recall story, unless a later corpus is tens of percent. The remaining corpora are cc_news_events_sonnet55_v2, cc_news_haiku45, cmnee_ner_off, duee_ner, rams_merged, synthetic_events_capped, wikievents_off.

---

## 7. Issue 2 A/B, pre-registered, not launched

Optional. One epoch. Do not start it while eb22d is billing, and do not read it as an argument fix. The opposing gradient is a fact about the first few thousand steps. By step 33,172 the anchor row is already solved.

**Arm.** `eb22c-ddp-trace` plus `boundary_head.record_drop_anchor_field_loss: true` and `save_epoch_checkpoint: true`. Config already generated: [tools/train/config/base/eb22c-anchorfix-trace.yaml](../train/config/base/eb22c-anchorfix-trace.yaml). One change. Divisor of the field mean stays the full field count, so role weights stay exact. Flag off is bit-identical.

**Controls.** The three eb22c originals at matched steps: control, seed43, sep. Not eb22d. eb22d changed the marginal loss to bce and the coref distance to an MLP, so an arm on the eb22d recipe would not be this one change.

**Proof line**, after logging exists, from the record head: `record_drop_anchor_field_loss: N anchor rows dropped from the field loss`. N = 0 means the arm did not train. The line is emitted where the row is skipped.

**Reads, in this order.**

1. Early window, steps 1k–4k, where the measured cosine is negative. Per-corpus trigger and argument strict F1 at each source's own best threshold, on maven, cmnee, casie, duee, against the three controls at the same steps. The two-control event_type floor on mean best-threshold strict F1 is **0.03 / 0.07 / 0.06 / 0.16** (maven / cmnee / casie / duee). A delta inside that spread is not a result.
2. CMNEE argument strict F1 at its own best threshold. That source has range at 20k–30k (**0.30–0.36**). At one epoch it may not. If every arm is at the floor, say the read is unreadable. Do not substitute another metric and call it NULL.
3. CASIE argument strict F1 at 16k is **not a readout**. Every eb22c run is at the floor there (own-best ≤ 0.01 through 16k; 0.04–0.06 only at 20k–30k).

**Stop.** One epoch. Resume only if CMNEE argument strict F1, or the early-window trigger read, clears the three-control spread. A null or an unreadable argument read stops. Do not put the flag in a base on a mechanism-only win.

**Guard heads.** Entity and classification strict F1 on the same per-corpus read. A cost there counts even if arguments move.

---

## 8. Argument-miss split, before any reverse link

Do not build junction Phase 2 from the King of Sweden negative. First count, on eb22d's own val, the share of gold arguments that are a strict miss for each reason. Gold menu, threshold **0.1** (the epoch-2 in-run sweep pick), chunk 4096, overlap 0, global decode on. Strict micro F1 is `(event_type, role, entity, trigger_key)`. Relaxed micro F1 drops `trigger_key`.

Buckets, a partition of gold arguments:

| bucket | definition |
|---|---|
| strict hit | the full key matches a prediction |
| wrong trigger | `(type, role, entity)` matches, the trigger key does not. Split into sibling gold, coreferent trigger (the prediction's triggers are a non-empty proper subset of this gold event's triggers), boundary (containment), false trigger |
| coreferent surface | the entity string differs, and both strings sit in the same `coref_clusters` cluster on a prediction of the same type and role |
| never proposed | none of the above |

`coreferent surface` is reported only over documents that carry `coref_clusters`. A zero with no clusters in the sample is "the field is absent," not "argument coreference is not the miss." CASIE, CMNEE, and DuEE are the val sources with argument gold. WikiEvents and the news coref corpora are `train_only` in eb22d and are not in this val.

Section 8 is filled when [tools/train/probe_argument_misses.py](../train/probe_argument_misses.py) finishes. Output: `/Volumes/Development/tmp/issue5a/misses.json`.

The older taxonomy on a pre-record-head checkpoint was correct **44.5%**, never proposed **48.8%**, boundary **5.3%**, label **1.3%** of 18,557 gold arguments ([EVENT_ARGUMENT_DIAGNOSIS.md](EVENT_ARGUMENT_DIAGNOSIS.md) §1). That checkpoint pooled same-type events. This count is on eb22d, which trains `event_records`. The two numbers are not a delta.

---

## 9. What is not on this line

Relation follow-ups (dropout off alone, or keep-gold with the loss reweighted near 0.3) and the news classification prior are real and unrebutted. They are not boundary defects, and they are not the next argument experiment.

Issue 7 (about 1.7M parameters that never receive a gradient: the shared pool under `per_query`, and the latent / anchorless record branches) is real and informational. Removing those branches is cleanup. It is not an argument treatment, and it does not belong in the Issue 2 arm.
