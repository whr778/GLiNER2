# eb22c: the event "collapse" is a score-level swing

Working paper, 2026-10-20 to 2026-10-21. Status: **mechanism located to the score LEVEL; gradient conflict between entity and event supervision
MEASURED and NOT FOUND (2026-10-21); cause still open.** Rows in [[EXPERIMENT_CATALOG]] (2026-10-20/21, Events line);
TODO rows 41-44.

## 1. What was believed, and what is true

Believed (2026-10-19/20): eb22c's events learn in epoch 1 and **collapse to 0.0** after the LR peak; focal loss,
dropped data, DDP drift, the schedule and the negative load were each suspected in turn.

Measured on the autopsy run (eb22c-ddp-trace = eb22c-bce-trace + eval/checkpoint every 500, 1x H100, 30k steps,
27 checkpoints on HF `whr778/gliner2-eb22c-collapse-trace`):

- The gold trigger is **always proposed** (reach 0.96-1.00) and **ranked first** (median rank 0-1) at every
  checkpoint, "collapsed" ones included.
- What moves is the **level** of every candidate's score. An anchor is emitted only when
  `sigmoid(pair_logit / record_temperature) >= 0.3`, i.e. logit >= -0.85; the level swings across that line.
- In-run event_type strict F1 swings 0.00-0.96 to ~15k steps, then 0.18-0.73 to 30k (30k: 0.73). **The swing
  narrows but does not stop**: gold trigger logit median CMNEE -4.33 at 20k vs +1.56 at 30k, DuEE -8.32 vs -3.22.
- A "collapsed" checkpoint is **mis-thresholded, not unlearned**: checkpoint-5000 (in-run event_type 0.004)
  sweeps to event_type strict F1 1.00 (CMNEE) / 0.83 (DuEE) at threshold 0.001.

Every earlier "collapse" verdict (eb21-5ep at 9k, eb22c-bce-fixed at 7k, the office bisect arms at ~6k) was read
inside this swing. The DDP gradient-averaging defect (fixed 0b21629, upstream PR #199) and the offset-gold
validator defect (fixed 181d300) were real and are fixed; neither caused the swing.

## 2. Where the in-run numbers come from

With `sliding_window: true` the eval is a **seeded random draw** of `max_eval_samples` (1500, seed 42), not a
prefix. Rebuilt and matched to the logged support (entity gold 6,857 vs 6,878 logged):

| head | composition of the in-run eval |
|---|---|
| entity | chfinann 38.3%, docee 24.5%, turkish_event 22.3%, docfee 8.5% (93% in four corpora) |
| events | **MAVEN 617 of 778 gold events**, casie 80, cmnee 48, mendeley_ed 17, duee 16 |

So the in-run event_type curve is mostly MAVEN, and the probes below (CMNEE, DuEE, CASIE) measure the same
phenomenon on other corpora, not the curve itself.

## 3. One head, not separate mechanisms

Entity labels, the event trigger and every argument role are **rows of one pair-score tensor** (a CASIE document:
12 queries x 384 candidates, task_type `entities` / `events`), scored by one `SparseBoundaryPairScorer` on one
encoder. Shared by all query types: the encoder, the start/end marginal heads, the proposer, the pair scorer, and
the pointwise pair loss plus the proposal and rerank listwise losses taken over all queries. The only per-query
difference is the label text that becomes the query state.

The pair logit is a sum of seven parts (`scoring.py`): `compat + start + end + prior + content + inside + length`,
where `prior` is the proposer's query-gated bilinear start.end compat (`proposal.py` ~705, no bias).

## 4. Measurements, in order

| check | result |
|---|---|
| decomposition, gold trigger, paired per-event | the **prior** is the largest single mover of the gold logit in every swing (4k->4.5k CMNEE -2.44 of which prior -2.23; 20k->30k DuEE +4.52, prior +2.15); the candidate level shares the move with the marginals and length; the late gain is about half gold-vs-level margin |
| hypothesis: the prior's level is unanchored by its listwise loss, dense entity gold anchors entities | **DISPROVED** -- entity gold logits swing 1.1-1.7 logits per 500 steps (triggers 1.8-2.7), and the prior is a minor part of the entity moves |
| entity vs trigger level change, same documents (casie_off) + biored/scierc vs cmnee/duee | **opposite** sign 7 of 7 steps 3.5k-7k (just after the LR peak); **same** sign 6 of 6 steps 7k-26k; opposite again 26k-30k (triggers +3.43, entities -0.49) |
| why entity F1 is smooth | chfinann gold entities sit ~2.4 logits ABOVE the gate (median +1.5/+1.6/+3.9 at 4k/4.5k/30k, rank 0), so a 1.5-logit swing rarely flips them; docfee does cross; CMNEE triggers sat AT the gate (-1.1 vs -0.85) |
| training log | pooled losses fall smoothly through the swing; pointwise pair_loss median ~0.005 against listwise ~3.5; warmup ends at step ~4k |
| record object loss (would push absent types' trigger scores down) | never fires: 0 of 725 logged steps |

Reading: a **common-mode** swing moves every query type by 1-2.6 logits, plus a **trigger-specific** component that
dominates right after the LR peak and again at 28k-30k.

## 5. Links between query types (code trace)

- **Trigger <-> argument (record head): a real loss link.** It scores each trigger instance against each role's
  candidates using the role's own query state (`records.py` ~819), so the binding loss trains the ROLE queries'
  representations through the encoder.
- **Argument <-> entity: no loss link in the record code.** Role and entity queries are siblings in the tensor.
- **Trigger x trigger coreference link (`TriggerLink`, weight 0.3): no direct path to the query weights.** It reads
  only the trigger candidates' states (`candidate_encoder` over encoder boundary states, `model.py` 670-675), so its
  gradient reaches the link, the `candidate_encoder`, the shared `boundary_encoder` and the encoder -- never the
  query states, pair scorer, proposer or query head.

## 6. Gradient conflict: measured, not found

`tools/train/gradient_conflict.py`: one training-path forward per batch (real trainer preparation, auto negatives,
collator, schedules at the checkpoint's step, dropout off, bf16 autocast on CUDA), then separate gradients for
`entity` / `event` span losses (exact `reduce_by_task` shares), `bind` (record loss without the link), `tlink`
(the trigger link, captured per group), `relation`, `elink` (entity mention link) and `rest`; per-group cosines
(each encoder layer, each head module). Gates that can fail: shares must sum to each term; weighted shares must
rebuild the span total; the link term must equal record(w=0.3) - record(w=0) on replay (mutation-checked: refuses
0.019431 against the real 0.005829).

It cannot run on the laptop: one training forward at batch 4 needs ~44 GB in fp32 (the H100 run logged
`peak_alloc_gb=22.0` in bf16); two such jobs swapped a 32 GB machine on 2026-10-21. Running on 1x H100 for
checkpoints 3,500 / 4,000 / 4,500 / 22,000 / 24,000, 48 batches x 4 each.

**Result (1x H100, 2026-10-21, 15/15 link replays exact):**

| group | cos(entity, event), steps 3.5k / 4k / 4.5k / 22k / 24k |
|---|---|
| encoder layers | agree or ~0 everywhere; rises with depth early (4.5k: +0.06 -> +0.62); ~0 late, -0.06 at layers 16-19 at 24k |
| pair scorer | +0.98 / +0.95 / +0.99 / +0.24 / -0.09 |
| proposer | +0.43 / -0.19 / +0.37 / +0.24 / -0.42 (sign flips, no stable opposition) |
| null projection | -0.90 / -0.33 / +0.98 / -0.98 / +0.93 (sign flips; tiny module) |

| encoder gradient L2 | entity | event | bind | relation | tlink |
|---|---|---|---|---|---|
| 3.5k | 57 | 16 | 51 | 10 | 4.3 |
| 4k | 51 | 14 | 50 | 67 | 3.5 |
| 4.5k | 39 | 13 | 43 | 62 | 3.7 |
| 22k | 38 | 13 | 24 | 21 | 3.2 |
| 24k | 37 | 13 | 35 | 21 | 4.2 |

Reading: entity and trigger supervision do NOT compete for the shared weights. The event span gradient is a small
passenger (3-5x smaller than entity, smaller than binding, and dwarfed by relation at 4k-4.5k), so the event score
level moves with whatever the larger gradients do to the shared weights. The trigger link is ~7% of the entity
gradient and orthogonal to it: no treatment. Separating the query weights would remove an opposition that is not
there. Caveat: 48 batches x 4 per checkpoint, summed; a level shift can ride on a small component a cosine of the
whole does not isolate.

Decision it fed (the rules written before the result):
- conflict in the HEAD (proposer, null projection) -> a per-query-type bias/temperature on the pair score first,
  then separate pair scorer + proposer per query type;
- conflict in the ENCODER, or mostly common-mode -> the learning rate (task_lr 4e-5), not separation;
- `tlink` opposing `entity` in the layers that matter -> a training A/B at `record_coref_link_weight` 0.3 vs 0.0.

## 7. Consequences already standing

- **Checkpoint selection under the swing is partly luck**; per-epoch eval reads one sample of a range.
- **Option 4 (arguments injected as entity spans) is narrower than its catalog verdict reads**: run 1 measured
  "positive-only arg-as-entity supervision through a SHARED head, with no negatives reachable" (the derived corpus
  was in no negative pool); the re-run was VOID. A clean re-test belongs after the head design is settled.

## 8. Corrections made along the way (kept, because they are the method)

- "Stabilises at 0.5-0.73 by 30k" -> the swing narrows, it does not stop.
- "Pair-scored heads swing, the span-scored entity head is smooth" -> entities use the same scorer.
- "The prior is the largest mover of the level" -> of the GOLD logit; the level move is shared.
- "Entities and triggers move together from 7k" -> until 26k; opposite again 26k-30k.
- A footprint watchdog read RSS, which excludes compressed memory on macOS, and could not see an 80 GB process.
