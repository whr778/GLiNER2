# Coreferent triggers: every mention of an event owns its arguments

**Status:** SPEC 2026-10-06, not built. Priority: before the English v2 annotations, or with them
(`ENGLISH_ANNOTATION_SPEC.md`). Those annotations list every coreferent trigger per event.

## 1. What happens today (traced, not read)

One event, two coreferent triggers ("Rebels **attacked** the base ... The **assault** killed four
soldiers"), was run through the real training collator and the record loss.

| stage | behaviour | where |
|---|---|---|
| Record format | **Supported.** `triggers` is a list (since 4436653). The record's trigger field holds ONE value whose span alternatives are both mentions: `[(1,2), (8,9)]`. | `processor.py:1365` |
| Instance pool | Each mention is its own anchor candidate, so "attacked" and "assault" are both instance hypotheses. | `forward_group` |
| Gold ownership | **First mention only.** `compute_group_loss` resolves `values[0]`, takes `cols[0]`, and seeds the gold instance on "attacked". The record's role targets train that one row. | `records.py` `compute_group_loss` |
| False triggers | **Protected.** `_negative_instances` excludes any instance overlapping ANY alternative, so "assault" is never trained as a false trigger. The same holds for the junction column loss's hard negatives. | `records.py` `_negative_instances` |
| "assault"'s own instance | **Untrained either way.** It is neither gold (no argument targets) nor false. | — |
| Decode | Both instances can clear the anchor gate, so the same event is emitted TWICE. Each argument goes to the stronger one (exclusive allocation), so the arguments are split or missing on the duplicate. | `records.decode_group` |

## 2. How much this matters (measured)

In eb19's training data, **0.8% of 151,758 events** have more than one trigger, all from
mendeley_ed (80% of its events, up to 12 triggers). Those are keyword SETS that jointly signal one
event ("remote", "anxieties", "rising"), not coreferent mentions.

Every other corpus, including MAVEN at 78k events, has exactly one trigger per event. So the gap
is **dormant today**. It goes live with:
- the English v2 annotations (every coreferent trigger listed);
- ACE 2005, once its converter merges `event_mention`s (TODO #27).

## 3. Design

Two halves, because ownership is a TRAINING question and duplicates are a DECODE question.

### 3a. Training: every mention owns the record (opt-in `record_coreferent_ownership: true`)

- **Role targets.** Resolve EVERY alternative in `values[0]` to its seeded instance, not just
  `cols[0]`. Each mention's row gets the record's argument targets, so the "attacked" row and the
  "assault" row both learn that "Rebels" is the Attacker.
- **The junction column loss already expects this.** `JUNCTION_LAYER_SPEC.md` section 4 defines
  G_j as "the instance(s) seeded by its gold trigger -- every alternative". Today G_j holds one
  instance. With this change, the column softmax puts mass on ANY mention of the right event,
  against the false triggers.
- **Loss normalisation.** A record with m mentions must not weigh m times one with a single
  trigger. Average the role loss over the record's mentions, its own mean, as `absent_reduction`
  established for every term since.
- **Keyword-set triggers (mendeley_ed) are the same mechanics.** Every keyword owning the record is
  arguably right there too. Its 1,139 events become the first real data the flag touches, so they
  are the trace target.

### 3b. Decode: one event per cluster, not one per mention

Training every mention to own the arguments makes the duplicates SHARPER at decode: both mentions
now fire with the same arguments. So 3a must not ship without 3b. Three options:

| option | how | cost | risk |
|---|---|---|---|
| **i. Argument-overlap merge** | merge same-type selected instances whose decoded arguments overlap (e.g. at least one identical (role, entity)), unioning triggers and arguments | decode-only, no training | Decode-time POOLING of boundary variants was measured NEGATIVE on 2026-10-04 (F1 -0.001 to -0.004; merges dropped correctly keyed arguments). Argument overlap is a stronger signal than boundary adjacency, but it must be A/B'd, not assumed. |
| **ii. Learned trigger-trigger link** | a second junction, instance x instance with zero init, trained on gold clusters (same record = positive, different records = negative); at decode, cluster instances by it | new parameters plus a loss term | needs enough cluster data. Today there is ~none (section 2), so this waits for v2 / ACE. |
| **iii. Keep duplicates, score at cluster level** | change the event metrics to match a predicted event to a gold CLUSTER by any mention | eval-only | hides the problem rather than fixing it, and the output still carries duplicates |

Recommendation: **i first**. It is measurable today on mendeley_ed and on the v2 pilot gold.
**ii** comes once v2 provides clusters at volume.

The eval must also learn clusters whichever is chosen. Today a gold event with 3 triggers is
scored per trigger key, so finding the same event through a different mention counts as a miss.
That is a metrics change in `eval_metrics.py`, specified with 3b, and it needs its own
before/after on an unchanged checkpoint.

## 4. Gates (each must be able to fail)

1. **Off is bit-identical.** With the flag off, the losses on real eb19 batches are identical to
   today's, as every opt-in record option has been.
2. **Ownership reaches every mention.** On the traced two-trigger event, both mention rows carry
   the record's role targets and both are in the column loss's G_j. Measured from the batch, not
   from the code.
3. **No double weight.** A record's role-loss contribution is the same with 1 or 3 mentions,
   given identical scores.
4. **The decode merge does not eat distinct events.** Two different same-type events with
   disjoint arguments stay two events.
5. **The metric can fail.** A prediction found through a non-first mention scores as a hit under
   cluster matching and as a miss under today's trigger key. Show both on one constructed doc
   before trusting either.

## 5. Build order (trace, code, trace, test)

1. Trace the two-trigger event and a real mendeley_ed batch, with today's ownership, G_j, the
   decode output (duplicates?), and the per-trigger score.
2. Code 3a behind the flag, then gates 1-3.
3. Code 3b(i) as a decode option, plus the cluster-aware event metric. Then gates 4-5.
4. Trace again on the same inputs.
5. Tests, mutation-checked.
6. Fast A/B (warm from eb19, p5link design), arms:
   - control;
   - 3a;
   - 3a+3b(i).

   Score on the v2 pilot gold (coreferent clusters) and mendeley_ed val. Floors are measured by
   the control pair.

## 6. Data that would make ii worth building

Human English coreference clusters, rather than LLM ones:
- **ACE 2005**: event coreference, in the user's office environment (TODO #27);
- **MAVEN-ERE**: MAVEN's own documents with event coreference, temporal, causal and subevent
  relations. Unverified here (licence and format not yet checked), but it would pair with MAVEN,
  which we already train on, and is worth checking first.
