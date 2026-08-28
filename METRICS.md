# Evaluation Metrics

How GLiNER2 scores a model during training and on a blind test set. All of this
lives in [`gliner2/training/metrics.py`](gliner2/training/metrics.py); the
trainer calls it once per evaluation pass and merges the returned dict into its
own metrics.

- [Quick start](#quick-start)
- [What gets scored](#what-gets-scored)
- [Strict vs relaxed](#strict-vs-relaxed)
- [Micro, macro, support](#micro-macro-support)
- [Per-category match semantics](#per-category-match-semantics)
- [The overall `event` metric](#the-overall-event-metric)
- [Fine-grained span error analysis](#fine-grained-span-error-analysis)
- [Returned keys](#returned-keys)
- [The classification report](#the-classification-report)
- [Worked example](#worked-example)
- [Driving best-checkpoint selection](#driving-best-checkpoint-selection)
- [Notes and edge cases](#notes-and-edge-cases)

---

## Quick start

```python
from gliner2.training import GLiNER2Trainer, TrainingConfig, make_compute_metrics

trainer = GLiNER2Trainer(
    model, TrainingConfig(...),
    compute_metrics=make_compute_metrics(batch_size=16, threshold=0.5),
)
```

Three entry points:

| Function | Use |
|---|---|
| `make_compute_metrics(batch_size, threshold)` | Returns a `(model, eval_dataset) -> dict` hook for the trainer. |
| `compute_metrics(model, eval_dataset, batch_size, threshold)` | Score a dataset directly; returns a flat `{metric_name: value}` dict. |
| `evaluate_checkpoint(checkpoint_dir, test_data, ...)` | Load a saved checkpoint and score a test set — the blind final-test pass, typically against `out/<run>/best`. |

`eval_dataset[i]` must yield `(text, gold_output_dict)`. The trainer's
`ExtractorDataset` already satisfies this.

### How a record is scored

For each gold record, `compute_metrics`:

1. **Reconstructs a schema** from the gold (`_schema_from_gold`) — the set of
   entity labels, relation names, classification tasks+labels, and event
   types+roles present. This schema is handed to the model, so evaluation is
   **closed-set against the gold's structure**: the model is told *which* labels
   and types to look for and is scored on whether it finds the right *surfaces*,
   not on discovering the label set from scratch.
2. Runs `model.batch_extract(texts, schemas, ...)` to get predictions.
3. Tallies per-label true positives / false positives / false negatives for
   every category, in both a **strict** and a **relaxed** regime.
4. Finalizes each into micro/macro precision, recall, F1, support, and a
   per-label text report.

A category absent from the eval set is silently omitted from the output.

---

## What gets scored

Seven categories, each derived from a gold field and a prediction block:

| Category | Scores | Gold field | Prediction block | Per-label key |
|---|---|---|---|---|
| `entity` | entity spans | `output.entities` `{label: [surface]}` | `entities` | label |
| `relation` | relation triples | `output.relations` `[{name: {head, tail}}]` | `relation_extraction` | relation name |
| `classification` | task labels | `output.classifications` `[{task, labels, true_label}]` | `<task>` key | task |
| `event_type` | event-type presence | `output.events[].event_type` | `event_extraction` keys | event type |
| `event_trigger` | trigger spans | `output.events[]` `{event_type, triggers}` | `event_extraction` | event type |
| `event_argument` | argument spans | `output.events[].arguments` `[{role, entity}]` | `event_extraction` | role |
| `event` | **combined** type + trigger + argument | (all of the above) | (all of the above) | namespaced |

The "per-label key" is the dimension along which results are broken out in the
per-label report and averaged for the macro score.

---

## Strict vs relaxed

Every category is reported twice:

- **strict** — exact match. Surfaces must be identical after trimming
  surrounding whitespace (case-sensitive).
- **relaxed** — the discrete *type/label* parts still match exactly, but the
  *surface* parts only need to **overlap**. Matching is one-to-one, and the
  matcher runs normalized-exact pairs first and overlap pairs second, so
  **relaxed can never score below strict** on the same data.

### The overlap rule

Two surfaces "overlap" (`_overlap`) when, after lowercasing and collapsing
whitespace, any of these hold:

- they are equal,
- one contains the other as a substring, or
- they share a token of length ≥ 2 that is not a stopword.

The stopword list (`the, a, an, of, in, on, at, to, for, and, or, by, with,
from, as, is, are, was, were`) keeps function words from creating spurious
matches — `"the president"` and `"the bombing"` do **not** overlap.

Examples that overlap: `New York City` ↔ `New York` (substring); `Bank of
America` ↔ `America` (shared content token); `USA` ↔ `usa` (normalized-equal).

### What the matcher is not

Four properties that are easy to assume and are not true here. They matter most
for events, where a mention carries several arguments.

**There is no IoU, Jaccard, or any overlap *ratio*.** `_overlap` is a boolean
predicate: equal, or substring either way, or **at least one** shared
non-stopword token of length ≥ 2. One token is enough, however long the two
surfaces are — `North Carolina` overlaps `Carolina Panthers stadium` on
`carolina`. There is no threshold to tune and no partial credit: a pair either
matches or it does not.

**Matching is on surface strings, not character offsets.** Nothing compares
span boundaries, so "span IoU" has no analogue in this implementation. Two
distinct mentions of the same string are identical to the matcher; for
arguments, the `trigger_key` in the strict key is what separates them.

**Scoring is per argument, never per event.** Each argument is one element of a
set, and TP/FP/FN accumulate over arguments. **An event is not required to have
all its arguments correct to earn credit** — a mention with four gold arguments
of which two are predicted correctly contributes 2 TP and 2 FN, not one failed
event. There is no event-level conjunction anywhere in the metric. The one
place a mention-level effect appears is strict scoring's `trigger_key`: get the
trigger set wrong and *every* argument of that mention misses under strict,
because the key no longer matches — but that is the key's doing, not an
all-or-nothing rule over arguments.

**Relaxed matching is greedy one-to-one, not optimal assignment.** Predictions
are walked in sorted order and each takes the first eligible unused gold item.
This is deterministic, but it is not a maximum matching: an early prediction can
consume a gold item that a later prediction would have matched better, leaving
the later one a false positive. Strict scoring is unaffected — it is plain set
intersection.

### Worked example: 4 gold arguments, 3 matched

> **"I matched 3 arguments but gold had 4 — what is the score, strict and
> relaxed?"**
>
> **TP 3, FP 0, FN 1 → precision 1.000, recall 0.750, F1 0.857 — and strict and
> relaxed are identical.**
>
> The fourth argument is a single false negative, not a failed event. Precision
> stays at 1.000 because the model predicted nothing wrong; it simply stopped
> early, and the metric charges that to recall alone. Strict and relaxed agree
> because relaxed only differs when a prediction is *nearly* right — here there
> is no fourth prediction for it to rescue.
>
> Had the model instead predicted a fourth argument and got it wrong, that is
> TP 3 / FP 1 / FN 1 → F1 0.750: **guessing wrong scores lower than staying
> silent**, on the same three correct arguments.

One event mention, four gold arguments. Every row below is the **contribution of
this one document** — micro F1 pools TP/FP/FN across the whole eval set before
computing P/R/F1, so a single event never has a score of its own. Numbers
produced by running the matchers, not by hand.

| what the model predicted | strict | relaxed |
|---|---|---|
| 3 of the 4, all exact, nothing extra | TP 3 / FP 0 / FN 1 → P 1.000, R 0.750, **F1 0.857** | identical |
| 4: three exact + one **wrong** entity | TP 3 / FP 1 / FN 1 → **F1 0.750** | identical |
| 4: three exact + one **near-miss** entity (`Carolina` for `North Carolina`) | **F1 0.750** | TP 4 → **F1 1.000** |
| all 4 arguments exact, but the **trigger** is wrong | TP 0 / FP 4 / FN 4 → **F1 0.000** | TP 4 → **F1 1.000** |

Three things to read off it:

- **Missing an argument costs recall only.** Predicting three of four and
  stopping gives precision 1.000 — silence is not punished, only the miss is.
- **Wrong and nearly-right are the same under strict and differ under relaxed.**
  That is the entire strict/relaxed distinction on a single row.
- **A wrong trigger zeroes every argument of that mention under strict**, because
  `trigger_key` is part of the strict key, while relaxed drops the trigger link
  and is untouched. A large strict-vs-relaxed gap on `event_argument` is usually
  a trigger problem, not an argument problem.

### Duplicates collapse

Both strict and relaxed operate on **sets**. Two gold arguments identical in
every scored component — same `(event_type, role, entity, trigger_key)` for
strict, same `(event_type, role, entity)` for relaxed — collapse to one element
and are scored once. A model that emits the same argument twice is not
penalised for the repeat, and gold that lists it twice does not get double
credit.

---

## Micro, macro, support

For a set of per-label counts (TP/FP/FN):

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)
```

(Any denominator of 0 yields 0.0.)

- **micro** — pool TP/FP/FN across *all* labels, then compute P/R/F1 once.
  Dominated by frequent labels; this is the headline number.
- **macro** — compute P/R/F1 per label, then take the **unweighted mean**.
  Every label counts equally, so rare labels matter as much as common ones.
- **support** — total gold count for the category, i.e. `Σ (TP + FN)`. This is
  the number of gold items the category was scored against.

---

## Per-category match semantics

- **Entity** — strict `(label, surface)`; relaxed = label exact + surface
  overlap. Aggregated per label.
- **Relation** — strict `(name, head, tail)`; relaxed = name exact + *both* head
  and tail surfaces overlap. Aggregated per relation name. The prediction parser
  accepts the engine's `(head, tail)` tuple shape and the
  `{"head": ..., "tail": ...}` dict shape (with or without spans/confidence).
- **Classification** — strict `(task, label)`; relaxed = task exact + label
  overlap. Multi-label gold/preds are unrolled and each `(task, label)` scored
  individually. Aggregated per task.
- **Event type** — `(event_type,)` presence. There is no surface to relax, so
  **strict == relaxed**.
- **Event trigger** — strict `(event_type, trigger)`; relaxed = event_type exact
  + trigger surface overlap (consistent with entity/relation relaxed). A
  mention's `triggers` list is flattened, so a two-trigger mention contributes
  two `(event_type, trigger)` pairs. Aggregated per event type.
- **Event argument** — strict `(event_type, role, entity, trigger_key)`, where
  `trigger_key` is the mention's full trigger set as a canonical sorted tuple
  (e.g. `("bombed",)`, or `("killed", "shot")` for a two-trigger mention);
  relaxed = `(event_type, role)` exact + entity overlap, dropping the trigger
  link entirely. Aggregated per role. `trigger_key` is part of the **strict**
  key so that identical `(type, role, entity)` arguments from different event
  mentions are distinguished — a consequence is that a wrong or incomplete
  predicted trigger set makes **all** of that mention's arguments miss under
  strict scoring (they still match under relaxed).

---

## The overall `event` metric

`event` is a single combined score over the three event sub-categories. Their
TP/FP/FN counters are **summed**, then finalized like any other category, so the
**micro** numbers are the pooled aggregate of event types, triggers, and
arguments.

Two deliberate design points:

1. **The event type is counted in all three buckets** at different
   granularities (as bare type presence, inside the trigger key, and inside each
   argument key). This is intentional — "combined type + trigger + arguments"
   means all three contribute.
2. **Per-label rows are namespaced** `type:`, `trigger:`, `arg:` before
   summing, so the combined report keeps `type:Attack` distinct from
   `trigger:Attack` and never silently merges them. Namespacing affects only the
   per-label/macro breakdown; the micro totals are unchanged.

---

## Fine-grained span error analysis

An **additive diagnostic** for **entities and event spans (triggers,
arguments)**, based on Ortmann (2022), [*Fine-Grained Error Analysis and Fair
Evaluation of Labeled Spans*](https://aclanthology.org/2022.lrec-1.150/). It
runs automatically whenever those categories are scored and leaves the
strict/relaxed regimes untouched.

**The problem it addresses.** Strict/relaxed **double-penalize a near-miss**: a
span with the right surface but the wrong label, or the right label but a
slightly-off boundary, becomes *both* a false positive (under the predicted
label) and a false negative (under the gold label) — two errors for one
almost-right annotation. The strict/relaxed matchers pair only within a label
(`_match_relaxed`), so a label confusion is invisible: it looks like a
hallucination plus an unrelated miss.

**What it does.** For each record it matches predicted vs gold spans **across
labels**, one-to-one, and tags each pairing with a single typed error. The
"label" is the entity type, the event type (for triggers), or the role (for
arguments); the "surface" is the span text:

| Type | Meaning |
|---|---|
| `COR` | correct — same label, same surface |
| `LE` | labeling error — same surface, wrong label |
| `BES` | boundary error, system **s**maller (pred surface ⊂ gold) |
| `BEL` | boundary error, system **l**arger (gold surface ⊂ pred) |
| `BEO` | boundary **o**verlap — shared content token, no containment |
| `LBE` | labeling+boundary error — wrong label *and* overlapping surface |
| `FP` / `FN` | pred / gold with no overlapping counterpart at all |

Matching is greedy in priority order `COR → LE → BE → LBE`, so every annotation
is counted once. The order is our adaptation; the result is deterministic given
the fixed order and sorted inputs. Relabeling an already-matched pair does not
move the fair score below (equal weights), but because the match is greedy, in
rare records where one prediction overlaps several gold spans the order can
change which pairs match and thus shift the counts slightly.

It also records **label confusions** — the `(gold_label → pred_label)` pairs
behind every `LE`/`LBE` — and prints them as a small table under the error
counts, so you can see *which* labels get swapped (e.g. `LOC → PER`, or a
`Target → Attacker` role swap) rather than just how often.

**Fair P/R/F1.** We use the **reference FairEval tool's default weights**, which
are the paper's Eq. 6/7 rather than its plainer Eq. 5. A near-miss is charged
once instead of twice, *and* a boundary error earns partial true-positive credit,
because the system did find the span:

```
LE  = 0.5 FP + 0.5 FN            right span, wrong label -- no credit
LBE = 0.5 FP + 0.5 FN            both wrong -- no credit
BES = 0.5 TP + 0.5 FN            system span smaller: a recall miss
BEL = 0.5 TP + 0.5 FP            system span larger: a precision miss
BEO = 0.5 TP + 0.25 FP + 0.25 FN

tp = COR + 0.5*(BES + BEL + BEO)
fp = FP + 0.5*(LE + LBE) + 0.5*BEL + 0.25*BEO
fn = FN + 0.5*(LE + LBE) + 0.5*BES + 0.25*BEO
fair_f1 = 2 * P * R / (P + R)
```

Unlike Eq. 5 these weights move **F1**, not just P and R, because they add TP
mass. Two consequences worth knowing:

- The `BES`/`BEL`/`BEO` split now *changes the score* (`BES` costs recall only,
  `BEL` precision only), where under Eq. 5 all three weighed the same. That makes
  the surface approximation below load-bearing rather than cosmetic.
- Fair is no longer strictly harsher than relaxed on boundary errors. Relaxed
  still credits any overlap as a full true positive; fair now credits half. A
  category whose only hits are boundary-off scores ~0.5 under fair rather than 0.

With no typed near-misses fair matches strict up to the normalization `COR`
applies — strict is case-sensitive, but `COR` (like `LE`) lowercases and
collapses whitespace first, so the two can still differ on case- or
whitespace-only surface variants (`Apple` vs `apple`).

Fair is emitted as a **selectable regime**: `eval_<cat>_fair_micro_{precision,
recall,f1}` and `eval_<cat>_fair_support`, for `<cat>` in `entity`,
`event_trigger`, `event_argument`. Any returned float can drive
`metric_for_best` (e.g. `eval_entity_fair_micro_f1`). The trainer's built-in
default remains `eval_loss` and the
[threshold sweep](#driving-best-checkpoint-selection) still optimizes strict, but
the event A/B and synthetic-fine-tune configs now **select on fair F1**.

Note `metric_for_best` no longer falls back when its key is absent — it raises.
The old fallback was `eval_loss`, which swapped both the quantity and its
direction, so a run configured to maximize an F1 maximized loss instead.

**Surface-approximation caveats.** Gold spans carry no character offsets, so the
positional boundary sub-types are approximated from substring containment:
`York` ⊂ `New York` reads as `BES` even if it is a different occurrence in the
text. Counts are over the record's distinct `(label, surface)` pairs, not
mentions — a surface repeated in the text collapses to one. This is the same
deduplicated surface set the strict/relaxed regimes already score.

---

## Returned keys

Plus the span error diagnostic, per span category `<cat>` in `entity`,
`event_trigger`, `event_argument` (when scored):
`eval_<cat>_error_{COR,LE,BES,BEL,BEO,LBE,FP,FN}` (integer counts),
`eval_<cat>_fair_micro_{precision,recall,f1}`, `eval_<cat>_fair_support`, and
`eval_<cat>_error_report` (a multi-line table, including the label-confusion
list). The `fair` keys follow the `<category>_<regime>` scheme below; the
`error_*` keys sit outside it.

`compute_metrics` returns a flat dict. For every present `<category>` and each
`<regime>` in `strict`, `relaxed`:

```
eval_<category>_<regime>_micro_precision
eval_<category>_<regime>_micro_recall
eval_<category>_<regime>_micro_f1
eval_<category>_<regime>_macro_precision
eval_<category>_<regime>_macro_recall
eval_<category>_<regime>_macro_f1
eval_<category>_<regime>_support
eval_<category>_<regime>_classification_report   # multi-line string
```

`<category>` ∈ `entity, relation, classification, event_type, event_trigger,
event_argument, event`. `<regime>` ∈ `strict, relaxed`.

Example: `eval_event_strict_micro_f1`, `eval_entity_relaxed_macro_precision`,
`eval_event_argument_strict_support`.

---

## The classification report

Each `eval_<category>_<regime>_classification_report` is a ready-to-print table:
one row per label (precision, recall, F1, per-label support), then `micro avg`
and `macro avg` rows. The avg rows show the category's total support. The
printed micro summary (one `P / R / F1  (strict -> relaxed)` line per category,
preceded by an `[eval]` header) is emitted on every eval pass. After the
blind-test pass, `tools/train/train.py` also prints the full per-label
classification reports followed by a second compact micro summary. With
`eval_by_language: true` in the `eval` config section this double-print runs once
per language in alphabetical order, then once over all data combined. Finally,
a `===== Blind test summary by language =====` recap prints just the compact
micro table for each language (labeled `[<lang>]`), plus a final `[all]` row
for the combined pass, so per-language results can be compared at a glance
without scrolling back through the detailed reports above.

---

## Worked example

Gold: one `Attack` event, trigger `bombed`, arguments `Attacker=rebels`,
`Target=base`, `Place=Aleppo`. Prediction: right event type, **wrong trigger
surface** `struck`, and one wrong argument entity `Place=Damascus`.

Micro `P / R / F1  (strict -> relaxed)`:

```
[eval] micro precision / recall / f1  (strict -> relaxed)
  event_type      P=1.0000->1.0000  R=1.0000->1.0000  F1=1.0000->1.0000
  event_trigger   P=0.0000->0.0000  R=0.0000->0.0000  F1=0.0000->0.0000
  event_argument  P=0.0000->0.6667  R=0.0000->0.6667  F1=0.0000->0.6667
  event           P=0.2000->0.6000  R=0.2000->0.6000  F1=0.2000->0.6000
```

Reading it:

- **event_type** is perfect — the type `Attack` was found (strict == relaxed).
- **event_trigger** strict is 0 (`struck` ≠ `bombed`); relaxed is also 0
  (`struck` and `bombed` share no content tokens or substring).
- **event_argument** strict is 0 because the wrong trigger poisons the strict
  argument key; relaxed is 2/3 (`rebels`, `base` match; `Damascus` ≠ `Aleppo`).
- **event** combines them. Strict: TP=1 (type) + 0 (trigger) + 0 (args) = 1,
  FP = 0 + 1 + 3 = 4, FN = 0 + 1 + 3 = 4, over **support 5** = 1 type + 1
  trigger + 3 args → P = R = 1/5 = 0.20. Relaxed: TP = 1 + 0 + 2 = 3 → 0.60.

The combined strict report (note the namespaced rows):

```
label                                     precision     recall         f1    support
----------------------------------------------------------------------------------
arg:Attacker                                 0.0000     0.0000     0.0000          1
arg:Place                                    0.0000     0.0000     0.0000          1
arg:Target                                   0.0000     0.0000     0.0000          1
trigger:Attack                               0.0000     0.0000     0.0000          1
type:Attack                                  1.0000     1.0000     1.0000          1
----------------------------------------------------------------------------------
micro avg                                    0.2000     0.2000     0.2000          5
macro avg                                    0.2000     0.2000     0.2000          5
```

---

## Driving best-checkpoint selection

The trainer selects the best checkpoint with `TrainingConfig.metric_for_best`
(default `eval_loss`, `greater_is_better=False`). To select on any metric above
instead, set the key and flip the direction:

```python
TrainingConfig(
    metric_for_best="eval_event_strict_micro_f1",
    greater_is_better=True,
    ...
)
```

All returned floats are eligible; `_classification_report` (a string) is not.

---

---

## References

The metric definitions on this page are conventional; where a choice follows a
specific source, it is cited here.

**Span matching and fine-grained error analysis**

- Ortmann, K. (2022). *Fine-Grained Error Analysis and Fair Evaluation of
  Labeled Spans.* LREC 2022, 1400–1407.
  <https://aclanthology.org/2022.lrec-1.150/> — the source of the error
  categories (`COR`, `SPU`, `MIS`, `BE*`, `LE`, `LBE`) used in the span error
  analysis, and of the **`fair` scoring regime**: this implementation uses the
  reference tool's default weights, which are the paper's Eq. 6/7 rather than
  the plainer Eq. 5, so a boundary error earns partial TP credit and therefore
  moves F1 rather than only P and R.
- FairEval, the reference implementation of the above:
  <https://github.com/katrinortmann/FairEval>
- Chinchor, N. (1992). *MUC-4 Evaluation Metrics.* Proceedings of the Fourth
  Message Understanding Conference, 22–29.
  <https://aclanthology.org/M92-1002/> — the origin of strict versus partial
  matching for information extraction, which is what this page calls strict
  versus relaxed.

**Precision, recall, F-measure and averaging**

- van Rijsbergen, C. J. (1979). *Information Retrieval*, 2nd ed.
  Butterworth-Heinemann — the F-measure.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to
  Information Retrieval*, Cambridge University Press, §8.3 — micro- versus
  macro-averaging, and why micro is dominated by frequent labels while macro
  weights every label equally.
- Opitz, J., & Burst, S. (2019). *Macro F1 and Macro F1.* arXiv:1911.03347.
  <https://arxiv.org/abs/1911.03347> — the two incompatible definitions of
  "macro F1" (mean of per-label F1 versus F1 of the mean P/R). This page uses
  the **mean of per-label F1**.
- Harbecke, D., Chen, Y., Hennig, L., & Alt, C. (2022). *Why only Micro-F1?
  Class Weighting of Measures for Relation Classification.* Proceedings of NLP
  Power! The First Workshop on Efficient Benchmarking in NLP, 32–41.
  <https://aclanthology.org/2022.nlppower-1.4/> — argues that reporting a single
  averaged measure hides where a model is strong and weak, and places micro and
  macro at the two ends of a class-weighting spectrum with intermediate schemes
  between them. Relevant here because this page reports micro **and** macro for
  every category rather than choosing one, and because the per-label breakdown
  is what makes the weighting choice inspectable.
- Sokolova, M., & Lapalme, G. (2009). *A systematic analysis of performance
  measures for classification tasks.* Information Processing & Management,
  45(4), 427–437. <https://doi.org/10.1016/j.ipm.2009.03.002>


## Notes and edge cases

- **Closed-set evaluation.** The schema is rebuilt from each gold record, so the
  model is scored on finding the right surfaces for *known* labels/types — not
  on discovering the label set. Event types that end up with no roles are
  dropped from the reconstructed schema (the schema needs ≥ 1 role).
- **Strict is case-sensitive**, surfaces trimmed of surrounding whitespace only.
  **Relaxed normalizes** (lowercase + whitespace-collapse) before comparing.
- **Prediction format drift is handled.** Surfaces may arrive as plain strings
  or as `{"text": ...}` dicts (under `include_spans`/`include_confidence`);
  relations as tuples or dicts; classifications as a string, a list, or
  `{"label": ...}`. All shapes are parsed.
- **Empty categories are omitted** — if neither gold nor pred has any item for a
  category in the whole eval set, none of its keys appear.
- **Order-independence.** Set/aggregate metrics like F1 don't depend on record
  order, so eval/test metrics are deterministic across epochs.
