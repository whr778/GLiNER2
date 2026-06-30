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
| `event_trigger` | trigger spans | `output.events[]` `{event_type, trigger}` | `event_extraction` | event type |
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
  **strict == relaxed**, and both also equal `event_trigger` *relaxed* (which
  likewise collapses a trigger down to its event type). This is the same number
  surfaced under its own name for convenience — not a bug.
- **Event trigger** — strict `(event_type, trigger)`; relaxed drops the exact
  trigger word and scores **event-type presence**. Aggregated per event type.
- **Event argument** — strict `(event_type, role, entity, trigger)`; relaxed =
  `(event_type, role)` exact + entity overlap, dropping the trigger link.
  Aggregated per role. The trigger is part of the **strict** key so that
  identical `(type, role, entity)` arguments from different event mentions are
  distinguished — a consequence is that a wrong predicted trigger makes **all**
  of that mention's arguments miss under strict scoring (they still match under
  relaxed).

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

## Returned keys

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
per language in alphabetical order, then once over all data combined.

---

## Worked example

Gold: one `Attack` event, trigger `bombed`, arguments `Attacker=rebels`,
`Target=base`, `Place=Aleppo`. Prediction: right event type, **wrong trigger
surface** `struck`, and one wrong argument entity `Place=Damascus`.

Micro `P / R / F1  (strict -> relaxed)`:

```
[eval] micro precision / recall / f1  (strict -> relaxed)
  event_type      P=1.0000->1.0000  R=1.0000->1.0000  F1=1.0000->1.0000
  event_trigger   P=0.0000->1.0000  R=0.0000->1.0000  F1=0.0000->1.0000
  event_argument  P=0.0000->0.6667  R=0.0000->0.6667  F1=0.0000->0.6667
  event           P=0.2000->0.8000  R=0.2000->0.8000  F1=0.2000->0.8000
```

Reading it:

- **event_type** is perfect — the type `Attack` was found (strict == relaxed).
- **event_trigger** strict is 0 (`struck` ≠ `bombed`) but relaxed is 1.0
  (type presence only).
- **event_argument** strict is 0 because the wrong trigger poisons the strict
  argument key; relaxed is 2/3 (`rebels`, `base` match; `Damascus` ≠ `Aleppo`).
- **event** combines them. Strict: TP=1 (type) + 0 (trigger) + 0 (args) = 1,
  FP = 0 + 1 + 3 = 4, FN = 0 + 1 + 3 = 4, over **support 5** = 1 type + 1
  trigger + 3 args → P = R = 1/5 = 0.20. Relaxed: TP = 1 + 1 + 2 = 4 → 0.80.

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
