# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cmnee_typed | train | 9,281 | 9,281 | 0 | 0 | 0 | 0 | 15,962 | 57,641 |
| cmnee_typed | val | 1,606 | 1,606 | 0 | 0 | 0 | 0 | 2,643 | 9,366 |
| cmnee_typed | test | 2,724 | 2,724 | 0 | 0 | 0 | 0 | 4,843 | 18,414 |

## cmnee_typed

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 9,281 | 9,281 | 302 | 395 | 487 | 0 | 0 | 0 | 0 | 15,962 | 57,641 |
| val | 1,606 | 1,606 | 300 | 391 | 490 | 0 | 0 | 0 | 0 | 2,643 | 9,366 |
| test | 2,724 | 2,724 | 289 | 390 | 473 | 0 | 0 | 0 | 0 | 4,843 | 18,414 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| zho | 9,281 | 1,606 | 2,723 | 13,610 | 99.99% |
| und | 0 | 0 | 1 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 2,724 |
| test_unique_inputs | 2,724 |
| train_records | 9,281 |
| train_unique_inputs | 9,281 |
| val_records | 1,606 |
| val_unique_inputs | 1,606 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| event types (per trigger mention) | train / val | 0.1840 WARN | 19,422 / 3,236 | Support 29%→18%, Experiment 39%→49%, Deploy 13%→21%, Injure 2%→0%, Accident 4%→2% |
| event types (per trigger mention) | train / test | 0.3107 STRONG | 19,422 / 6,553 | Support 29%→10%, Experiment 39%→58%, Deploy 13%→25%, Exhibit 8%→2%, Accident 4%→2% |
| event types (per trigger mention) | val / test | 0.1298 | 3,236 / 6,553 | Experiment 49%→58%, Support 18%→10%, Exhibit 7%→2%, Deploy 21%→25%, Accident 2%→2% |
| event roles (per argument) | train / val | 0.0942 | 62,573 / 10,270 | Object 10%→6%, Equipment 17%→20%, Materials 9%→6%, Militaryforce 4%→8%, Location 9%→11% |
| event roles (per argument) | train / test | 0.1463 | 62,573 / 20,813 | Object 10%→3%, Materials 9%→3%, Equipment 17%→22%, Militaryforce 4%→9%, Date 14%→17% |
| event roles (per argument) | val / test | 0.0624 | 10,270 / 20,813 | Date 14%→17%, Materials 6%→3%, Object 6%→3%, Equipment 20%→22%, Militaryforce 8%→9% |


### Events

**train.** 19,422 instances in 9,281 records, 0 triggerless. Trigger gold 15,962. Argument gold 57,641. Arguments per instance, mean 3.22. Trigger length in whitespace tokens, mean 2.08, single-token share 0.1%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Experiment | 7,638 | 39.33% |
| Support | 5,632 | 29.00% |
| Deploy | 2,454 | 12.64% |
| Exhibit | 1,544 | 7.95% |
| Accident | 703 | 3.62% |
| Manoeuvre | 627 | 3.23% |
| Injure | 438 | 2.26% |
| Conflict | 386 | 1.99% |

**val.** 3,236 instances in 1,606 records, 0 triggerless. Trigger gold 2,643. Argument gold 9,366. Arguments per instance, mean 3.17. Trigger length in whitespace tokens, mean 2.05, single-token share 0.2%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Experiment | 1,585 | 48.98% |
| Deploy | 692 | 21.38% |
| Support | 575 | 17.77% |
| Exhibit | 217 | 6.71% |
| Accident | 66 | 2.04% |
| Manoeuvre | 58 | 1.79% |
| Conflict | 28 | 0.87% |
| Injure | 15 | 0.46% |

**test.** 6,553 instances in 2,724 records, 0 triggerless. Trigger gold 4,843. Argument gold 18,414. Arguments per instance, mean 3.18. Trigger length in whitespace tokens, mean 2.01, single-token share 0.1%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Experiment | 3,795 | 57.91% |
| Deploy | 1,646 | 25.12% |
| Support | 631 | 9.63% |
| Exhibit | 136 | 2.08% |
| Manoeuvre | 122 | 1.86% |
| Accident | 120 | 1.83% |
| Conflict | 63 | 0.96% |
| Injure | 40 | 0.61% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 62,573 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 10,270 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 20,813 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2.19 | 51.28% | 1.84 | 36.04% | 0.02% | 0 | 62,573 | 0 | 19,422 |
| val | 2.19 | 51.24% | 1.86 | 37.11% | 0.04% | 0 | 10,270 | 0 | 3,236 |
| test | 2.78 | 61.04% | 1.94 | 37.69% | 0.07% | 0 | 20,813 | 0 | 6,553 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 19,422 | 0.00% | 8,651 | 76.50% | 75.56% | 7.24% | 2.33% | 32.23% |
| val | 3,236 | 0.00% | 1,591 | 81.84% | 81.14% | 8.49% | 2.33% | 31.76% |
| test | 6,553 | 0.00% | 4,344 | 79.77% | 79.12% | 7.83% | 0.94% | 51.40% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 8,651 |
| val | 0 | 0 | 1,591 |
| test | 0 | 0 | 4,344 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

### Entities

None.

### Relations

None.

### Classifications

None.

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | unknown |
| relations | unknown |
| events | unknown |
| classifications | unknown |
| structures | unknown |
