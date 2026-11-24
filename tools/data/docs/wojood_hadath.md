# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wojood_hadath | train | 1,096 | 1,096 | 0 | 0 | 0 | 0 | 1,197 | 1,533 |
| wojood_hadath | val | 137 | 137 | 0 | 0 | 0 | 0 | 138 | 174 |
| wojood_hadath | test | 533 | 533 | 0 | 0 | 0 | 0 | 598 | 770 |

## wojood_hadath

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,096 | 1,096 | 20 | 35 | 242 | 0 | 0 | 0 | 0 | 1,197 | 1,533 |
| val | 137 | 137 | 19 | 30 | 84 | 0 | 0 | 0 | 0 | 138 | 174 |
| test | 533 | 533 | 19 | 34 | 179 | 0 | 0 | 0 | 0 | 598 | 770 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| ara | 1,096 | 137 | 533 | 1,766 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 533 |
| test_unique_inputs | 533 |
| train_records | 1,096 |
| train_unique_inputs | 1,096 |
| val_records | 137 |
| val_unique_inputs | 137 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| event roles (per argument) | train / val | 0.0565 | 1,533 / 174 | hasLocation 32%→37%, hasAgent 16%→12%, hasDate 52%→51% |
| event roles (per argument) | train / test | 0.0206 | 1,533 / 770 | hasLocation 32%→34%, hasAgent 16%→14%, hasDate 52%→52% |
| event roles (per argument) | val / test | 0.0359 | 174 / 770 | hasLocation 37%→34%, hasAgent 12%→14%, hasDate 51%→52% |


### Events

**train.** 1,197 instances in 1,096 records, 1,197 triggerless. Trigger gold 1,197. Argument gold 1,533. Arguments per instance, mean 1.28. Trigger length in whitespace tokens, mean 0.00, single-token share 0.0%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Event | 0 | n/a |

**val.** 138 instances in 137 records, 138 triggerless. Trigger gold 138. Argument gold 174. Arguments per instance, mean 1.26. Trigger length in whitespace tokens, mean 0.00, single-token share 0.0%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Event | 0 | n/a |

**test.** 598 instances in 533 records, 598 triggerless. Trigger gold 598. Argument gold 770. Arguments per instance, mean 1.29. Trigger length in whitespace tokens, mean 0.00, single-token share 0.0%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Event | 0 | n/a |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,145 | 1,533 | 74.69% | 1 | 1,144 | 4 | 0 | 384 | 0 |
| val | 129 | 174 | 74.14% | 1 | 128 | 0 | 0 | 45 | 0 |
| test | 540 | 770 | 70.13% | 1 | 539 | 0 | 0 | 230 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

**train by role.**

| role | inside | placed | inside share | unmeasured |
| --- | --- | --- | --- | --- |
| hasDate | 551 | 802 | 68.70% | 0 |
| hasLocation | 427 | 486 | 87.86% | 0 |
| hasAgent | 167 | 245 | 68.16% | 0 |

**val by role.**

| role | inside | placed | inside share | unmeasured |
| --- | --- | --- | --- | --- |
| hasDate | 58 | 88 | 65.91% | 0 |
| hasLocation | 57 | 65 | 87.69% | 0 |
| hasAgent | 14 | 21 | 66.67% | 0 |

**test by role.**

| role | inside | placed | inside share | unmeasured |
| --- | --- | --- | --- | --- |
| hasDate | 257 | 400 | 64.25% | 0 |
| hasLocation | 219 | 260 | 84.23% | 0 |
| hasAgent | 64 | 110 | 58.18% | 0 |

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.00 | 0.00% | 1.05 | 5.15% | 0.00% | 1,533 | 0 | 1,197 | 0 |
| val | 1.00 | 0.00% | 1.06 | 6.32% | 0.00% | 174 | 0 | 138 | 0 |
| test | 1.00 | 0.00% | 1.04 | 4.16% | 0.00% | 770 | 0 | 598 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,197 | 0.00% | 106 | 24.53% | 24.53% | 24.53% | 24.53% | 2.37% |
| val | 138 | 0.00% | 1 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| test | 598 | 0.00% | 70 | 34.29% | 34.29% | 34.29% | 32.86% | 4.50% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 106 |
| val | 0 | 0 | 1 |
| test | 0 | 0 | 70 |

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
| entities | human |
| relations | human |
| events | human |
| classifications | human |
| structures | human |
