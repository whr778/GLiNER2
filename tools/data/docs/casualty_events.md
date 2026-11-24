# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| casualty_events | train | 23,627 | 23,627 | 0 | 0 | 0 | 0 | 54,028 | 124,561 |
| casualty_events | val | 2,666 | 2,666 | 0 | 0 | 0 | 0 | 6,049 | 13,878 |
| casualty_events | test | 2,852 | 2,852 | 0 | 0 | 0 | 0 | 6,409 | 14,614 |

## casualty_events

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 23,627 | 23,627 | 186 | 291 | 358 | 0 | 0 | 0 | 0 | 54,028 | 124,561 |
| val | 2,666 | 2,666 | 194 | 291 | 346 | 0 | 0 | 0 | 0 | 6,049 | 13,878 |
| test | 2,852 | 2,852 | 186 | 289 | 345 | 0 | 0 | 0 | 0 | 6,409 | 14,614 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 23,627 | 2,666 | 2,852 | 29,145 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 2,852 |
| test_unique_inputs | 2,852 |
| train_records | 23,627 |
| train_unique_inputs | 23,627 |
| val_records | 2,666 |
| val_unique_inputs | 2,666 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| event types (per trigger mention) | train / val | 0.2604 WARN | 57,726 / 6,414 | Mine Collapses 16%→0%, Air Crash 12%→23%, Train Collisions 10%→21%, Fire 14%→9%, Gas Explosion 12%→8% |
| event types (per trigger mention) | train / test | 0.1766 WARN | 57,726 / 6,828 | Armed Conflict 13%→28%, Shipwreck 12%→4%, Mine Collapses 16%→12%, Fire 14%→10%, Gas Explosion 12%→14% |
| event types (per trigger mention) | val / test | 0.3476 STRONG | 6,414 / 6,828 | Armed Conflict 13%→28%, Mine Collapses 0%→12%, Air Crash 23%→11%, Train Collisions 21%→11%, Shipwreck 11%→4% |
| event roles (per argument) | train / val | 0.0111 | 124,561 / 13,878 | missing 18%→19%, dead 18%→17%, injured 21%→21%, location 43%→43% |
| event roles (per argument) | train / test | 0.0292 | 124,561 / 14,614 | location 43%→40%, dead 18%→19%, missing 18%→19%, injured 21%→21% |
| event roles (per argument) | val / test | 0.0312 | 13,878 / 14,614 | location 43%→40%, dead 17%→19%, missing 19%→19%, injured 21%→21% |


### Events

**train.** 57,726 instances in 23,627 records, 0 triggerless. Trigger gold 54,028. Argument gold 124,561. Arguments per instance, mean 2.16. Trigger length in whitespace tokens, mean 1.04, single-token share 95.9%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Mine Collapses | 9,301 | 16.11% |
| Fire | 7,847 | 13.59% |
| Armed Conflict | 7,313 | 12.67% |
| Road Crash | 6,975 | 12.08% |
| Shipwreck | 6,935 | 12.01% |
| Air Crash | 6,841 | 11.85% |
| Gas Explosion | 6,764 | 11.72% |
| Train Collisions | 5,750 | 9.96% |

**val.** 6,414 instances in 2,666 records, 0 triggerless. Trigger gold 6,049. Argument gold 13,878. Arguments per instance, mean 2.16. Trigger length in whitespace tokens, mean 1.12, single-token share 87.7%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Air Crash | 1,495 | 23.31% |
| Train Collisions | 1,368 | 21.33% |
| Road Crash | 969 | 15.11% |
| Armed Conflict | 825 | 12.86% |
| Shipwreck | 702 | 10.94% |
| Fire | 563 | 8.78% |
| Gas Explosion | 492 | 7.67% |

**test.** 6,828 instances in 2,852 records, 0 triggerless. Trigger gold 6,409. Argument gold 14,614. Arguments per instance, mean 2.14. Trigger length in whitespace tokens, mean 1.06, single-token share 94.2%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 1,893 | 27.72% |
| Gas Explosion | 941 | 13.78% |
| Mine Collapses | 852 | 12.48% |
| Air Crash | 746 | 10.93% |
| Train Collisions | 717 | 10.50% |
| Fire | 689 | 10.09% |
| Road Crash | 689 | 10.09% |
| Shipwreck | 301 | 4.41% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 124,561 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 13,878 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 14,614 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.23 | 19.85% | 1.02 | 1.98% | 0.57% | 0 | 124,561 | 0 | 57,726 |
| val | 1.25 | 20.28% | 1.01 | 0.86% | 1.18% | 0 | 13,878 | 0 | 6,414 |
| test | 1.23 | 19.39% | 1.01 | 1.39% | 0.27% | 0 | 14,614 | 0 | 6,828 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 57,726 | 0.00% | 6,819 | 0.32% | 0.32% | 0.09% | 0.03% | 0.09% |
| val | 6,414 | 0.00% | 778 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| test | 6,828 | 0.00% | 763 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 6,819 |
| val | 0 | 0 | 778 |
| test | 0 | 0 | 763 |

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
| entities | llm_real |
| relations | llm_real |
| events | llm_real |
| classifications | llm_real |
| structures | llm_real |
