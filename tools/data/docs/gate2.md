# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate2 | train | 11,781 | 11,781 | 0 | 0 | 23,562 | 0 | 0 | 0 |
| gate2 | val | 1,424 | 1,424 | 0 | 0 | 2,848 | 0 | 0 | 0 |
| gate2 | test | 1,441 | 1,441 | 0 | 0 | 2,882 | 0 | 0 | 0 |

## gate2

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 11,781 | 11,781 | 525 | 1143 | 1500 | 0 | 0 | 23,562 | 0 | 0 | 0 |
| val | 1,424 | 1,424 | 518 | 1143 | 1402 | 0 | 0 | 2,848 | 0 | 0 | 0 |
| test | 1,441 | 1,441 | 534 | 1140 | 1341 | 0 | 0 | 2,882 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 11,298 | 1,363 | 1,378 | 14,039 | 95.86% |
| zho | 417 | 56 | 52 | 525 | 3.58% |
| und | 60 | 4 | 11 | 75 | 0.51% |
| fin | 1 | 1 | 0 | 2 | 0.01% |
| kor | 1 | 0 | 0 | 1 | 0.01% |
| wuu | 1 | 0 | 0 | 1 | 0.01% |
| rus | 1 | 0 | 0 | 1 | 0.01% |
| fra | 1 | 0 | 0 | 1 | 0.01% |
| spa | 1 | 0 | 0 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,441 |
| test_unique_inputs | 1,441 |
| train_records | 11,781 |
| train_unique_inputs | 11,781 |
| val_records | 1,424 |
| val_unique_inputs | 1,424 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0059 | 23,562 / 2,848 | toll_kind: no_toll 17%→17%, relevance: mass_casualty 25%→25%, toll_kind: current_toll 25%→25%, relevance: other 25%→25%, toll_kind: historical_toll 6%→6% |
| classifications (true_label) | train / test | 0.0114 | 23,562 / 2,882 | toll_kind: no_toll 17%→17%, relevance: other 25%→25%, relevance: mass_casualty 25%→25%, toll_kind: current_toll 25%→25%, toll_kind: historical_toll 6%→6% |
| classifications (true_label) | val / test | 0.0055 | 2,848 / 2,882 | relevance: other 25%→25%, relevance: mass_casualty 25%→25%, toll_kind: current_toll 25%→25%, toll_kind: no_toll 17%→17%, toll_kind: exposure_only 2%→2% |


### Events

No event gold.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

No event arguments.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

No trigger or argument surfaces.

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

No event instances.

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

No event lists more than one trigger, and no document has two events of one type, so there is no trigger pair.

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

No event arguments.

### Entities

None.

### Relations

None.

### Classifications

**train, relevance.** Majority class 50.1% of 11,781 answers. The marginal is `true_label`, not the menu.

#### train relevance

| label | count | share |
| --- | --- | --- |
| mass_casualty | 5,905 | 50.12% |
| other | 5,876 | 49.88% |

**train, toll_kind.** Majority class 50.1% of 11,781 answers. The marginal is `true_label`, not the menu.

#### train toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 5,905 | 50.12% |
| no_toll | 3,957 | 33.59% |
| historical_toll | 1,367 | 11.60% |
| exposure_only | 552 | 4.69% |

**val, relevance.** Majority class 50.3% of 1,424 answers. The marginal is `true_label`, not the menu.

#### val relevance

| label | count | share |
| --- | --- | --- |
| other | 716 | 50.28% |
| mass_casualty | 708 | 49.72% |

**val, toll_kind.** Majority class 49.7% of 1,424 answers. The marginal is `true_label`, not the menu.

#### val toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 708 | 49.72% |
| no_toll | 489 | 34.34% |
| historical_toll | 160 | 11.24% |
| exposure_only | 67 | 4.71% |

**test, relevance.** Majority class 50.7% of 1,441 answers. The marginal is `true_label`, not the menu.

#### test relevance

| label | count | share |
| --- | --- | --- |
| other | 731 | 50.73% |
| mass_casualty | 710 | 49.27% |

**test, toll_kind.** Majority class 49.3% of 1,441 answers. The marginal is `true_label`, not the menu.

#### test toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 710 | 49.27% |
| no_toll | 500 | 34.70% |
| historical_toll | 159 | 11.03% |
| exposure_only | 72 | 5.00% |

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
