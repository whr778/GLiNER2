# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate3_eb22g | train | 20,556 | 20,556 | 0 | 0 | 41,112 | 0 | 0 | 0 |
| gate3_eb22g | val | 811 | 811 | 0 | 0 | 1,622 | 0 | 0 | 0 |
| gate3_eb22g | test | 797 | 797 | 0 | 0 | 1,594 | 0 | 0 | 0 |

## gate3_eb22g

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 20,556 | 20,556 | 499 | 1166 | 5972 | 0 | 0 | 41,112 | 0 | 0 | 0 |
| val | 811 | 811 | 461 | 1771 | 5891 | 0 | 0 | 1,622 | 0 | 0 | 0 |
| test | 797 | 797 | 409 | 1563 | 5977 | 0 | 0 | 1,594 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 14,413 | 51 | 47 | 14,511 | 65.47% |
| tur | 3,460 | 389 | 429 | 4,278 | 19.30% |
| zho | 2,676 | 371 | 321 | 3,368 | 15.20% |
| und | 5 | 0 | 0 | 5 | 0.02% |
| fra | 1 | 0 | 0 | 1 | 0.00% |
| spa | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 797 |
| test_unique_inputs | 797 |
| train_records | 20,556 |
| train_unique_inputs | 20,556 |
| val_records | 811 |
| val_unique_inputs | 811 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0153 | 41,112 / 1,622 | toll_kind: historical_toll 5%→4%, toll_kind: exposure_only 3%→3%, relevance: other 25%→25%, relevance: mass_casualty 25%→25%, toll_kind: current_toll 25%→25% |
| classifications (true_label) | train / test | 0.0267 | 41,112 / 1,594 | toll_kind: exposure_only 3%→4%, toll_kind: historical_toll 5%→4%, relevance: mass_casualty 25%→24%, toll_kind: current_toll 25%→24%, relevance: other 25%→26% |
| classifications (true_label) | val / test | 0.0261 | 1,622 / 1,594 | relevance: other 25%→26%, relevance: mass_casualty 25%→24%, toll_kind: current_toll 25%→24%, toll_kind: exposure_only 3%→4%, toll_kind: no_toll 17%→18% |


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

**train, relevance.** Majority class 50.0% of 20,556 answers. The marginal is `true_label`, not the menu.

#### train relevance

| label | count | share |
| --- | --- | --- |
| mass_casualty | 10,284 | 50.03% |
| other | 10,272 | 49.97% |

**train, toll_kind.** Majority class 50.0% of 20,556 answers. The marginal is `true_label`, not the menu.

#### train toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 10,284 | 50.03% |
| no_toll | 7,123 | 34.65% |
| historical_toll | 2,090 | 10.17% |
| exposure_only | 1,059 | 5.15% |

**val, relevance.** Majority class 50.8% of 811 answers. The marginal is `true_label`, not the menu.

#### val relevance

| label | count | share |
| --- | --- | --- |
| mass_casualty | 412 | 50.80% |
| other | 399 | 49.20% |

**val, toll_kind.** Majority class 50.8% of 811 answers. The marginal is `true_label`, not the menu.

#### val toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 412 | 50.80% |
| no_toll | 276 | 34.03% |
| historical_toll | 69 | 8.51% |
| exposure_only | 54 | 6.66% |

**test, relevance.** Majority class 51.6% of 797 answers. The marginal is `true_label`, not the menu.

#### test relevance

| label | count | share |
| --- | --- | --- |
| other | 411 | 51.57% |
| mass_casualty | 386 | 48.43% |

**test, toll_kind.** Majority class 48.4% of 797 answers. The marginal is `true_label`, not the menu.

#### test toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 386 | 48.43% |
| no_toll | 282 | 35.38% |
| exposure_only | 65 | 8.16% |
| historical_toll | 64 | 8.03% |

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | mixed (adjudicated news) |
| relations | mixed (adjudicated news) |
| events | mixed (adjudicated news) |
| classifications | mixed (adjudicated news) |
| structures | mixed (adjudicated news) |
