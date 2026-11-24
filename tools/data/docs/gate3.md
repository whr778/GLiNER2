# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate3 | train | 17,827 | 17,827 | 0 | 0 | 35,654 | 0 | 0 | 0 |
| gate3 | val | 2,181 | 2,181 | 0 | 0 | 4,362 | 0 | 0 | 0 |
| gate3 | test | 2,156 | 2,156 | 0 | 0 | 4,312 | 0 | 0 | 0 |

## gate3

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 17,827 | 17,827 | 495 | 1172 | 5972 | 0 | 0 | 35,654 | 0 | 0 | 0 |
| val | 2,181 | 2,181 | 492 | 1176 | 5891 | 0 | 0 | 4,362 | 0 | 0 | 0 |
| test | 2,156 | 2,156 | 493 | 1173 | 5977 | 0 | 0 | 4,312 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 11,685 | 1,421 | 1,405 | 14,511 | 65.47% |
| tur | 3,460 | 389 | 429 | 4,278 | 19.30% |
| zho | 2,676 | 371 | 321 | 3,368 | 15.20% |
| und | 4 | 0 | 1 | 5 | 0.02% |
| fra | 1 | 0 | 0 | 1 | 0.00% |
| spa | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 2,156 |
| test_unique_inputs | 2,156 |
| train_records | 17,827 |
| train_unique_inputs | 17,827 |
| val_records | 2,181 |
| val_unique_inputs | 2,181 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0072 | 35,654 / 4,362 | relevance: other 25%→25%, relevance: mass_casualty 25%→25%, toll_kind: current_toll 25%→25%, toll_kind: exposure_only 3%→3%, toll_kind: no_toll 17%→17% |
| classifications (true_label) | train / test | 0.0089 | 35,654 / 4,312 | toll_kind: exposure_only 3%→3%, relevance: other 25%→25%, relevance: mass_casualty 25%→25%, toll_kind: current_toll 25%→25%, toll_kind: no_toll 17%→17% |
| classifications (true_label) | val / test | 0.0033 | 4,362 / 4,312 | toll_kind: exposure_only 3%→3%, toll_kind: no_toll 17%→17%, toll_kind: historical_toll 5%→5%, relevance: mass_casualty 25%→25%, toll_kind: current_toll 25%→25% |


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

**train, relevance.** Majority class 50.1% of 17,827 answers. The marginal is `true_label`, not the menu.

#### train relevance

| label | count | share |
| --- | --- | --- |
| mass_casualty | 8,940 | 50.15% |
| other | 8,887 | 49.85% |

**train, toll_kind.** Majority class 50.1% of 17,827 answers. The marginal is `true_label`, not the menu.

#### train toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 8,940 | 50.15% |
| no_toll | 6,178 | 34.66% |
| historical_toll | 1,784 | 10.01% |
| exposure_only | 925 | 5.19% |

**val, relevance.** Majority class 50.6% of 2,181 answers. The marginal is `true_label`, not the menu.

#### val relevance

| label | count | share |
| --- | --- | --- |
| other | 1,103 | 50.57% |
| mass_casualty | 1,078 | 49.43% |

**val, toll_kind.** Majority class 49.4% of 2,181 answers. The marginal is `true_label`, not the menu.

#### val toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 1,078 | 49.43% |
| no_toll | 760 | 34.85% |
| historical_toll | 222 | 10.18% |
| exposure_only | 121 | 5.55% |

**test, relevance.** Majority class 50.6% of 2,156 answers. The marginal is `true_label`, not the menu.

#### test relevance

| label | count | share |
| --- | --- | --- |
| other | 1,092 | 50.65% |
| mass_casualty | 1,064 | 49.35% |

**test, toll_kind.** Majority class 49.4% of 2,156 answers. The marginal is `true_label`, not the menu.

#### test toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 1,064 | 49.35% |
| no_toll | 743 | 34.46% |
| historical_toll | 217 | 10.06% |
| exposure_only | 132 | 6.12% |

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
