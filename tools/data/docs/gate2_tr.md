# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate2_tr | train | 15,449 | 15,449 | 0 | 0 | 30,898 | 0 | 0 | 0 |
| gate2_tr | val | 1,837 | 1,837 | 0 | 0 | 3,674 | 0 | 0 | 0 |
| gate2_tr | test | 1,870 | 1,870 | 0 | 0 | 3,740 | 0 | 0 | 0 |

## gate2_tr

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 15,449 | 15,449 | 437 | 1112 | 1500 | 0 | 0 | 30,898 | 0 | 0 | 0 |
| val | 1,837 | 1,837 | 434 | 1111 | 1402 | 0 | 0 | 3,674 | 0 | 0 | 0 |
| test | 1,870 | 1,870 | 433 | 1104 | 1348 | 0 | 0 | 3,740 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 11,556 | 1,389 | 1,384 | 14,329 | 74.80% |
| tur | 3,406 | 390 | 424 | 4,220 | 22.03% |
| zho | 430 | 55 | 52 | 537 | 2.80% |
| und | 52 | 2 | 9 | 63 | 0.33% |
| fin | 1 | 0 | 0 | 1 | 0.01% |
| war | 1 | 0 | 0 | 1 | 0.01% |
| epo | 1 | 0 | 0 | 1 | 0.01% |
| spa | 1 | 0 | 0 | 1 | 0.01% |
| nor | 1 | 0 | 0 | 1 | 0.01% |
| por | 0 | 1 | 0 | 1 | 0.01% |
| vie | 0 | 0 | 1 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,870 |
| test_unique_inputs | 1,870 |
| train_records | 15,449 |
| train_unique_inputs | 15,449 |
| val_records | 1,837 |
| val_unique_inputs | 1,837 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0050 | 30,898 / 3,674 | toll_kind: no_toll 17%→17%, toll_kind: exposure_only 3%→3%, relevance: other 25%→25%, relevance: mass_casualty 25%→25%, toll_kind: current_toll 25%→25% |
| classifications (true_label) | train / test | 0.0087 | 30,898 / 3,740 | toll_kind: no_toll 17%→17%, toll_kind: exposure_only 3%→3%, relevance: mass_casualty 25%→25%, toll_kind: current_toll 25%→25%, relevance: other 25%→25% |
| classifications (true_label) | val / test | 0.0037 | 3,674 / 3,740 | toll_kind: no_toll 17%→17%, toll_kind: exposure_only 3%→3%, relevance: mass_casualty 25%→25%, toll_kind: current_toll 25%→25%, relevance: other 25%→25% |


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

**train, relevance.** Majority class 50.1% of 15,449 answers. The marginal is `true_label`, not the menu.

#### train relevance

| label | count | share |
| --- | --- | --- |
| other | 7,735 | 50.07% |
| mass_casualty | 7,714 | 49.93% |

**train, toll_kind.** Majority class 49.9% of 15,449 answers. The marginal is `true_label`, not the menu.

#### train toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 7,714 | 49.93% |
| no_toll | 5,305 | 34.34% |
| historical_toll | 1,567 | 10.14% |
| exposure_only | 863 | 5.59% |

**val, relevance.** Majority class 50.2% of 1,837 answers. The marginal is `true_label`, not the menu.

#### val relevance

| label | count | share |
| --- | --- | --- |
| mass_casualty | 922 | 50.19% |
| other | 915 | 49.81% |

**val, toll_kind.** Majority class 50.2% of 1,837 answers. The marginal is `true_label`, not the menu.

#### val toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 922 | 50.19% |
| no_toll | 617 | 33.59% |
| historical_toll | 189 | 10.29% |
| exposure_only | 109 | 5.93% |

**test, relevance.** Majority class 50.4% of 1,870 answers. The marginal is `true_label`, not the menu.

#### test relevance

| label | count | share |
| --- | --- | --- |
| mass_casualty | 942 | 50.37% |
| other | 928 | 49.63% |

**test, toll_kind.** Majority class 50.4% of 1,870 answers. The marginal is `true_label`, not the menu.

#### test toll_kind

| label | count | share |
| --- | --- | --- |
| current_toll | 942 | 50.37% |
| no_toll | 618 | 33.05% |
| historical_toll | 192 | 10.27% |
| exposure_only | 118 | 6.31% |

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
