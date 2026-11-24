# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews | train | 21,499 | 21,499 | 0 | 0 | 21,499 | 0 | 0 | 0 |
| masakhanews | val | 3,094 | 3,094 | 0 | 0 | 3,094 | 0 | 0 | 0 |
| masakhanews | test | 6,236 | 6,236 | 0 | 0 | 6,236 | 0 | 0 | 0 |

## masakhanews

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 21,499 | 21,499 | 399 | 1077 | 9991 | 0 | 0 | 21,499 | 0 | 0 | 0 |
| val | 3,094 | 3,094 | 395 | 1038 | 6331 | 0 | 0 | 3,094 | 0 | 0 | 0 |
| test | 6,236 | 6,236 | 403 | 1053 | 5580 | 0 | 0 | 6,236 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 8,548 | 1,238 | 2,479 | 12,265 | 39.78% |
| eng | 4,028 | 574 | 1,155 | 5,757 | 18.67% |
| swa | 2,761 | 396 | 818 | 3,975 | 12.89% |
| amh | 2,134 | 312 | 645 | 3,091 | 10.03% |
| fra | 1,473 | 210 | 422 | 2,105 | 6.83% |
| yor | 884 | 121 | 255 | 1,260 | 4.09% |
| fin | 313 | 39 | 75 | 427 | 1.39% |
| war | 244 | 48 | 67 | 359 | 1.16% |
| epo | 222 | 34 | 73 | 329 | 1.07% |
| vie | 173 | 24 | 44 | 241 | 0.78% |
| ido | 150 | 17 | 41 | 208 | 0.67% |
| gle | 125 | 14 | 30 | 169 | 0.55% |
| kom | 88 | 10 | 25 | 123 | 0.40% |
| fil | 83 | 10 | 24 | 117 | 0.38% |
| sun | 75 | 14 | 20 | 109 | 0.35% |
| som | 65 | 11 | 18 | 94 | 0.30% |
| que | 41 | 8 | 16 | 65 | 0.21% |
| cor | 16 | 2 | 2 | 20 | 0.06% |
| mlt | 15 | 0 | 5 | 20 | 0.06% |
| hat | 9 | 3 | 7 | 19 | 0.06% |
| bre | 9 | 2 | 2 | 13 | 0.04% |
| roh | 8 | 1 | 3 | 12 | 0.04% |
| jav | 9 | 1 | 1 | 11 | 0.04% |
| lat | 4 | 1 | 1 | 6 | 0.02% |
| gla | 4 | 0 | 1 | 5 | 0.02% |
| hrv | 2 | 1 | 1 | 4 | 0.01% |
| vol | 3 | 0 | 0 | 3 | 0.01% |
| mlg | 1 | 1 | 1 | 3 | 0.01% |
| nld | 2 | 0 | 0 | 2 | 0.01% |
| urd | 2 | 0 | 0 | 2 | 0.01% |
| ceb | 2 | 0 | 0 | 2 | 0.01% |
| lao | 1 | 1 | 0 | 2 | 0.01% |
| lmo | 0 | 0 | 2 | 2 | 0.01% |
| tur | 1 | 0 | 0 | 1 | 0.00% |
| cym | 1 | 0 | 0 | 1 | 0.00% |
| pfl | 1 | 0 | 0 | 1 | 0.00% |
| grn | 1 | 0 | 0 | 1 | 0.00% |
| cat | 1 | 0 | 0 | 1 | 0.00% |
| eus | 0 | 1 | 0 | 1 | 0.00% |
| srp | 0 | 0 | 1 | 1 | 0.00% |
| por | 0 | 0 | 1 | 1 | 0.00% |
| cbk | 0 | 0 | 1 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 6,236 |
| test_unique_inputs | 6,236 |
| train_records | 21,499 |
| train_unique_inputs | 21,499 |
| val_records | 3,094 |
| val_unique_inputs | 3,094 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0044 | 21,499 / 3,094 | news topic: health 19%→20%, news topic: politics 26%→26%, news topic: business 13%→13%, news topic: entertainment 12%→12%, news topic: sports 20%→20% |
| classifications (true_label) | train / test | 0.0068 | 21,499 / 6,236 | news topic: health 19%→20%, news topic: politics 26%→25%, news topic: sports 20%→20%, news topic: entertainment 12%→12%, news topic: business 13%→13% |
| classifications (true_label) | val / test | 0.0034 | 3,094 / 6,236 | news topic: health 20%→20%, news topic: politics 26%→25%, news topic: sports 20%→20%, news topic: religion 5%→5%, news topic: business 13%→13% |


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

**train, news topic.** Majority class 25.8% of 21,499 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| politics | 5,546 | 25.80% |
| sports | 4,398 | 20.46% |
| health | 4,172 | 19.41% |
| business | 2,718 | 12.64% |
| entertainment | 2,651 | 12.33% |
| technology | 1,029 | 4.79% |
| religion | 985 | 4.58% |

**val, news topic.** Majority class 25.7% of 3,094 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| politics | 794 | 25.66% |
| sports | 631 | 20.39% |
| health | 614 | 19.84% |
| business | 388 | 12.54% |
| entertainment | 379 | 12.25% |
| technology | 148 | 4.78% |
| religion | 140 | 4.52% |

**test, news topic.** Majority class 25.5% of 6,236 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| politics | 1,589 | 25.48% |
| sports | 1,265 | 20.29% |
| health | 1,252 | 20.08% |
| business | 785 | 12.59% |
| entertainment | 762 | 12.22% |
| technology | 297 | 4.76% |
| religion | 286 | 4.59% |

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
