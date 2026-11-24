# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_hau | train | 2,219 | 2,219 | 0 | 0 | 2,219 | 0 | 0 | 0 |
| masakhanews_hau | val | 317 | 317 | 0 | 0 | 317 | 0 | 0 | 0 |
| masakhanews_hau | test | 637 | 637 | 0 | 0 | 637 | 0 | 0 | 0 |

## masakhanews_hau

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,219 | 2,219 | 389 | 1069 | 3400 | 0 | 0 | 2,219 | 0 | 0 | 0 |
| val | 317 | 317 | 382 | 955 | 3510 | 0 | 0 | 317 | 0 | 0 | 0 |
| test | 637 | 637 | 421 | 1115 | 5580 | 0 | 0 | 637 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 1,576 | 228 | 424 | 2,228 | 70.22% |
| swa | 322 | 45 | 113 | 480 | 15.13% |
| ido | 150 | 17 | 41 | 208 | 6.56% |
| epo | 107 | 16 | 36 | 159 | 5.01% |
| que | 41 | 8 | 16 | 65 | 2.05% |
| bre | 9 | 2 | 2 | 13 | 0.41% |
| jav | 8 | 1 | 1 | 10 | 0.32% |
| nld | 2 | 0 | 0 | 2 | 0.06% |
| cor | 1 | 0 | 1 | 2 | 0.06% |
| vol | 1 | 0 | 0 | 1 | 0.03% |
| tur | 1 | 0 | 0 | 1 | 0.03% |
| eng | 1 | 0 | 0 | 1 | 0.03% |
| lmo | 0 | 0 | 1 | 1 | 0.03% |
| srp | 0 | 0 | 1 | 1 | 0.03% |
| por | 0 | 0 | 1 | 1 | 0.03% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 637 |
| test_unique_inputs | 637 |
| train_records | 2,219 |
| train_unique_inputs | 2,219 |
| val_records | 317 |
| val_unique_inputs | 317 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0018 | 2,219 / 317 | news topic: sports 16%→16%, news topic: health 16%→15%, news topic: religion 16%→15%, news topic: business 13%→13%, news topic: entertainment 16%→16% |
| classifications (true_label) | train / test | 0.0017 | 2,219 / 637 | news topic: technology 9%→9%, news topic: entertainment 16%→16%, news topic: politics 16%→16%, news topic: sports 16%→16%, news topic: business 13%→13% |
| classifications (true_label) | val / test | 0.0028 | 317 / 637 | news topic: technology 9%→9%, news topic: health 15%→16%, news topic: religion 15%→16%, news topic: entertainment 16%→16%, news topic: sports 16%→16% |


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

**train, news topic.** Majority class 15.8% of 2,219 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| entertainment | 350 | 15.77% |
| politics | 350 | 15.77% |
| sports | 347 | 15.64% |
| religion | 345 | 15.55% |
| health | 345 | 15.55% |
| business | 279 | 12.57% |
| technology | 203 | 9.15% |

**val, news topic.** Majority class 15.8% of 317 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| entertainment | 50 | 15.77% |
| sports | 50 | 15.77% |
| politics | 50 | 15.77% |
| health | 49 | 15.46% |
| religion | 49 | 15.46% |
| business | 40 | 12.62% |
| technology | 29 | 9.15% |

**test, news topic.** Majority class 15.7% of 637 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| entertainment | 100 | 15.70% |
| politics | 100 | 15.70% |
| sports | 100 | 15.70% |
| health | 99 | 15.54% |
| religion | 99 | 15.54% |
| business | 80 | 12.56% |
| technology | 59 | 9.26% |

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
