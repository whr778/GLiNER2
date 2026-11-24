# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_eng | train | 3,308 | 3,308 | 0 | 0 | 3,308 | 0 | 0 | 0 |
| masakhanews_eng | val | 472 | 472 | 0 | 0 | 472 | 0 | 0 | 0 |
| masakhanews_eng | test | 948 | 948 | 0 | 0 | 948 | 0 | 0 | 0 |

## masakhanews_eng

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,308 | 3,308 | 530 | 1168 | 5912 | 0 | 0 | 3,308 | 0 | 0 | 0 |
| val | 472 | 472 | 547 | 1111 | 4266 | 0 | 0 | 472 | 0 | 0 | 0 |
| test | 948 | 948 | 538 | 1138 | 4082 | 0 | 0 | 948 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 3,308 | 472 | 948 | 4,728 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 948 |
| test_unique_inputs | 948 |
| train_records | 3,308 |
| train_unique_inputs | 3,308 |
| val_records | 472 |
| val_unique_inputs | 472 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0015 | 3,308 / 472 | news topic: health 16%→16%, news topic: politics 17%→17%, news topic: business 17%→17%, news topic: technology 13%→13%, news topic: sports 21%→21% |
| classifications (true_label) | train / test | 0.0013 | 3,308 / 948 | news topic: politics 17%→17%, news topic: sports 21%→21%, news topic: entertainment 16%→16%, news topic: health 16%→16%, news topic: business 17%→17% |
| classifications (true_label) | val / test | 0.0023 | 472 / 948 | news topic: health 16%→16%, news topic: sports 21%→21%, news topic: business 17%→17%, news topic: entertainment 16%→16%, news topic: technology 13%→13% |


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

**train, news topic.** Majority class 21.2% of 3,308 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| sports | 700 | 21.16% |
| politics | 573 | 17.32% |
| business | 559 | 16.90% |
| entertainment | 525 | 15.87% |
| health | 522 | 15.78% |
| technology | 429 | 12.97% |

**val, news topic.** Majority class 21.2% of 472 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| sports | 100 | 21.19% |
| politics | 82 | 17.37% |
| business | 80 | 16.95% |
| entertainment | 75 | 15.89% |
| health | 74 | 15.68% |
| technology | 61 | 12.92% |

**test, news topic.** Majority class 21.1% of 948 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| sports | 200 | 21.10% |
| politics | 165 | 17.41% |
| business | 160 | 16.88% |
| entertainment | 150 | 15.82% |
| health | 150 | 15.82% |
| technology | 123 | 12.97% |

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
