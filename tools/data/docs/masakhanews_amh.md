# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_amh | train | 1,187 | 1,187 | 0 | 0 | 1,187 | 0 | 0 | 0 |
| masakhanews_amh | val | 175 | 175 | 0 | 0 | 175 | 0 | 0 | 0 |
| masakhanews_amh | test | 373 | 373 | 0 | 0 | 373 | 0 | 0 | 0 |

## masakhanews_amh

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,187 | 1,187 | 358 | 913 | 3890 | 0 | 0 | 1,187 | 0 | 0 | 0 |
| val | 175 | 175 | 329 | 765 | 2187 | 0 | 0 | 175 | 0 | 0 | 0 |
| test | 373 | 373 | 352 | 986 | 2560 | 0 | 0 | 373 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| amh | 1,187 | 175 | 373 | 1,735 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 373 |
| test_unique_inputs | 373 |
| train_records | 1,187 |
| train_unique_inputs | 1,187 |
| val_records | 175 |
| val_unique_inputs | 175 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0208 | 1,187 / 175 | news topic: health 20%→22%, news topic: politics 29%→29%, news topic: sports 28%→27%, news topic: business 24%→23% |
| classifications (true_label) | train / test | 0.0638 | 1,187 / 373 | news topic: health 20%→26%, news topic: politics 29%→27%, news topic: sports 28%→25%, news topic: business 24%→22% |
| classifications (true_label) | val / test | 0.0429 | 175 / 373 | news topic: health 22%→26%, news topic: politics 29%→27%, news topic: sports 27%→25%, news topic: business 23%→22% |


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

**train, news topic.** Majority class 29.3% of 1,187 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| politics | 348 | 29.32% |
| sports | 327 | 27.55% |
| business | 279 | 23.50% |
| health | 233 | 19.63% |

**val, news topic.** Majority class 28.6% of 175 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| politics | 50 | 28.57% |
| sports | 47 | 26.86% |
| business | 40 | 22.86% |
| health | 38 | 21.71% |

**test, news topic.** Majority class 26.8% of 373 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| politics | 100 | 26.81% |
| health | 97 | 26.01% |
| sports | 95 | 25.47% |
| business | 81 | 21.72% |

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
