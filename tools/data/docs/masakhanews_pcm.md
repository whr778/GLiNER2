# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_pcm | train | 1,054 | 1,054 | 0 | 0 | 1,054 | 0 | 0 | 0 |
| masakhanews_pcm | val | 152 | 152 | 0 | 0 | 152 | 0 | 0 | 0 |
| masakhanews_pcm | test | 305 | 305 | 0 | 0 | 305 | 0 | 0 | 0 |

## masakhanews_pcm

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,054 | 1,054 | 378 | 821 | 6117 | 0 | 0 | 1,054 | 0 | 0 | 0 |
| val | 152 | 152 | 413 | 835 | 1903 | 0 | 0 | 152 | 0 | 0 | 0 |
| test | 305 | 305 | 426 | 972 | 2921 | 0 | 0 | 305 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 718 | 102 | 207 | 1,027 | 67.97% |
| und | 336 | 50 | 98 | 484 | 32.03% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 305 |
| test_unique_inputs | 305 |
| train_records | 1,054 |
| train_unique_inputs | 1,054 |
| val_records | 152 |
| val_unique_inputs | 152 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0059 | 1,054 / 152 | news topic: business 6%→7%, news topic: sports 33%→32%, news topic: entertainment 31%→30%, news topic: politics 20%→20%, news topic: health 10%→11% |
| classifications (true_label) | train / test | 0.0047 | 1,054 / 305 | news topic: entertainment 31%→30%, news topic: business 6%→7%, news topic: politics 20%→20%, news topic: sports 33%→32%, news topic: health 10%→10% |
| classifications (true_label) | val / test | 0.0022 | 152 / 305 | news topic: sports 32%→32%, news topic: entertainment 30%→30%, news topic: politics 20%→20%, news topic: health 11%→10%, news topic: business 7%→7% |


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

**train, news topic.** Majority class 32.5% of 1,054 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| sports | 343 | 32.54% |
| entertainment | 322 | 30.55% |
| politics | 213 | 20.21% |
| health | 110 | 10.44% |
| business | 66 | 6.26% |

**val, news topic.** Majority class 32.2% of 152 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| sports | 49 | 32.24% |
| entertainment | 46 | 30.26% |
| politics | 31 | 20.39% |
| health | 16 | 10.53% |
| business | 10 | 6.58% |

**test, news topic.** Majority class 32.5% of 305 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| sports | 99 | 32.46% |
| entertainment | 92 | 30.16% |
| politics | 62 | 20.33% |
| health | 32 | 10.49% |
| business | 20 | 6.56% |

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
