# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_som | train | 1,021 | 1,021 | 0 | 0 | 1,021 | 0 | 0 | 0 |
| masakhanews_som | val | 148 | 148 | 0 | 0 | 148 | 0 | 0 | 0 |
| masakhanews_som | test | 294 | 294 | 0 | 0 | 294 | 0 | 0 | 0 |

## masakhanews_som

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,021 | 1,021 | 578 | 1022 | 3015 | 0 | 0 | 1,021 | 0 | 0 | 0 |
| val | 148 | 148 | 563 | 982 | 2169 | 0 | 0 | 148 | 0 | 0 | 0 |
| test | 294 | 294 | 563 | 1016 | 4792 | 0 | 0 | 294 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 955 | 137 | 276 | 1,368 | 93.51% |
| som | 65 | 11 | 18 | 94 | 6.43% |
| cym | 1 | 0 | 0 | 1 | 0.07% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 294 |
| test_unique_inputs | 294 |
| train_records | 1,021 |
| train_unique_inputs | 1,021 |
| val_records | 148 |
| val_unique_inputs | 148 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0080 | 1,021 / 148 | news topic: politics 34%→34%, news topic: business 8%→8%, news topic: religion 5%→5%, news topic: technology 9%→9%, news topic: health 24%→24% |
| classifications (true_label) | train / test | 0.0033 | 1,021 / 294 | news topic: politics 34%→34%, news topic: sports 10%→10%, news topic: religion 5%→5%, news topic: business 8%→8%, news topic: health 24%→24% |
| classifications (true_label) | val / test | 0.0074 | 148 / 294 | news topic: religion 5%→5%, news topic: business 8%→8%, news topic: technology 9%→9%, news topic: politics 34%→34%, news topic: health 24%→24% |

Thin support in val, classifications (true_label): 1 label(s) under 10 gold. news topic: religion ×7.

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

**train, news topic.** Majority class 34.3% of 1,021 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| politics | 350 | 34.28% |
| health | 247 | 24.19% |
| sports | 103 | 10.09% |
| entertainment | 97 | 9.50% |
| technology | 94 | 9.21% |
| business | 79 | 7.74% |
| religion | 51 | 5.00% |

**val, news topic.** Majority class 33.8% of 148 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| politics | 50 | 33.78% |
| health | 36 | 24.32% |
| sports | 15 | 10.14% |
| technology | 14 | 9.46% |
| entertainment | 14 | 9.46% |
| business | 12 | 8.11% |
| religion | 7 | 4.73% |

**test, news topic.** Majority class 34.0% of 294 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| politics | 100 | 34.01% |
| health | 71 | 24.15% |
| sports | 30 | 10.20% |
| entertainment | 28 | 9.52% |
| technology | 27 | 9.18% |
| business | 23 | 7.82% |
| religion | 15 | 5.10% |

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
