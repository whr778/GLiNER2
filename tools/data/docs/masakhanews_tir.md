# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_tir | train | 947 | 947 | 0 | 0 | 947 | 0 | 0 | 0 |
| masakhanews_tir | val | 137 | 137 | 0 | 0 | 137 | 0 | 0 | 0 |
| masakhanews_tir | test | 272 | 272 | 0 | 0 | 272 | 0 | 0 | 0 |

## masakhanews_tir

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 947 | 947 | 412 | 1119 | 3324 | 0 | 0 | 947 | 0 | 0 | 0 |
| val | 137 | 137 | 469 | 1185 | 2375 | 0 | 0 | 137 | 0 | 0 | 0 |
| test | 272 | 272 | 413 | 983 | 2378 | 0 | 0 | 272 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| amh | 947 | 137 | 272 | 1,356 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 272 |
| test_unique_inputs | 272 |
| train_records | 947 |
| train_unique_inputs | 947 |
| val_records | 137 |
| val_unique_inputs | 137 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0054 | 947 / 137 | news topic: politics 37%→36%, news topic: sports 9%→9%, news topic: entertainment 12%→12%, news topic: business 6%→6%, news topic: health 29%→29% |
| classifications (true_label) | train / test | 0.0033 | 947 / 272 | news topic: entertainment 12%→12%, news topic: politics 37%→37%, news topic: health 29%→29%, news topic: technology 7%→7%, news topic: business 6%→6% |
| classifications (true_label) | val / test | 0.0045 | 137 / 272 | news topic: sports 9%→9%, news topic: politics 36%→37%, news topic: health 29%→29%, news topic: entertainment 12%→12%, news topic: technology 7%→7% |

Thin support in val, classifications (true_label): 2 label(s) under 10 gold. news topic: business ×8, news topic: technology ×9.

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

**train, news topic.** Majority class 37.0% of 947 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| politics | 350 | 36.96% |
| health | 276 | 29.14% |
| entertainment | 116 | 12.25% |
| sports | 87 | 9.19% |
| technology | 62 | 6.55% |
| business | 56 | 5.91% |

**val, news topic.** Majority class 36.5% of 137 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| politics | 50 | 36.50% |
| health | 40 | 29.20% |
| entertainment | 17 | 12.41% |
| sports | 13 | 9.49% |
| technology | 9 | 6.57% |
| business | 8 | 5.84% |

**test, news topic.** Majority class 36.8% of 272 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| politics | 100 | 36.76% |
| health | 79 | 29.04% |
| entertainment | 34 | 12.50% |
| sports | 25 | 9.19% |
| technology | 18 | 6.62% |
| business | 16 | 5.88% |

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
