# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_swa | train | 1,658 | 1,658 | 0 | 0 | 1,658 | 0 | 0 | 0 |
| masakhanews_swa | val | 237 | 237 | 0 | 0 | 237 | 0 | 0 | 0 |
| masakhanews_swa | test | 476 | 476 | 0 | 0 | 476 | 0 | 0 | 0 |

## masakhanews_swa

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,658 | 1,658 | 573 | 1169 | 4321 | 0 | 0 | 1,658 | 0 | 0 | 0 |
| val | 237 | 237 | 550 | 1117 | 1967 | 0 | 0 | 237 | 0 | 0 | 0 |
| test | 476 | 476 | 569 | 1116 | 2088 | 0 | 0 | 476 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| swa | 1,657 | 236 | 475 | 2,368 | 99.87% |
| und | 1 | 1 | 0 | 2 | 0.08% |
| cbk | 0 | 0 | 1 | 1 | 0.04% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 476 |
| test_unique_inputs | 476 |
| train_records | 1,658 |
| train_unique_inputs | 1,658 |
| val_records | 237 |
| val_unique_inputs | 237 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0036 | 1,658 / 237 | news topic: business 13%→13%, news topic: technology 7%→7%, news topic: entertainment 4%→4%, news topic: religion 12%→12%, news topic: health 21%→21% |
| classifications (true_label) | train / test | 0.0031 | 1,658 / 476 | news topic: business 13%→13%, news topic: health 21%→21%, news topic: sports 21%→21%, news topic: politics 21%→21%, news topic: entertainment 4%→4% |
| classifications (true_label) | val / test | 0.0052 | 237 / 476 | news topic: business 13%→13%, news topic: technology 7%→7%, news topic: religion 12%→12%, news topic: health 21%→21%, news topic: sports 21%→21% |


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

**train, news topic.** Majority class 21.1% of 1,658 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| sports | 350 | 21.11% |
| health | 350 | 21.11% |
| politics | 350 | 21.11% |
| business | 221 | 13.33% |
| religion | 204 | 12.30% |
| technology | 115 | 6.94% |
| entertainment | 68 | 4.10% |

**val, news topic.** Majority class 21.1% of 237 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| politics | 50 | 21.10% |
| sports | 50 | 21.10% |
| health | 50 | 21.10% |
| business | 31 | 13.08% |
| religion | 29 | 12.24% |
| technology | 17 | 7.17% |
| entertainment | 10 | 4.22% |

**test, news topic.** Majority class 21.0% of 476 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| health | 100 | 21.01% |
| politics | 100 | 21.01% |
| sports | 100 | 21.01% |
| business | 64 | 13.45% |
| religion | 59 | 12.39% |
| technology | 33 | 6.93% |
| entertainment | 20 | 4.20% |

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
