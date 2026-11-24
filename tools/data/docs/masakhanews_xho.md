# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_xho | train | 1,032 | 1,032 | 0 | 0 | 1,032 | 0 | 0 | 0 |
| masakhanews_xho | val | 147 | 147 | 0 | 0 | 147 | 0 | 0 | 0 |
| masakhanews_xho | test | 297 | 297 | 0 | 0 | 297 | 0 | 0 | 0 |

## masakhanews_xho

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,032 | 1,032 | 257 | 404 | 1201 | 0 | 0 | 1,032 | 0 | 0 | 0 |
| val | 147 | 147 | 247 | 377 | 733 | 0 | 0 | 147 | 0 | 0 | 0 |
| test | 297 | 297 | 257 | 385 | 812 | 0 | 0 | 297 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 1,017 | 146 | 293 | 1,456 | 98.64% |
| fil | 6 | 0 | 2 | 8 | 0.54% |
| epo | 4 | 1 | 1 | 6 | 0.41% |
| swa | 3 | 0 | 1 | 4 | 0.27% |
| eng | 1 | 0 | 0 | 1 | 0.07% |
| vol | 1 | 0 | 0 | 1 | 0.07% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 297 |
| test_unique_inputs | 297 |
| train_records | 1,032 |
| train_unique_inputs | 1,032 |
| val_records | 147 |
| val_unique_inputs | 147 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0037 | 1,032 / 147 | news topic: sports 34%→33%, news topic: politics 21%→21%, news topic: entertainment 34%→34%, news topic: business 5%→5%, news topic: health 7%→7% |
| classifications (true_label) | train / test | 0.0029 | 1,032 / 297 | news topic: entertainment 34%→34%, news topic: business 5%→5%, news topic: health 7%→7%, news topic: sports 34%→34%, news topic: politics 21%→21% |
| classifications (true_label) | val / test | 0.0063 | 147 / 297 | news topic: entertainment 34%→34%, news topic: sports 33%→34%, news topic: business 5%→5%, news topic: politics 21%→21%, news topic: health 7%→7% |

Thin support in val, classifications (true_label): 1 label(s) under 10 gold. news topic: business ×7.

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

**train, news topic.** Majority class 33.9% of 1,032 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| entertainment | 350 | 33.91% |
| sports | 347 | 33.62% |
| politics | 215 | 20.83% |
| health | 70 | 6.78% |
| business | 50 | 4.84% |

**val, news topic.** Majority class 34.0% of 147 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| entertainment | 50 | 34.01% |
| sports | 49 | 33.33% |
| politics | 31 | 21.09% |
| health | 10 | 6.80% |
| business | 7 | 4.76% |

**test, news topic.** Majority class 33.7% of 297 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| entertainment | 100 | 33.67% |
| sports | 100 | 33.67% |
| politics | 62 | 20.88% |
| health | 20 | 6.73% |
| business | 15 | 5.05% |

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
