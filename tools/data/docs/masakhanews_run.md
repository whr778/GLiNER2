# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_run | train | 1,117 | 1,117 | 0 | 0 | 1,117 | 0 | 0 | 0 |
| masakhanews_run | val | 159 | 159 | 0 | 0 | 159 | 0 | 0 | 0 |
| masakhanews_run | test | 322 | 322 | 0 | 0 | 322 | 0 | 0 | 0 |

## masakhanews_run

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,117 | 1,117 | 472 | 927 | 3134 | 0 | 0 | 1,117 | 0 | 0 | 0 |
| val | 159 | 159 | 508 | 1014 | 2111 | 0 | 0 | 159 | 0 | 0 | 0 |
| test | 322 | 322 | 485 | 878 | 2354 | 0 | 0 | 322 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 625 | 85 | 175 | 885 | 55.38% |
| swa | 391 | 57 | 119 | 567 | 35.48% |
| epo | 68 | 10 | 21 | 99 | 6.20% |
| fil | 26 | 6 | 7 | 39 | 2.44% |
| lat | 2 | 0 | 0 | 2 | 0.13% |
| war | 2 | 0 | 0 | 2 | 0.13% |
| roh | 1 | 0 | 0 | 1 | 0.06% |
| jav | 1 | 0 | 0 | 1 | 0.06% |
| vol | 1 | 0 | 0 | 1 | 0.06% |
| eus | 0 | 1 | 0 | 1 | 0.06% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 322 |
| test_unique_inputs | 322 |
| train_records | 1,117 |
| train_unique_inputs | 1,117 |
| val_records | 159 |
| val_unique_inputs | 159 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0051 | 1,117 / 159 | news topic: business 5%→4%, news topic: entertainment 10%→10%, news topic: sports 26%→26%, news topic: religion 5%→4%, news topic: politics 31%→31% |
| classifications (true_label) | train / test | 0.0042 | 1,117 / 322 | news topic: politics 31%→31%, news topic: business 5%→5%, news topic: sports 26%→26%, news topic: religion 5%→5%, news topic: entertainment 10%→10% |
| classifications (true_label) | val / test | 0.0084 | 159 / 322 | news topic: business 4%→5%, news topic: politics 31%→31%, news topic: sports 26%→26%, news topic: religion 4%→5%, news topic: entertainment 10%→10% |

Thin support in val, classifications (true_label): 2 label(s) under 10 gold. news topic: business ×7, news topic: religion ×7.

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

**train, news topic.** Majority class 31.3% of 1,117 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| politics | 350 | 31.33% |
| sports | 293 | 26.23% |
| health | 260 | 23.28% |
| entertainment | 110 | 9.85% |
| business | 53 | 4.74% |
| religion | 51 | 4.57% |

**val, news topic.** Majority class 31.4% of 159 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| politics | 50 | 31.45% |
| sports | 42 | 26.42% |
| health | 37 | 23.27% |
| entertainment | 16 | 10.06% |
| business | 7 | 4.40% |
| religion | 7 | 4.40% |

**test, news topic.** Majority class 31.1% of 322 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| politics | 100 | 31.06% |
| sports | 84 | 26.09% |
| health | 75 | 23.29% |
| entertainment | 32 | 9.94% |
| business | 16 | 4.97% |
| religion | 15 | 4.66% |

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
