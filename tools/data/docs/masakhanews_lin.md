# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_lin | train | 608 | 608 | 0 | 0 | 608 | 0 | 0 | 0 |
| masakhanews_lin | val | 87 | 87 | 0 | 0 | 87 | 0 | 0 | 0 |
| masakhanews_lin | test | 175 | 175 | 0 | 0 | 175 | 0 | 0 | 0 |

## masakhanews_lin

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 608 | 608 | 188 | 304 | 598 | 0 | 0 | 608 | 0 | 0 | 0 |
| val | 87 | 87 | 205 | 284 | 410 | 0 | 0 | 87 | 0 | 0 | 0 |
| test | 175 | 175 | 195 | 316 | 722 | 0 | 0 | 175 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| swa | 387 | 57 | 109 | 553 | 63.56% |
| und | 77 | 9 | 24 | 110 | 12.64% |
| sun | 75 | 14 | 20 | 109 | 12.53% |
| epo | 29 | 4 | 12 | 45 | 5.17% |
| mlt | 15 | 0 | 5 | 20 | 2.30% |
| cor | 14 | 2 | 1 | 17 | 1.95% |
| roh | 7 | 1 | 3 | 11 | 1.26% |
| gla | 4 | 0 | 0 | 4 | 0.46% |
| fil | 0 | 0 | 1 | 1 | 0.11% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 175 |
| test_unique_inputs | 175 |
| train_records | 608 |
| train_unique_inputs | 608 |
| val_records | 87 |
| val_unique_inputs | 87 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0064 | 608 / 87 | news topic: sports 11%→11%, news topic: health 22%→22%, news topic: business 9%→9%, news topic: politics 58%→57% |
| classifications (true_label) | train / test | 0.0042 | 608 / 175 | news topic: politics 58%→57%, news topic: business 9%→10%, news topic: health 22%→22%, news topic: sports 11%→11% |
| classifications (true_label) | val / test | 0.0097 | 87 / 175 | news topic: sports 11%→11%, news topic: business 9%→10%, news topic: health 22%→22%, news topic: politics 57%→57% |

Thin support in val, classifications (true_label): 1 label(s) under 10 gold. news topic: business ×8.

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

**train, news topic.** Majority class 57.6% of 608 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| politics | 350 | 57.57% |
| health | 135 | 22.20% |
| sports | 66 | 10.86% |
| business | 57 | 9.38% |

**val, news topic.** Majority class 57.5% of 87 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| politics | 50 | 57.47% |
| health | 19 | 21.84% |
| sports | 10 | 11.49% |
| business | 8 | 9.20% |

**test, news topic.** Majority class 57.1% of 175 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| politics | 100 | 57.14% |
| health | 39 | 22.29% |
| sports | 19 | 10.86% |
| business | 17 | 9.71% |

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
