# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_bbj | train | 1,717 | 1,717 | 2,599 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_bbj | val | 234 | 234 | 348 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_bbj | test | 607 | 607 | 1,027 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_bbj

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,717 | 1,717 | 15 | 25 | 48 | 2,599 | 0 | 0 | 0 | 0 | 0 |
| val | 234 | 234 | 16 | 27 | 43 | 348 | 0 | 0 | 0 | 0 | 0 |
| test | 607 | 607 | 17 | 29 | 45 | 1,027 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 835 | 125 | 297 | 1,257 | 49.14% |
| aze | 508 | 57 | 176 | 741 | 28.97% |
| tat | 77 | 15 | 50 | 142 | 5.55% |
| som | 60 | 7 | 17 | 84 | 3.28% |
| fry | 23 | 2 | 9 | 34 | 1.33% |
| hye | 22 | 4 | 4 | 30 | 1.17% |
| est | 23 | 2 | 5 | 30 | 1.17% |
| sah | 20 | 4 | 4 | 28 | 1.09% |
| vep | 9 | 3 | 9 | 21 | 0.82% |
| gle | 8 | 3 | 8 | 19 | 0.74% |
| ltz | 13 | 0 | 5 | 18 | 0.70% |
| lao | 11 | 1 | 2 | 14 | 0.55% |
| new | 11 | 0 | 0 | 11 | 0.43% |
| tyv | 8 | 0 | 0 | 8 | 0.31% |
| rus | 6 | 2 | 0 | 8 | 0.31% |
| mon | 7 | 1 | 0 | 8 | 0.31% |
| mrj | 5 | 0 | 2 | 7 | 0.27% |
| afr | 6 | 0 | 1 | 7 | 0.27% |
| ces | 4 | 0 | 3 | 7 | 0.27% |
| diq | 6 | 0 | 1 | 7 | 0.27% |
| mhr | 4 | 1 | 1 | 6 | 0.23% |
| lmo | 5 | 0 | 0 | 5 | 0.20% |
| fra | 2 | 0 | 3 | 5 | 0.20% |
| cat | 3 | 0 | 1 | 4 | 0.16% |
| glg | 2 | 2 | 0 | 4 | 0.16% |
| por | 3 | 1 | 0 | 4 | 0.16% |
| jbo | 2 | 1 | 0 | 3 | 0.12% |
| mwl | 3 | 0 | 0 | 3 | 0.12% |
| epo | 3 | 0 | 0 | 3 | 0.12% |
| bak | 2 | 0 | 1 | 3 | 0.12% |
| als | 2 | 0 | 0 | 2 | 0.08% |
| cor | 1 | 0 | 1 | 2 | 0.08% |
| lez | 2 | 0 | 0 | 2 | 0.08% |
| scn | 2 | 0 | 0 | 2 | 0.08% |
| hsb | 2 | 0 | 0 | 2 | 0.08% |
| ilo | 2 | 0 | 0 | 2 | 0.08% |
| kur | 1 | 0 | 1 | 2 | 0.08% |
| myv | 1 | 1 | 0 | 2 | 0.08% |
| war | 0 | 0 | 2 | 2 | 0.08% |
| swa | 1 | 0 | 0 | 1 | 0.04% |
| cym | 1 | 0 | 0 | 1 | 0.04% |
| wln | 1 | 0 | 0 | 1 | 0.04% |
| mar | 1 | 0 | 0 | 1 | 0.04% |
| kan | 1 | 0 | 0 | 1 | 0.04% |
| gla | 1 | 0 | 0 | 1 | 0.04% |
| fin | 1 | 0 | 0 | 1 | 0.04% |
| gom | 1 | 0 | 0 | 1 | 0.04% |
| pnb | 1 | 0 | 0 | 1 | 0.04% |
| fil | 1 | 0 | 0 | 1 | 0.04% |
| pol | 1 | 0 | 0 | 1 | 0.04% |
| yid | 1 | 0 | 0 | 1 | 0.04% |
| oss | 1 | 0 | 0 | 1 | 0.04% |
| ceb | 0 | 1 | 0 | 1 | 0.04% |
| eng | 0 | 1 | 0 | 1 | 0.04% |
| slk | 0 | 0 | 1 | 1 | 0.04% |
| kom | 0 | 0 | 1 | 1 | 0.04% |
| krc | 0 | 0 | 1 | 1 | 0.04% |
| cbk | 0 | 0 | 1 | 1 | 0.04% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 607 |
| test_unique_inputs | 607 |
| train_records | 1,717 |
| train_unique_inputs | 1,717 |
| val_records | 234 |
| val_unique_inputs | 234 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0454 | 2,599 / 348 | person 28%→32%, date 22%→20%, organization 22%→20%, location 28%→27% |
| entities (per mention) | train / test | 0.1092 | 2,599 / 1,027 | person 28%→39%, date 22%→16%, location 28%→25%, organization 22%→20% |
| entities (per mention) | val / test | 0.0638 | 348 / 1,027 | person 32%→39%, date 20%→16%, location 27%→25%, organization 20%→20% |


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

#### train

| label | count | share |
| --- | --- | --- |
| location | 730 | 28.09% |
| person | 726 | 27.93% |
| date | 576 | 22.16% |
| organization | 567 | 21.82% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 113 | 32.47% |
| location | 94 | 27.01% |
| organization | 71 | 20.40% |
| date | 70 | 20.11% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 399 | 38.85% |
| location | 254 | 24.73% |
| organization | 207 | 20.16% |
| date | 167 | 16.26% |

### Relations

None.

### Classifications

None.

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
