# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_fon | train | 2,419 | 2,345 | 4,936 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_fon | val | 313 | 313 | 544 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_fon | test | 693 | 693 | 1,287 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_fon

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,419 | 2,345 | 31 | 56 | 88 | 4,936 | 0 | 0 | 0 | 0 | 0 |
| val | 313 | 313 | 30 | 51 | 68 | 544 | 0 | 0 | 0 | 0 | 0 |
| test | 693 | 693 | 31 | 51 | 70 | 1,287 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 1,420 | 161 | 364 | 1,945 | 56.79% |
| eml | 286 | 65 | 133 | 484 | 14.13% |
| gle | 182 | 22 | 47 | 251 | 7.33% |
| ita | 120 | 26 | 49 | 195 | 5.69% |
| vec | 51 | 2 | 4 | 57 | 1.66% |
| lmo | 37 | 3 | 9 | 49 | 1.43% |
| cat | 43 | 0 | 4 | 47 | 1.37% |
| nap | 25 | 8 | 12 | 45 | 1.31% |
| cbk | 28 | 1 | 4 | 33 | 0.96% |
| ces | 17 | 6 | 7 | 30 | 0.88% |
| roh | 18 | 4 | 7 | 29 | 0.85% |
| por | 25 | 0 | 1 | 26 | 0.76% |
| bak | 15 | 2 | 8 | 25 | 0.73% |
| yor | 4 | 0 | 19 | 23 | 0.67% |
| cor | 12 | 3 | 6 | 21 | 0.61% |
| ast | 20 | 0 | 1 | 21 | 0.61% |
| glg | 17 | 2 | 0 | 19 | 0.55% |
| fra | 13 | 3 | 2 | 18 | 0.53% |
| spa | 9 | 0 | 0 | 9 | 0.26% |
| bre | 4 | 1 | 3 | 8 | 0.23% |
| slk | 4 | 0 | 3 | 7 | 0.20% |
| gla | 7 | 0 | 0 | 7 | 0.20% |
| pms | 6 | 1 | 0 | 7 | 0.20% |
| grn | 5 | 1 | 0 | 6 | 0.18% |
| kom | 5 | 0 | 0 | 5 | 0.15% |
| xal | 3 | 0 | 2 | 5 | 0.15% |
| pam | 5 | 0 | 0 | 5 | 0.15% |
| gom | 2 | 1 | 1 | 4 | 0.12% |
| hun | 4 | 0 | 0 | 4 | 0.12% |
| oss | 2 | 1 | 0 | 3 | 0.09% |
| ava | 3 | 0 | 0 | 3 | 0.09% |
| tat | 1 | 0 | 1 | 2 | 0.06% |
| lez | 1 | 0 | 1 | 2 | 0.06% |
| hat | 2 | 0 | 0 | 2 | 0.06% |
| oci | 2 | 0 | 0 | 2 | 0.06% |
| mlt | 2 | 0 | 0 | 2 | 0.06% |
| rus | 2 | 0 | 0 | 2 | 0.06% |
| lao | 2 | 0 | 0 | 2 | 0.06% |
| eng | 2 | 0 | 0 | 2 | 0.06% |
| mrj | 2 | 0 | 0 | 2 | 0.06% |
| vep | 2 | 0 | 0 | 2 | 0.06% |
| diq | 2 | 0 | 0 | 2 | 0.06% |
| lav | 1 | 0 | 0 | 1 | 0.03% |
| vie | 1 | 0 | 0 | 1 | 0.03% |
| mwl | 1 | 0 | 0 | 1 | 0.03% |
| jav | 1 | 0 | 0 | 1 | 0.03% |
| arg | 1 | 0 | 0 | 1 | 0.03% |
| epo | 1 | 0 | 0 | 1 | 0.03% |
| als | 1 | 0 | 0 | 1 | 0.03% |
| hsb | 0 | 0 | 1 | 1 | 0.03% |
| nld | 0 | 0 | 1 | 1 | 0.03% |
| mlg | 0 | 0 | 1 | 1 | 0.03% |
| krc | 0 | 0 | 1 | 1 | 0.03% |
| glv | 0 | 0 | 1 | 1 | 0.03% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 693 |
| test_unique_inputs | 693 |
| train_records | 2,419 |
| train_unique_inputs | 2,345 |
| val_records | 313 |
| val_unique_inputs | 313 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.1025 | 4,936 / 544 | date 21%→28%, person 26%→19%, location 32%→29%, organization 21%→24% |
| entities (per mention) | train / test | 0.0777 | 4,936 / 1,287 | date 21%→29%, organization 21%→17%, person 26%→23%, location 32%→31% |
| entities (per mention) | val / test | 0.0704 | 544 / 1,287 | organization 24%→17%, person 19%→23%, location 29%→31%, date 28%→29% |


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
| location | 1,580 | 32.01% |
| person | 1,276 | 25.85% |
| organization | 1,052 | 21.31% |
| date | 1,028 | 20.83% |

#### val

| label | count | share |
| --- | --- | --- |
| location | 156 | 28.68% |
| date | 152 | 27.94% |
| organization | 133 | 24.45% |
| person | 103 | 18.93% |

#### test

| label | count | share |
| --- | --- | --- |
| location | 399 | 31.00% |
| date | 368 | 28.59% |
| person | 296 | 23.00% |
| organization | 224 | 17.40% |

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
