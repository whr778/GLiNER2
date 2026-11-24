# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_yor | train | 4,553 | 4,553 | 11,266 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_yor | val | 537 | 537 | 969 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_yor | test | 996 | 996 | 1,966 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_yor

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,553 | 4,553 | 28 | 48 | 94 | 11,266 | 0 | 0 | 0 | 0 | 0 |
| val | 537 | 537 | 26 | 48 | 74 | 969 | 0 | 0 | 0 | 0 | 0 |
| test | 996 | 996 | 27 | 47 | 69 | 1,966 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| gle | 2,238 | 236 | 473 | 2,947 | 48.42% |
| und | 1,430 | 182 | 305 | 1,917 | 31.50% |
| yor | 333 | 64 | 89 | 486 | 7.99% |
| cat | 225 | 25 | 61 | 311 | 5.11% |
| lmo | 122 | 13 | 23 | 158 | 2.60% |
| vec | 46 | 4 | 8 | 58 | 0.95% |
| lao | 21 | 6 | 4 | 31 | 0.51% |
| roh | 27 | 1 | 2 | 30 | 0.49% |
| ast | 13 | 0 | 6 | 19 | 0.31% |
| gla | 14 | 2 | 2 | 18 | 0.30% |
| vep | 13 | 0 | 1 | 14 | 0.23% |
| hun | 9 | 0 | 4 | 13 | 0.21% |
| eml | 8 | 0 | 3 | 11 | 0.18% |
| cor | 9 | 0 | 0 | 9 | 0.15% |
| ces | 4 | 2 | 2 | 8 | 0.13% |
| spa | 5 | 1 | 1 | 7 | 0.12% |
| pam | 4 | 0 | 3 | 7 | 0.12% |
| eng | 7 | 0 | 0 | 7 | 0.12% |
| slk | 4 | 0 | 1 | 5 | 0.08% |
| vie | 2 | 0 | 1 | 3 | 0.05% |
| por | 3 | 0 | 0 | 3 | 0.05% |
| fra | 2 | 0 | 1 | 3 | 0.05% |
| ita | 2 | 1 | 0 | 3 | 0.05% |
| kom | 2 | 0 | 0 | 2 | 0.03% |
| oci | 1 | 0 | 1 | 2 | 0.03% |
| glg | 1 | 0 | 1 | 2 | 0.03% |
| lez | 1 | 0 | 1 | 2 | 0.03% |
| mlg | 0 | 0 | 2 | 2 | 0.03% |
| mrj | 1 | 0 | 0 | 1 | 0.02% |
| als | 1 | 0 | 0 | 1 | 0.02% |
| cbk | 1 | 0 | 0 | 1 | 0.02% |
| ind | 1 | 0 | 0 | 1 | 0.02% |
| war | 1 | 0 | 0 | 1 | 0.02% |
| rus | 1 | 0 | 0 | 1 | 0.02% |
| oss | 1 | 0 | 0 | 1 | 0.02% |
| cos | 0 | 0 | 1 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 996 |
| test_unique_inputs | 996 |
| train_records | 4,553 |
| train_unique_inputs | 4,553 |
| val_records | 537 |
| val_unique_inputs | 537 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0557 | 11,266 / 969 | person 32%→37%, location 33%→29%, date 15%→13%, organization 20%→20% |
| entities (per mention) | train / test | 0.0711 | 11,266 / 1,966 | location 33%→26%, person 32%→38%, date 15%→16%, organization 20%→20% |
| entities (per mention) | val / test | 0.0335 | 969 / 1,966 | location 29%→26%, date 13%→16%, person 37%→38%, organization 20%→20% |


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
| location | 3,716 | 32.98% |
| person | 3,607 | 32.02% |
| organization | 2,265 | 20.10% |
| date | 1,678 | 14.89% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 363 | 37.46% |
| location | 282 | 29.10% |
| organization | 196 | 20.23% |
| date | 128 | 13.21% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 756 | 38.45% |
| location | 510 | 25.94% |
| organization | 394 | 20.04% |
| date | 306 | 15.56% |

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
