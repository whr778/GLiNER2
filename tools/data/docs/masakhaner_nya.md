# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_nya | train | 4,304 | 4,304 | 10,915 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_nya | val | 598 | 598 | 1,601 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_nya | test | 1,307 | 1,307 | 3,325 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_nya

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,304 | 4,304 | 30 | 50 | 72 | 10,915 | 0 | 0 | 0 | 0 | 0 |
| val | 598 | 598 | 30 | 49 | 63 | 1,601 | 0 | 0 | 0 | 0 | 0 |
| test | 1,307 | 1,307 | 29 | 48 | 62 | 3,325 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 2,879 | 396 | 884 | 4,159 | 66.98% |
| swa | 1,097 | 162 | 342 | 1,601 | 25.79% |
| epo | 147 | 19 | 40 | 206 | 3.32% |
| eng | 51 | 1 | 5 | 57 | 0.92% |
| war | 40 | 4 | 12 | 56 | 0.90% |
| fil | 31 | 3 | 12 | 46 | 0.74% |
| eus | 18 | 7 | 5 | 30 | 0.48% |
| ceb | 9 | 0 | 2 | 11 | 0.18% |
| pol | 5 | 0 | 3 | 8 | 0.13% |
| jav | 6 | 1 | 0 | 7 | 0.11% |
| ido | 4 | 1 | 1 | 6 | 0.10% |
| ilo | 3 | 0 | 1 | 4 | 0.06% |
| ind | 3 | 0 | 0 | 3 | 0.05% |
| ita | 2 | 0 | 0 | 2 | 0.03% |
| mlg | 2 | 0 | 0 | 2 | 0.03% |
| fin | 1 | 1 | 0 | 2 | 0.03% |
| hrv | 2 | 0 | 0 | 2 | 0.03% |
| cor | 1 | 0 | 0 | 1 | 0.02% |
| lat | 1 | 0 | 0 | 1 | 0.02% |
| nld | 1 | 0 | 0 | 1 | 0.02% |
| sun | 1 | 0 | 0 | 1 | 0.02% |
| vol | 0 | 1 | 0 | 1 | 0.02% |
| por | 0 | 1 | 0 | 1 | 0.02% |
| que | 0 | 1 | 0 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,307 |
| test_unique_inputs | 1,307 |
| train_records | 4,304 |
| train_unique_inputs | 4,304 |
| val_records | 598 |
| val_unique_inputs | 598 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0500 | 10,915 / 1,601 | organization 17%→22%, person 44%→40%, date 13%→12%, location 26%→26% |
| entities (per mention) | train / test | 0.0513 | 10,915 / 3,325 | organization 17%→21%, location 26%→23%, person 44%→42%, date 13%→14% |
| entities (per mention) | val / test | 0.0448 | 1,601 / 3,325 | location 26%→23%, date 12%→14%, person 40%→42%, organization 22%→21% |


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
| person | 4,823 | 44.19% |
| location | 2,811 | 25.75% |
| organization | 1,879 | 17.21% |
| date | 1,402 | 12.84% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 640 | 39.98% |
| location | 414 | 25.86% |
| organization | 354 | 22.11% |
| date | 193 | 12.05% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 1,403 | 42.20% |
| location | 752 | 22.62% |
| organization | 694 | 20.87% |
| date | 476 | 14.32% |

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
