# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_sna | train | 5,310 | 5,310 | 13,070 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_sna | val | 782 | 782 | 2,027 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_sna | test | 1,510 | 1,510 | 3,827 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_sna

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 5,310 | 5,310 | 23 | 35 | 58 | 13,070 | 0 | 0 | 0 | 0 | 0 |
| val | 782 | 782 | 23 | 37 | 53 | 2,027 | 0 | 0 | 0 | 0 | 0 |
| test | 1,510 | 1,510 | 22 | 36 | 72 | 3,827 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 5,019 | 737 | 1,431 | 7,187 | 94.54% |
| epo | 95 | 12 | 22 | 129 | 1.70% |
| swa | 41 | 15 | 16 | 72 | 0.95% |
| eng | 34 | 6 | 10 | 50 | 0.66% |
| fin | 25 | 2 | 10 | 37 | 0.49% |
| tur | 30 | 1 | 5 | 36 | 0.47% |
| ceb | 11 | 4 | 3 | 18 | 0.24% |
| srp | 7 | 3 | 2 | 12 | 0.16% |
| pol | 6 | 0 | 1 | 7 | 0.09% |
| vol | 6 | 0 | 1 | 7 | 0.09% |
| fil | 4 | 1 | 1 | 6 | 0.08% |
| eus | 6 | 0 | 0 | 6 | 0.08% |
| slv | 3 | 0 | 1 | 4 | 0.05% |
| cor | 2 | 0 | 1 | 3 | 0.04% |
| nld | 1 | 0 | 2 | 3 | 0.04% |
| sun | 2 | 0 | 0 | 2 | 0.03% |
| ido | 1 | 0 | 1 | 2 | 0.03% |
| fry | 2 | 0 | 0 | 2 | 0.03% |
| ita | 2 | 0 | 0 | 2 | 0.03% |
| roh | 2 | 0 | 0 | 2 | 0.03% |
| jbo | 1 | 0 | 1 | 2 | 0.03% |
| dan | 1 | 0 | 1 | 2 | 0.03% |
| lav | 2 | 0 | 0 | 2 | 0.03% |
| war | 1 | 0 | 1 | 2 | 0.03% |
| deu | 1 | 1 | 0 | 2 | 0.03% |
| uzb | 1 | 0 | 0 | 1 | 0.01% |
| ilo | 1 | 0 | 0 | 1 | 0.01% |
| pam | 1 | 0 | 0 | 1 | 0.01% |
| hrv | 1 | 0 | 0 | 1 | 0.01% |
| spa | 1 | 0 | 0 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,510 |
| test_unique_inputs | 1,510 |
| train_records | 5,310 |
| train_unique_inputs | 5,310 |
| val_records | 782 |
| val_unique_inputs | 782 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0208 | 13,070 / 2,027 | organization 34%→32%, person 34%→36%, date 12%→13%, location 20%→20% |
| entities (per mention) | train / test | 0.0393 | 13,070 / 3,827 | organization 34%→30%, person 34%→37%, location 20%→22%, date 12%→12% |
| entities (per mention) | val / test | 0.0225 | 2,027 / 3,827 | organization 32%→30%, location 20%→22%, person 36%→37%, date 13%→12% |


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
| person | 4,495 | 34.39% |
| organization | 4,387 | 33.57% |
| location | 2,655 | 20.31% |
| date | 1,533 | 11.73% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 723 | 35.67% |
| organization | 639 | 31.52% |
| location | 411 | 20.28% |
| date | 254 | 12.53% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 1,401 | 36.61% |
| organization | 1,134 | 29.63% |
| location | 826 | 21.58% |
| date | 466 | 12.18% |

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
