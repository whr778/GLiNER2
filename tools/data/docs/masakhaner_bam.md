# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_bam | train | 2,048 | 2,048 | 3,589 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_bam | val | 304 | 304 | 619 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_bam | test | 681 | 681 | 1,283 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_bam

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,048 | 2,048 | 29 | 54 | 78 | 3,589 | 0 | 0 | 0 | 0 | 0 |
| val | 304 | 304 | 34 | 56 | 82 | 619 | 0 | 0 | 0 | 0 | 0 |
| test | 681 | 681 | 31 | 56 | 78 | 1,283 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 1,223 | 177 | 373 | 1,773 | 58.46% |
| sun | 251 | 33 | 97 | 381 | 12.56% |
| cym | 121 | 21 | 52 | 194 | 6.40% |
| cor | 118 | 22 | 30 | 170 | 5.61% |
| jbo | 56 | 11 | 17 | 84 | 2.77% |
| epo | 47 | 7 | 13 | 67 | 2.21% |
| fil | 27 | 2 | 11 | 40 | 1.32% |
| bih | 16 | 7 | 11 | 34 | 1.12% |
| ilo | 24 | 1 | 6 | 31 | 1.02% |
| que | 13 | 3 | 8 | 24 | 0.79% |
| ceb | 14 | 1 | 6 | 21 | 0.69% |
| kur | 9 | 4 | 7 | 20 | 0.66% |
| war | 12 | 1 | 6 | 19 | 0.63% |
| swa | 15 | 0 | 4 | 19 | 0.63% |
| bre | 14 | 1 | 1 | 16 | 0.53% |
| hat | 9 | 1 | 2 | 12 | 0.40% |
| fin | 8 | 1 | 3 | 12 | 0.40% |
| grn | 8 | 1 | 2 | 11 | 0.36% |
| lat | 9 | 0 | 1 | 10 | 0.33% |
| som | 6 | 1 | 3 | 10 | 0.33% |
| ile | 3 | 1 | 3 | 7 | 0.23% |
| ido | 5 | 1 | 1 | 7 | 0.23% |
| bcl | 2 | 0 | 3 | 5 | 0.16% |
| vol | 4 | 0 | 1 | 5 | 0.16% |
| fry | 2 | 1 | 2 | 5 | 0.16% |
| spa | 1 | 3 | 0 | 4 | 0.13% |
| diq | 3 | 0 | 0 | 3 | 0.10% |
| cbk | 3 | 0 | 0 | 3 | 0.10% |
| vep | 3 | 0 | 0 | 3 | 0.10% |
| ita | 1 | 0 | 2 | 3 | 0.10% |
| jav | 2 | 0 | 0 | 2 | 0.07% |
| roh | 2 | 0 | 0 | 2 | 0.07% |
| uzb | 1 | 0 | 1 | 2 | 0.07% |
| pnb | 1 | 0 | 1 | 2 | 0.07% |
| hun | 2 | 0 | 0 | 2 | 0.07% |
| mlt | 2 | 0 | 0 | 2 | 0.07% |
| gla | 2 | 0 | 0 | 2 | 0.07% |
| msa | 1 | 0 | 1 | 2 | 0.07% |
| scn | 0 | 0 | 2 | 2 | 0.07% |
| srp | 0 | 0 | 2 | 2 | 0.07% |
| tur | 0 | 0 | 2 | 2 | 0.07% |
| ind | 1 | 0 | 0 | 1 | 0.03% |
| glv | 1 | 0 | 0 | 1 | 0.03% |
| mlg | 1 | 0 | 0 | 1 | 0.03% |
| tgk | 1 | 0 | 0 | 1 | 0.03% |
| eng | 1 | 0 | 0 | 1 | 0.03% |
| wln | 1 | 0 | 0 | 1 | 0.03% |
| pol | 1 | 0 | 0 | 1 | 0.03% |
| tel | 1 | 0 | 0 | 1 | 0.03% |
| als | 0 | 1 | 0 | 1 | 0.03% |
| eus | 0 | 1 | 0 | 1 | 0.03% |
| wuu | 0 | 1 | 0 | 1 | 0.03% |
| mon | 0 | 0 | 1 | 1 | 0.03% |
| min | 0 | 0 | 1 | 1 | 0.03% |
| hsb | 0 | 0 | 1 | 1 | 0.03% |
| ron | 0 | 0 | 1 | 1 | 0.03% |
| ina | 0 | 0 | 1 | 1 | 0.03% |
| nld | 0 | 0 | 1 | 1 | 0.03% |
| pms | 0 | 0 | 1 | 1 | 0.03% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 681 |
| test_unique_inputs | 681 |
| train_records | 2,048 |
| train_unique_inputs | 2,048 |
| val_records | 304 |
| val_unique_inputs | 304 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0551 | 3,589 / 619 | date 15%→20%, person 40%→37%, organization 6%→4%, location 40%→39% |
| entities (per mention) | train / test | 0.0411 | 3,589 / 1,283 | person 40%→36%, location 40%→42%, organization 6%→7%, date 15%→15% |
| entities (per mention) | val / test | 0.0621 | 619 / 1,283 | date 20%→15%, location 39%→42%, organization 4%→7%, person 37%→36% |


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
| person | 1,426 | 39.73% |
| location | 1,424 | 39.68% |
| date | 527 | 14.68% |
| organization | 212 | 5.91% |

#### val

| label | count | share |
| --- | --- | --- |
| location | 240 | 38.77% |
| person | 228 | 36.83% |
| date | 125 | 20.19% |
| organization | 26 | 4.20% |

#### test

| label | count | share |
| --- | --- | --- |
| location | 541 | 42.17% |
| person | 457 | 35.62% |
| date | 195 | 15.20% |
| organization | 90 | 7.01% |

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
