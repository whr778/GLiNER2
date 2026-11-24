# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_zul | train | 2,660 | 2,657 | 5,426 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_zul | val | 530 | 530 | 1,103 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_zul | test | 998 | 998 | 1,906 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_zul

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,660 | 2,657 | 17 | 29 | 76 | 5,426 | 0 | 0 | 0 | 0 | 0 |
| val | 530 | 530 | 17 | 27 | 74 | 1,103 | 0 | 0 | 0 | 0 | 0 |
| test | 998 | 998 | 17 | 28 | 47 | 1,906 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 2,215 | 431 | 854 | 3,500 | 83.57% |
| war | 92 | 15 | 26 | 133 | 3.18% |
| fil | 59 | 14 | 15 | 88 | 2.10% |
| epo | 49 | 14 | 18 | 81 | 1.93% |
| jav | 42 | 9 | 17 | 68 | 1.62% |
| ido | 40 | 12 | 12 | 64 | 1.53% |
| eng | 31 | 6 | 2 | 39 | 0.93% |
| lat | 14 | 9 | 6 | 29 | 0.69% |
| eus | 13 | 5 | 7 | 25 | 0.60% |
| ilo | 15 | 2 | 7 | 24 | 0.57% |
| por | 15 | 3 | 4 | 22 | 0.53% |
| fin | 12 | 2 | 7 | 21 | 0.50% |
| bre | 16 | 1 | 3 | 20 | 0.48% |
| swa | 7 | 1 | 7 | 15 | 0.36% |
| pol | 7 | 1 | 2 | 10 | 0.24% |
| sun | 5 | 0 | 2 | 7 | 0.17% |
| tur | 3 | 0 | 3 | 6 | 0.14% |
| srp | 4 | 0 | 1 | 5 | 0.12% |
| ita | 2 | 1 | 1 | 4 | 0.10% |
| ceb | 2 | 1 | 1 | 4 | 0.10% |
| vol | 4 | 0 | 0 | 4 | 0.10% |
| uzb | 3 | 1 | 0 | 4 | 0.10% |
| deu | 2 | 1 | 0 | 3 | 0.07% |
| jbo | 1 | 1 | 1 | 3 | 0.07% |
| cor | 1 | 0 | 0 | 1 | 0.02% |
| kur | 1 | 0 | 0 | 1 | 0.02% |
| fra | 1 | 0 | 0 | 1 | 0.02% |
| hsb | 1 | 0 | 0 | 1 | 0.02% |
| cym | 1 | 0 | 0 | 1 | 0.02% |
| ina | 1 | 0 | 0 | 1 | 0.02% |
| ron | 1 | 0 | 0 | 1 | 0.02% |
| hun | 0 | 0 | 1 | 1 | 0.02% |
| bos | 0 | 0 | 1 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 998 |
| test_unique_inputs | 998 |
| train_records | 2,660 |
| train_unique_inputs | 2,657 |
| val_records | 530 |
| val_unique_inputs | 530 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.2472 WARN | 5,426 / 1,103 | person 31%→55%, organization 41%→24%, location 16%→9%, date 13%→13% |
| entities (per mention) | train / test | 0.2145 WARN | 5,426 / 1,906 | organization 41%→19%, person 31%→46%, date 13%→17%, location 16%→18% |
| entities (per mention) | val / test | 0.1255 | 1,103 / 1,906 | location 9%→18%, person 55%→46%, organization 24%→19%, date 13%→17% |


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
| organization | 2,206 | 40.66% |
| person | 1,658 | 30.56% |
| location | 877 | 16.16% |
| date | 685 | 12.62% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 603 | 54.67% |
| organization | 260 | 23.57% |
| date | 146 | 13.24% |
| location | 94 | 8.52% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 886 | 46.48% |
| organization | 366 | 19.20% |
| location | 334 | 17.52% |
| date | 320 | 16.79% |

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
