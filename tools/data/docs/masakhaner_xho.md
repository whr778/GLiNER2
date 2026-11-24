# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_xho | train | 2,905 | 2,905 | 6,188 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_xho | val | 683 | 683 | 1,989 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_xho | test | 1,172 | 1,172 | 2,797 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_xho

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,905 | 2,905 | 15 | 27 | 63 | 6,188 | 0 | 0 | 0 | 0 | 0 |
| val | 683 | 683 | 17 | 29 | 52 | 1,989 | 0 | 0 | 0 | 0 | 0 |
| test | 1,172 | 1,172 | 17 | 27 | 50 | 2,797 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 2,548 | 602 | 1,000 | 4,150 | 87.18% |
| war | 93 | 29 | 41 | 163 | 3.42% |
| epo | 63 | 17 | 28 | 108 | 2.27% |
| eng | 29 | 4 | 20 | 53 | 1.11% |
| fil | 33 | 5 | 14 | 52 | 1.09% |
| jav | 16 | 4 | 7 | 27 | 0.57% |
| eus | 19 | 0 | 4 | 23 | 0.48% |
| swa | 10 | 0 | 11 | 21 | 0.44% |
| fin | 14 | 3 | 4 | 21 | 0.44% |
| pol | 9 | 1 | 9 | 19 | 0.40% |
| ilo | 12 | 1 | 4 | 17 | 0.36% |
| ido | 8 | 0 | 3 | 11 | 0.23% |
| ceb | 4 | 0 | 6 | 10 | 0.21% |
| lat | 2 | 1 | 6 | 9 | 0.19% |
| nld | 7 | 1 | 0 | 8 | 0.17% |
| vol | 4 | 0 | 3 | 7 | 0.15% |
| bre | 3 | 1 | 3 | 7 | 0.15% |
| cor | 4 | 0 | 1 | 5 | 0.11% |
| sun | 3 | 2 | 0 | 5 | 0.11% |
| min | 2 | 0 | 1 | 3 | 0.06% |
| cym | 2 | 0 | 1 | 3 | 0.06% |
| scn | 2 | 1 | 0 | 3 | 0.06% |
| tur | 2 | 1 | 0 | 3 | 0.06% |
| jbo | 1 | 0 | 2 | 3 | 0.06% |
| ile | 1 | 0 | 1 | 2 | 0.04% |
| uzb | 2 | 0 | 0 | 2 | 0.04% |
| spa | 1 | 1 | 0 | 2 | 0.04% |
| por | 1 | 0 | 1 | 2 | 0.04% |
| que | 1 | 0 | 1 | 2 | 0.04% |
| deu | 1 | 1 | 0 | 2 | 0.04% |
| srp | 0 | 2 | 0 | 2 | 0.04% |
| als | 0 | 2 | 0 | 2 | 0.04% |
| hsb | 0 | 2 | 0 | 2 | 0.04% |
| nno | 1 | 0 | 0 | 1 | 0.02% |
| lav | 1 | 0 | 0 | 1 | 0.02% |
| swe | 1 | 0 | 0 | 1 | 0.02% |
| ina | 1 | 0 | 0 | 1 | 0.02% |
| sco | 1 | 0 | 0 | 1 | 0.02% |
| slv | 1 | 0 | 0 | 1 | 0.02% |
| ind | 1 | 0 | 0 | 1 | 0.02% |
| ron | 1 | 0 | 0 | 1 | 0.02% |
| ita | 0 | 1 | 0 | 1 | 0.02% |
| lmo | 0 | 1 | 0 | 1 | 0.02% |
| srd | 0 | 0 | 1 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,172 |
| test_unique_inputs | 1,172 |
| train_records | 2,905 |
| train_unique_inputs | 2,905 |
| val_records | 683 |
| val_unique_inputs | 683 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0605 | 6,188 / 1,989 | organization 25%→31%, person 47%→44%, location 16%→14%, date 11%→12% |
| entities (per mention) | train / test | 0.0658 | 6,188 / 2,797 | person 47%→41%, organization 25%→29%, location 16%→19%, date 11%→10% |
| entities (per mention) | val / test | 0.0511 | 1,989 / 2,797 | location 14%→19%, person 44%→41%, date 12%→10%, organization 31%→29% |


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
| person | 2,935 | 47.43% |
| organization | 1,573 | 25.42% |
| location | 1,011 | 16.34% |
| date | 669 | 10.81% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 872 | 43.84% |
| organization | 608 | 30.57% |
| location | 276 | 13.88% |
| date | 233 | 11.71% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 1,155 | 41.29% |
| organization | 821 | 29.35% |
| location | 531 | 18.98% |
| date | 290 | 10.37% |

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
