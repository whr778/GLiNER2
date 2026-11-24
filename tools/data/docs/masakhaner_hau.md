# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_hau | train | 4,405 | 4,405 | 9,207 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_hau | val | 541 | 541 | 1,139 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_hau | test | 1,423 | 1,423 | 4,020 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_hau

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,405 | 4,405 | 28 | 48 | 94 | 9,207 | 0 | 0 | 0 | 0 | 0 |
| val | 541 | 541 | 29 | 43 | 76 | 1,139 | 0 | 0 | 0 | 0 | 0 |
| test | 1,423 | 1,423 | 28 | 48 | 87 | 4,020 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 3,109 | 365 | 1,138 | 4,612 | 72.41% |
| swa | 456 | 47 | 98 | 601 | 9.44% |
| epo | 223 | 35 | 39 | 297 | 4.66% |
| ido | 174 | 24 | 36 | 234 | 3.67% |
| que | 128 | 22 | 32 | 182 | 2.86% |
| bre | 55 | 6 | 20 | 81 | 1.27% |
| jav | 48 | 3 | 20 | 71 | 1.11% |
| fil | 27 | 7 | 5 | 39 | 0.61% |
| hat | 26 | 7 | 4 | 37 | 0.58% |
| roh | 18 | 4 | 3 | 25 | 0.39% |
| ind | 18 | 0 | 3 | 21 | 0.33% |
| eng | 16 | 1 | 4 | 21 | 0.33% |
| cor | 13 | 2 | 2 | 17 | 0.27% |
| nld | 6 | 8 | 1 | 15 | 0.24% |
| ceb | 7 | 1 | 2 | 10 | 0.16% |
| sun | 8 | 0 | 2 | 10 | 0.16% |
| lat | 6 | 2 | 1 | 9 | 0.14% |
| ita | 5 | 0 | 2 | 7 | 0.11% |
| ilo | 4 | 1 | 1 | 6 | 0.09% |
| grn | 3 | 1 | 1 | 5 | 0.08% |
| fin | 4 | 1 | 0 | 5 | 0.08% |
| lmo | 5 | 0 | 0 | 5 | 0.08% |
| eus | 3 | 0 | 2 | 5 | 0.08% |
| tur | 4 | 0 | 0 | 4 | 0.06% |
| war | 2 | 0 | 1 | 3 | 0.05% |
| cbk | 3 | 0 | 0 | 3 | 0.05% |
| vol | 2 | 1 | 0 | 3 | 0.05% |
| pms | 3 | 0 | 0 | 3 | 0.05% |
| ile | 3 | 0 | 0 | 3 | 0.05% |
| hsb | 2 | 0 | 0 | 2 | 0.03% |
| deu | 2 | 0 | 0 | 2 | 0.03% |
| spa | 1 | 0 | 1 | 2 | 0.03% |
| fra | 2 | 0 | 0 | 2 | 0.03% |
| swe | 2 | 0 | 0 | 2 | 0.03% |
| kur | 2 | 0 | 0 | 2 | 0.03% |
| oci | 1 | 0 | 1 | 2 | 0.03% |
| bcl | 1 | 1 | 0 | 2 | 0.03% |
| nap | 2 | 0 | 0 | 2 | 0.03% |
| eml | 1 | 0 | 0 | 1 | 0.02% |
| slv | 1 | 0 | 0 | 1 | 0.02% |
| lav | 1 | 0 | 0 | 1 | 0.02% |
| dan | 1 | 0 | 0 | 1 | 0.02% |
| uzb | 1 | 0 | 0 | 1 | 0.02% |
| glv | 1 | 0 | 0 | 1 | 0.02% |
| por | 1 | 0 | 0 | 1 | 0.02% |
| ces | 1 | 0 | 0 | 1 | 0.02% |
| pol | 1 | 0 | 0 | 1 | 0.02% |
| bar | 1 | 0 | 0 | 1 | 0.02% |
| scn | 1 | 0 | 0 | 1 | 0.02% |
| als | 0 | 1 | 0 | 1 | 0.02% |
| srd | 0 | 1 | 0 | 1 | 0.02% |
| srp | 0 | 0 | 1 | 1 | 0.02% |
| lim | 0 | 0 | 1 | 1 | 0.02% |
| oss | 0 | 0 | 1 | 1 | 0.02% |
| min | 0 | 0 | 1 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,423 |
| test_unique_inputs | 1,423 |
| train_records | 4,405 |
| train_unique_inputs | 4,405 |
| val_records | 541 |
| val_unique_inputs | 541 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0140 | 9,207 / 1,139 | location 30%→28%, date 15%→15%, organization 20%→20%, person 36%→36% |
| entities (per mention) | train / test | 0.1315 | 9,207 / 4,020 | organization 20%→29%, person 36%→29%, location 30%→23%, date 15%→18% |
| entities (per mention) | val / test | 0.1211 | 1,139 / 4,020 | organization 20%→29%, person 36%→29%, location 28%→23%, date 15%→18% |


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
| person | 3,314 | 35.99% |
| location | 2,748 | 29.85% |
| organization | 1,800 | 19.55% |
| date | 1,345 | 14.61% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 414 | 36.35% |
| location | 324 | 28.45% |
| organization | 227 | 19.93% |
| date | 174 | 15.28% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 1,175 | 29.23% |
| organization | 1,160 | 28.86% |
| location | 943 | 23.46% |
| date | 742 | 18.46% |

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
