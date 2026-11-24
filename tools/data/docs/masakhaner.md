# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner | train | 66,684 | 66,684 | 149,898 | 0 | 0 | 0 | 0 | 0 |
| masakhaner | val | 9,919 | 9,919 | 22,789 | 0 | 0 | 0 | 0 | 0 |
| masakhaner | test | 20,384 | 20,384 | 46,492 | 0 | 0 | 0 | 0 | 0 |

## masakhaner

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 66,684 | 66,684 | 26 | 49 | 110 | 149,898 | 0 | 0 | 0 | 0 | 0 |
| val | 9,919 | 9,919 | 25 | 47 | 100 | 22,789 | 0 | 0 | 0 | 0 | 0 |
| test | 20,384 | 20,384 | 25 | 47 | 104 | 46,492 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 43,205 | 6,570 | 13,429 | 63,204 | 65.17% |
| swa | 7,845 | 1,115 | 2,269 | 11,229 | 11.58% |
| gle | 2,427 | 261 | 528 | 3,216 | 3.32% |
| epo | 2,348 | 257 | 587 | 3,192 | 3.29% |
| eng | 1,975 | 307 | 577 | 2,859 | 2.95% |
| vie | 1,072 | 149 | 406 | 1,627 | 1.68% |
| war | 810 | 138 | 303 | 1,251 | 1.29% |
| sun | 566 | 90 | 184 | 840 | 0.87% |
| aze | 508 | 57 | 176 | 741 | 0.76% |
| fil | 421 | 66 | 130 | 617 | 0.64% |
| yor | 352 | 65 | 111 | 528 | 0.54% |
| eml | 285 | 65 | 137 | 487 | 0.50% |
| cor | 321 | 43 | 88 | 452 | 0.47% |
| ido | 297 | 49 | 68 | 414 | 0.43% |
| cat | 282 | 25 | 80 | 387 | 0.40% |
| ita | 214 | 43 | 84 | 341 | 0.35% |
| lmo | 231 | 29 | 49 | 309 | 0.32% |
| cym | 190 | 32 | 72 | 294 | 0.30% |
| fin | 174 | 42 | 69 | 285 | 0.29% |
| ceb | 190 | 26 | 66 | 282 | 0.29% |
| bre | 187 | 19 | 61 | 267 | 0.28% |
| que | 183 | 30 | 45 | 258 | 0.27% |
| est | 158 | 28 | 72 | 258 | 0.27% |
| jav | 150 | 28 | 51 | 229 | 0.24% |
| fra | 126 | 21 | 60 | 207 | 0.21% |
| ilo | 98 | 16 | 37 | 151 | 0.16% |
| tat | 78 | 15 | 51 | 144 | 0.15% |
| roh | 98 | 11 | 29 | 138 | 0.14% |
| eus | 91 | 17 | 29 | 137 | 0.14% |
| hat | 100 | 19 | 17 | 136 | 0.14% |
| jbo | 85 | 16 | 29 | 130 | 0.13% |
| vec | 98 | 6 | 12 | 116 | 0.12% |
| tur | 72 | 26 | 16 | 114 | 0.12% |
| lat | 60 | 15 | 25 | 100 | 0.10% |
| por | 72 | 11 | 13 | 96 | 0.10% |
| som | 66 | 8 | 21 | 95 | 0.10% |
| ast | 62 | 7 | 24 | 93 | 0.10% |
| pol | 52 | 11 | 23 | 86 | 0.09% |
| scn | 55 | 11 | 18 | 84 | 0.09% |
| grn | 50 | 5 | 16 | 71 | 0.07% |
| fry | 46 | 5 | 18 | 69 | 0.07% |
| nap | 36 | 10 | 13 | 59 | 0.06% |
| lao | 34 | 7 | 7 | 48 | 0.05% |
| kom | 38 | 2 | 7 | 47 | 0.05% |
| cbk | 39 | 2 | 5 | 46 | 0.05% |
| ces | 26 | 8 | 12 | 46 | 0.05% |
| kur | 25 | 5 | 14 | 44 | 0.05% |
| bih | 22 | 7 | 14 | 43 | 0.04% |
| wln | 27 | 5 | 10 | 42 | 0.04% |
| spa | 29 | 6 | 6 | 41 | 0.04% |
| vep | 28 | 3 | 10 | 41 | 0.04% |
| vol | 29 | 5 | 6 | 40 | 0.04% |
| deu | 29 | 4 | 7 | 40 | 0.04% |
| srp | 25 | 6 | 9 | 40 | 0.04% |
| ind | 29 | 0 | 10 | 39 | 0.04% |
| hun | 27 | 2 | 9 | 38 | 0.04% |
| ile | 23 | 7 | 7 | 37 | 0.04% |
| ltz | 22 | 1 | 14 | 37 | 0.04% |
| pms | 25 | 5 | 7 | 37 | 0.04% |
| mlt | 22 | 4 | 8 | 34 | 0.04% |
| nld | 18 | 9 | 7 | 34 | 0.04% |
| gla | 28 | 2 | 2 | 32 | 0.03% |
| ina | 21 | 3 | 7 | 31 | 0.03% |
| hye | 22 | 4 | 4 | 30 | 0.03% |
| sah | 20 | 4 | 4 | 28 | 0.03% |
| bak | 17 | 2 | 9 | 28 | 0.03% |
| glg | 21 | 4 | 1 | 26 | 0.03% |
| hrv | 11 | 0 | 12 | 23 | 0.02% |
| mlg | 10 | 3 | 8 | 21 | 0.02% |
| diq | 19 | 0 | 1 | 20 | 0.02% |
| srd | 12 | 5 | 3 | 20 | 0.02% |
| glv | 9 | 5 | 5 | 19 | 0.02% |
| als | 10 | 5 | 4 | 19 | 0.02% |
| hsb | 10 | 5 | 4 | 19 | 0.02% |
| pam | 14 | 1 | 3 | 18 | 0.02% |
| oci | 11 | 2 | 3 | 16 | 0.02% |
| gom | 10 | 2 | 3 | 15 | 0.02% |
| sco | 10 | 2 | 2 | 14 | 0.01% |
| uzb | 11 | 1 | 1 | 13 | 0.01% |
| swe | 10 | 0 | 3 | 13 | 0.01% |
| slk | 8 | 0 | 5 | 13 | 0.01% |
| bcl | 6 | 1 | 4 | 11 | 0.01% |
| new | 11 | 0 | 0 | 11 | 0.01% |
| mrj | 8 | 0 | 2 | 10 | 0.01% |
| rus | 8 | 2 | 0 | 10 | 0.01% |
| mon | 7 | 1 | 1 | 9 | 0.01% |
| nds | 6 | 3 | 0 | 9 | 0.01% |
| slv | 5 | 0 | 4 | 9 | 0.01% |
| tyv | 8 | 0 | 0 | 8 | 0.01% |
| afr | 7 | 0 | 1 | 8 | 0.01% |
| min | 5 | 0 | 3 | 8 | 0.01% |
| nah | 6 | 0 | 2 | 8 | 0.01% |
| lez | 5 | 0 | 2 | 7 | 0.01% |
| xal | 3 | 0 | 4 | 7 | 0.01% |
| lim | 5 | 0 | 2 | 7 | 0.01% |
| cos | 6 | 0 | 1 | 7 | 0.01% |
| mhr | 4 | 1 | 1 | 6 | 0.01% |
| oss | 4 | 1 | 1 | 6 | 0.01% |
| lav | 5 | 0 | 0 | 5 | 0.01% |
| ron | 3 | 0 | 2 | 5 | 0.01% |
| sqi | 3 | 1 | 1 | 5 | 0.01% |
| pnb | 2 | 1 | 1 | 4 | 0.00% |
| mwl | 4 | 0 | 0 | 4 | 0.00% |
| dan | 3 | 0 | 1 | 4 | 0.00% |
| tuk | 2 | 1 | 0 | 3 | 0.00% |
| nno | 3 | 0 | 0 | 3 | 0.00% |
| msa | 1 | 0 | 1 | 2 | 0.00% |
| myv | 1 | 1 | 0 | 2 | 0.00% |
| ava | 2 | 0 | 0 | 2 | 0.00% |
| urd | 1 | 1 | 0 | 2 | 0.00% |
| vls | 1 | 1 | 0 | 2 | 0.00% |
| krc | 0 | 0 | 2 | 2 | 0.00% |
| tgk | 1 | 0 | 0 | 1 | 0.00% |
| tel | 1 | 0 | 0 | 1 | 0.00% |
| mar | 1 | 0 | 0 | 1 | 0.00% |
| kan | 1 | 0 | 0 | 1 | 0.00% |
| yid | 1 | 0 | 0 | 1 | 0.00% |
| asm | 1 | 0 | 0 | 1 | 0.00% |
| arg | 1 | 0 | 0 | 1 | 0.00% |
| bar | 1 | 0 | 0 | 1 | 0.00% |
| nor | 1 | 0 | 0 | 1 | 0.00% |
| pfl | 1 | 0 | 0 | 1 | 0.00% |
| che | 1 | 0 | 0 | 1 | 0.00% |
| div | 1 | 0 | 0 | 1 | 0.00% |
| wuu | 0 | 1 | 0 | 1 | 0.00% |
| azb | 0 | 1 | 0 | 1 | 0.00% |
| isl | 0 | 0 | 1 | 1 | 0.00% |
| bos | 0 | 0 | 1 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 20,384 |
| test_unique_inputs | 20,384 |
| train_records | 66,684 |
| train_unique_inputs | 66,684 |
| val_records | 9,919 |
| val_unique_inputs | 9,919 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0197 | 149,898 / 22,789 | location 30%→28%, person 33%→34%, organization 22%→23%, date 15%→15% |
| entities (per mention) | train / test | 0.0070 | 149,898 / 46,492 | person 33%→32%, location 30%→31%, date 15%→15%, organization 22%→22% |
| entities (per mention) | val / test | 0.0235 | 22,789 / 46,492 | location 28%→31%, person 34%→32%, organization 23%→22%, date 15%→15% |


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
| person | 48,947 | 32.65% |
| location | 45,470 | 30.33% |
| organization | 32,865 | 21.92% |
| date | 22,616 | 15.09% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 7,662 | 33.62% |
| location | 6,465 | 28.37% |
| organization | 5,150 | 22.60% |
| date | 3,512 | 15.41% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 14,969 | 32.20% |
| location | 14,282 | 30.72% |
| organization | 10,079 | 21.68% |
| date | 7,162 | 15.40% |

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
